import sys
import random
import numpy as np
import cv2
import mss
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFrame
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtCore import Qt, QTimer

# Constants for the display dimensions
DISPLAY_WIDTH = 3840
DISPLAY_HEIGHT = 2160

def capture_screen(x, y, width, height):
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Convert BGRA to RGB

kernel_code = """
__global__ void resize_image(float *d_src, float *d_dst, int src_width, int src_height, int dst_width, int dst_height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dst_width && y < dst_height) {
        int src_x = x / 3;
        int src_y = y / 3;
        for (int c = 0; c < 3; c++) {
            d_dst[(y * dst_width + x) * 3 + c] = d_src[(src_y * src_width + src_x) * 3 + c];
        }
    }
}
"""

mod = SourceModule(kernel_code)
resize_image = mod.get_function("resize_image")

d_src = cuda.mem_alloc(267 * 150 * 3 * 4)  # 267x150 image with 3 channels, float32
d_dst = cuda.mem_alloc(800 * 450 * 3 * 4)  # 800x450 image with 3 channels, float32

def capture_and_magnify(x, y, width, height, scale_factor):
    region = capture_screen(x, y, width, height)
    src = region.astype(np.float32).flatten()
    dst = np.zeros((800, 450, 3), np.float32).flatten()

    cuda.memcpy_htod(d_src, src)

    block = (16, 16, 1)
    grid = (int(800 / block[0]) + 1, int(450 / block[1]) + 1)

    resize_image(d_src, d_dst, np.int32(267), np.int32(150), np.int32(800), np.int32(450), block=block, grid=grid)

    cuda.memcpy_dtoh(dst, d_dst)

    magnified_region = dst.reshape((800, 450, 3)).astype(np.uint8)

    return magnified_region

class MagnifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(100, 100, 800, 450)

        self.setStyleSheet("background-color: black; border: 2px solid white;")  # Add a border

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, 800, 450)
        
        # Add a handle for manual movement outside the window
        self.handle = QFrame(self)
        self.handle.setGeometry(-20, 20, 20, 60)  # Positioned to the left of the window
        self.handle.setStyleSheet("background-color: gray; border: 2px solid white;")
        self.handle.setCursor(Qt.SizeAllCursor)
        self.handle.mousePressEvent = self.handleMousePress
        self.handle.mouseMoveEvent = self.handleMouseMove

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_magnifier)
        self.timer.start(16)  # Approximately 60 FPS

    def handleMousePress(self, event):
        if event.button() == Qt.LeftButton:
            self.handle_drag_start = event.globalPos() - self.frameGeometry().topLeft()

    def handleMouseMove(self, event):
        if event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.handle_drag_start)

    def update_magnifier(self):
        cursor_pos = QCursor.pos()
        x, y = cursor_pos.x() - 133, cursor_pos.y() - 75  # Adjust to center the 267x150 capture area

        if self.geometry().contains(cursor_pos):
            # Move to a random position within the display bounds
            new_x = random.randint(0, DISPLAY_WIDTH - 800)
            new_y = random.randint(0, DISPLAY_HEIGHT - 450)
            self.move(new_x, new_y)

        region = capture_and_magnify(x, y, 267, 150, 3)

        image = QImage(region, 800, 450, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    magnifier = MagnifierWindow()
    magnifier.show()
    sys.exit(app.exec_())
