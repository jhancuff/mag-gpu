import sys
import random
import numpy as np
import cv2
import mss
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QTime, QEasingCurve

# Constants for the display dimensions, window size, and animation
DISPLAY_WIDTH = 3840
DISPLAY_HEIGHT = 2160
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 450
MIN_MOVE_DISTANCE = 200  # Minimum distance the window should move
MAGNIFICATION_FACTOR = 3  # Magnification factor
CAPTURE_WIDTH = WINDOW_WIDTH // MAGNIFICATION_FACTOR
CAPTURE_HEIGHT = WINDOW_HEIGHT // MAGNIFICATION_FACTOR
IDLE_TIMEOUT_MS = 3000  # Idle timeout in milliseconds

def capture_screen(x, y, width, height):
    with mss.mss() as sct:
        monitor = {"top": y, "left": x, "width": width, "height": height}
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # Convert BGRA to RGB

kernel_code = f"""
__global__ void resize_image(float *d_src, float *d_dst, int src_width, int src_height, int dst_width, int dst_height) {{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < dst_width && y < dst_height) {{
        int src_x = x / {MAGNIFICATION_FACTOR};
        int src_y = y / {MAGNIFICATION_FACTOR};
        for (int c = 0; c < 3; c++) {{
            d_dst[(y * dst_width + x) * 3 + c] = d_src[(src_y * src_width + src_x) * 3 + c];
        }}
    }}
}}
"""

mod = SourceModule(kernel_code)
resize_image = mod.get_function("resize_image")

d_src = cuda.mem_alloc(CAPTURE_WIDTH * CAPTURE_HEIGHT * 3 * 4)  # Capture size with 3 channels, float32
d_dst = cuda.mem_alloc(WINDOW_WIDTH * WINDOW_HEIGHT * 3 * 4)  # Window size with 3 channels, float32

def capture_and_magnify(x, y, width, height):
    region = capture_screen(x, y, width, height)
    src = region.astype(np.float32).flatten()
    dst = np.zeros((WINDOW_WIDTH, WINDOW_HEIGHT, 3), np.float32).flatten()

    cuda.memcpy_htod(d_src, src)

    block = (16, 16, 1)
    grid = (int(WINDOW_WIDTH / block[0]) + 1, int(WINDOW_HEIGHT / block[1]) + 1)

    resize_image(d_src, d_dst, np.int32(CAPTURE_WIDTH), np.int32(CAPTURE_HEIGHT), np.int32(WINDOW_WIDTH), np.int32(WINDOW_HEIGHT), block=block, grid=grid)

    cuda.memcpy_dtoh(dst, d_dst)

    magnified_region = dst.reshape((WINDOW_WIDTH, WINDOW_HEIGHT, 3)).astype(np.uint8)

    return magnified_region

class MagnifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        self.setGeometry(100, 100, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.setStyleSheet("background-color: black; border: 2px solid white;")  # Add a border

        self.label = QLabel(self)
        self.label.setGeometry(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)
        
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(300)  # Total duration of the animation in milliseconds
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)  # Easing curve for soft landing

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_magnifier)
        self.timer.start(16)  # Approximately 60 FPS

        self.last_mouse_pos = QCursor.pos()
        self.last_mouse_time = QTime.currentTime()

    def update_magnifier(self):
        current_mouse_pos = QCursor.pos()
        current_time = QTime.currentTime()

        if current_mouse_pos != self.last_mouse_pos:
            self.last_mouse_time = current_time
            self.last_mouse_pos = current_mouse_pos

        if self.last_mouse_time.msecsTo(current_time) < IDLE_TIMEOUT_MS:
            x, y = current_mouse_pos.x() - CAPTURE_WIDTH // 2, current_mouse_pos.y() - CAPTURE_HEIGHT // 2

            if self.animation.state() == QPropertyAnimation.Stopped and self.geometry().contains(current_mouse_pos):
                new_x, new_y = self.generate_new_position()
                new_geometry = QRect(new_x, new_y, WINDOW_WIDTH, WINDOW_HEIGHT)
                self.animation.stop()  # Stop any ongoing animation
                self.animation.setStartValue(self.geometry())
                self.animation.setEndValue(new_geometry)
                self.animation.start()

            region = capture_and_magnify(x, y, CAPTURE_WIDTH, CAPTURE_HEIGHT)

            image = QImage(region, WINDOW_WIDTH, WINDOW_HEIGHT, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.label.setPixmap(pixmap)
            self.update()
        else:
            self.label.clear()  # Clear the label if the mouse hasn't moved for the duration of the animation

    def generate_new_position(self):
        current_geometry = self.geometry()
        current_x = current_geometry.x()
        current_y = current_geometry.y()

        while True:
            new_x = random.randint(20, DISPLAY_WIDTH - WINDOW_WIDTH - 20)
            new_y = random.randint(20, DISPLAY_HEIGHT - WINDOW_HEIGHT - 20)
            if abs(new_x - current_x) > MIN_MOVE_DISTANCE or abs(new_y - current_y) > MIN_MOVE_DISTANCE:
                break

        return new_x, new_y

if __name__ == '__main__':
    app = QApplication(sys.argv)
    magnifier = MagnifierWindow()
    magnifier.show()
    sys.exit(app.exec_())
