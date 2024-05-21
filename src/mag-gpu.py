import sys
import random
import re
import numpy as np
import cv2
import mss
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtGui import QPixmap, QImage, QCursor, QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QRect, QTime, QEasingCurve
from pynput import keyboard

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

# Check if the allocation sizes are within a reasonable range
try:
    d_src = cuda.mem_alloc(CAPTURE_WIDTH * CAPTURE_HEIGHT * 3 * 4)  # Capture size with 3 channels, float32
    d_dst = cuda.mem_alloc(WINDOW_WIDTH * WINDOW_HEIGHT * 3 * 4)  # Window size with 3 channels, float32
except cuda.LogicError as e:
    sys.exit(1)

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

class KeystrokeLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logs = []
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.SubWindow | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(10, 10, 700, DISPLAY_HEIGHT)  # Increased vertical space to full screen height
        self.setFont(QFont("Arial", 72))
        self.setStyleSheet("color: white;")

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_logs)
        self.timer.start(1000)  # Timer to update logs every second

    def add_keystroke(self, keystroke):
        self.logs.append((keystroke, QTime.currentTime()))
        self.update()

    def update_logs(self):
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        current_time = QTime.currentTime()
        new_logs = []
        y_offset = 0

        for text, time in self.logs:
            if time.msecsTo(current_time) < 1000:
                painter.setPen(QPen(Qt.black, 8))
                painter.drawText(4, 84 + y_offset, text)
                painter.setPen(QPen(Qt.white, 2))
                painter.drawText(0, 80 + y_offset, text)
                new_logs.append((text, time))
                y_offset += 100

        self.logs = new_logs

class MagnifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setup_global_listeners()

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

        self.hide_timer = QTimer(self)
        self.hide_timer.timeout.connect(self.hide_magnifier)

        self.last_mouse_pos = QCursor.pos()
        self.last_mouse_time = QTime.currentTime()

        self.keystroke_label = KeystrokeLabel()
        self.keystroke_label.show()

    def setup_global_listeners(self):
        self.keyboard_listener = keyboard.Listener(on_press=self.on_key_event)
        self.keyboard_listener.start()

    def on_key_event(self, key):
        key_name = self.get_key_name(key)
        self.keystroke_label.add_keystroke(key_name)

    def get_key_name(self, key):
        try:
            key_name = key.char
        except AttributeError:
            key_name = key.name
        # Use regex to remove _l, _r, and any other suffixes
        return re.sub(r'(_l|_r|_.*)', '', key_name)

    def update_magnifier(self):
        current_mouse_pos = QCursor.pos()
        current_time = QTime.currentTime()

        if current_mouse_pos != self.last_mouse_pos:
            self.last_mouse_time = current_time
            self.last_mouse_pos = current_mouse_pos
            if not self.isVisible():
                self.show()

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
            self.hide_timer.start(IDLE_TIMEOUT_MS)  # Restart the hide timer
        else:
            self.hide_magnifier()

    def hide_magnifier(self):
        self.hide()

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
