# 카메라 캘리브레이션 예제

import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt

class CameraCalibrationUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.captured_images = []

    def initUI(self):
        self.setWindowTitle('카메라 캘리브레이션')
        self.setGeometry(100, 100, 640, 480)  # 창 크기 설정
        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        self.capture_btn = QPushButton('이미지 캡처', self)
        self.capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_btn)

        self.calibrate_btn = QPushButton('캘리브레이션 실행', self)
        self.calibrate_btn.clicked.connect(self.run_calibration)
        layout.addWidget(self.calibrate_btn)

        self.setLayout(layout)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap.scaled(640, 480, Qt.KeepAspectRatio))

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            self.captured_images.append(frame)
            print(f"이미지 캡처됨 (총 {len(self.captured_images)}개)")

    def run_calibration(self):
        if len(self.captured_images) < 3:
            print("최소 3개의 이미지가 필요합니다.")
            return

        square_size = 25  # 체스판 한 변의 크기 (mm)
        square_width = 9
        square_height = 6

        ret, mtx, dist, rvecs, tvecs = self.calibrate_camera(self.captured_images, square_size, square_width, square_height)
        if ret:
            print("캘리브레이션 완료")
            print("Camera Matrix:")
            print(mtx)
        else:
            print("캘리브레이션 실패")

    def calibrate_camera(self, images, square_size, square_width, square_height):
        objp = np.zeros((square_width * square_height, 3), np.float32)
        objp[:, :2] = np.mgrid[0:square_width, 0:square_height].T.reshape(-1, 2)
        objp *= square_size

        obj_points = []
        img_points = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (square_width, square_height), None)

            if ret:
                obj_points.append(objp)
                img_points.append(corners)
            else:
                print("체스판을 찾을 수 없습니다.")

        if not obj_points:
            print("유효한 이미지를 찾을 수 없습니다.")
            return False, None, None, None, None

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
        return ret, mtx, dist, rvecs, tvecs

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = CameraCalibrationUI()
    ex.show()
    sys.exit(app.exec_())




