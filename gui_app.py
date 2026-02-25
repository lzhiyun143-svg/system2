# gui_app.py
import sys
import time
import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from demo_action.action_player import ActionPlayer
from pose_analysis.pose_detector import PoseDetector
from pose_analysis.evaluator import PoseEvaluator
from llm.llm_agent import LLM_Agent
from digital_human.musetalk_driver import MuseTalkAvatar


class RehabWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("上肢康复训练系统 - 对齐分析原型")
        self.resize(1300, 750)

        self.pose_detector = PoseDetector()
        self.evaluator = PoseEvaluator()
        self.llm_agent = LLM_Agent()
        self.action_player = ActionPlayer()

        self.avatar = MuseTalkAvatar("http://YOUR_SERVER_IP:5000/musetalk")

        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)

        video_layout = QHBoxLayout()
        layout.addLayout(video_layout, stretch=3)

        self.label_demo = QLabel("标准动作")
        self.label_demo.setAlignment(Qt.AlignCenter)
        self.label_demo.setStyleSheet("background:#222; color:white;")
        video_layout.addWidget(self.label_demo)

        self.label_camera = QLabel("用户动作")
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setStyleSheet("background:#222; color:white;")
        video_layout.addWidget(self.label_camera)

        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.on_start_clicked)
        layout.addWidget(self.btn_start)

        self.text_feedback = QTextEdit()
        layout.addWidget(self.text_feedback)

    # -------------------
    def on_start_clicked(self):
        self.text_feedback.clear()
        template_seq = self.load_template_pose("raise_arm")
        user_seq = self.capture_user_pose_sequence(5)

        evaluation = self.evaluator.evaluate(template_seq, user_seq)
        feedback = self.llm_agent.generate_feedback(evaluation)

        self.text_feedback.setPlainText(feedback)
        video_path = self.avatar.speak(feedback)
        QMessageBox.information(self, "完成", f"数字人视频已生成：\n{video_path}")

    # -------------------
    def load_template_pose(self, action_name):
        """读取模板视频并生成骨架序列"""
        path = self.action_player.get_video_path(action_name)
        cap = cv2.VideoCapture(path)

        seq = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            pose = self.pose_detector.detect(frame)
            seq.append(pose)
        cap.release()
        return seq

    # -------------------
    def capture_user_pose_sequence(self, duration=5):
        cap = cv2.VideoCapture(0)
        start = time.time()
        seq = []

        while time.time() - start < duration:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            # 显示原始图，不显示骨架
            self.show_frame(frame, self.label_camera)

            pose = self.pose_detector.detect(frame)
            seq.append(pose)

            QApplication.processEvents()

        cap.release()
        return seq

    # -------------------
    def show_frame(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h,w,ch = rgb.shape
        qimg = QImage(rgb.data,w,h,ch*w,QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
        label.setPixmap(pix)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RehabWindow()
    win.show()
    sys.exit(app.exec_())
