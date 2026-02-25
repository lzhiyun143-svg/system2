# gui_app.py
import sys
import time
import json
import cv2
import vlc   # VLC 播放器

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QMessageBox
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer

from pose_analysis.pose_detector import PoseDetector
from pose_analysis.evaluator import PoseEvaluator
from llm.llm_agent import LLM_Agent


class RehabWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("上肢康复训练系统 - VLC版")
        self.resize(1200, 700)

        # 标准视频路径（可以改）
        self.demo_video_path = "demo.mp4"

        # 模块
        self.pose_detector = PoseDetector()
        self.evaluator = PoseEvaluator()
        self.llm_agent = LLM_Agent()

        # VLC 播放器
        self.vlc_instance = vlc.Instance()
        self.vlc_player = self.vlc_instance.media_player_new()

        self._build_ui()

        # 摄像头
        self.cap = cv2.VideoCapture(0)
        self.timer_cam = QTimer()
        self.timer_cam.timeout.connect(self.update_camera)

        # 录制
        self.recording = False
        self.user_seq = []
        self.record_start = 0
        self.record_duration = 5


    # ---------------- 界面布局 ----------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        video_layout = QHBoxLayout()
        layout.addLayout(video_layout, stretch=3)

        # VLC 视频输出窗口（左侧）
        self.label_demo = QLabel("示范视频区域")
        self.label_demo.setStyleSheet("background:#333; color:white; font-size:18px;")
        self.label_demo.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.label_demo)

        # 摄像头（右侧）
        self.label_camera = QLabel("摄像头区域")
        self.label_camera.setStyleSheet("background:#111; color:white; font-size:18px;")
        self.label_camera.setAlignment(Qt.AlignCenter)
        video_layout.addWidget(self.label_camera)

        # 按钮
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_training)
        layout.addWidget(self.btn_start)

        # 文本反馈
        self.text_feedback = QTextEdit()
        layout.addWidget(self.text_feedback, stretch=1)

        # 中间提示条
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: rgba(0,0,0,160); color:white; font-size:20px; padding:10px;"
        )
        self.status_label.hide()

    # 提示条居中
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        self.status_label.setGeometry(w//2-200, h//2-40, 400, 80)

    def show_status(self, msg):
        self.status_label.setText(msg)
        self.status_label.show()

    def hide_status(self):
        self.status_label.hide()


    # ---------------- 开始训练 ----------------
    def start_training(self):
        self.text_feedback.clear()

        # 播放示范视频（含声音）
        media = self.vlc_instance.media_new(self.demo_video_path)
        self.vlc_player.set_media(media)
        self.vlc_player.set_hwnd(int(self.label_demo.winId()))
        self.vlc_player.play()

        # 摄像头开
        if not self.cap.isOpened():
            self.cap.open(0)
        self.timer_cam.start(30)

        # 录制动作
        self.user_seq = []
        self.recording = True
        self.record_start = time.time()

        self.show_status("正在录制动作（约 5 秒）…")
        self.btn_start.setEnabled(False)


    # ---------------- 摄像头更新 ----------------
    def update_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # 录制动作
        if self.recording:
            t = time.time() - self.record_start
            if t <= self.record_duration:
                pts = self.pose_detector.detect(frame)
                if pts and len(pts) == 33:
                    self.user_seq.append({"pose": pts})
            else:
                self.recording = False
                self.show_status("正在分析动作…")
                self._analyze()
                self.hide_status()
                self.btn_start.setEnabled(True)

        # 显示摄像头
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._set_label_image(self.label_camera, rgb)


    # ---------------- 分析动作 ----------------
    def _analyze(self):

        # 保存用户骨架 JSON
        with open("user_pose.json", "w") as f:
            json.dump(self.user_seq, f, indent=2)

        result = self.evaluator.evaluate("raise_arm", self.user_seq)

        # 大模型
        try:
            llm_out = self.llm_agent.generate_feedback(result)
        except Exception as e:
            llm_out = f"LLM 调用失败：{e}"

        # 输出到文本框
        self.text_feedback.clear()
        self.text_feedback.append("【系统结果】")
        self.text_feedback.append(f"完成度：{result['accuracy']:.2f}")
        self.text_feedback.append(f"DTW：{result['dtw_score']:.2f}")
        self.text_feedback.append("\n关节误差：")
        self.text_feedback.append(str(result["joint_errors"]))

        self.text_feedback.append("\n【规则建议】")
        for c in result["rule_based_comments"]:
            self.text_feedback.append("· " + c)

        self.text_feedback.append("\n【大模型反馈】")
        self.text_feedback.append(llm_out)


    # ---------------- 显示图像 ----------------
    def _set_label_image(self, label, rgb):
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, w*ch, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)


    # ---------------- 关闭事件 ----------------
    def closeEvent(self, event):
        if self.cap.isOpened():
            self.cap.release()
        self.vlc_player.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RehabWindow()
    win.show()
    sys.exit(app.exec_())
