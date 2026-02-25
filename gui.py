# 美化版本 gui_app.py（医疗风）
import sys
import time
import json
import cv2
import vlc
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QMessageBox, QGraphicsDropShadowEffect
)
from PyQt5.QtGui import QImage, QPixmap, QColor, QFont
from PyQt5.QtCore import Qt, QTimer

from pose_analysis.pose_detector import PoseDetector
from pose_analysis.evaluator import PoseEvaluator
from llm.llm_agent import LLM_Agent


class RehabWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("上肢康复训练系统 - 美化版")
        self.resize(1300, 750)

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

        # 录制设置
        self.recording = False
        self.user_seq = []
        self.record_start = 0
        self.record_duration = 5


    # -------------------- UI 美化 --------------------
    def _build_ui(self):
        self.setStyleSheet("""
            QWidget { background: #E9EFF6; }
            QLabel { font-size: 18px; color: #103F91; }
            QTextEdit {
                background: white;
                border-radius: 10px;
                font-size: 17px;
                padding: 10px;
            }
            QPushButton {
                background-color: #3A78D1;
                color: white;
                font-size: 20px;
                padding: 12px;
                border-radius: 20px;
            }
            QPushButton:hover {
                background-color: #4F8EF5;
            }
            QPushButton:disabled {
                background-color: #A0B9DD;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ------------ 顶部两个卡片 ------------
        video_layout = QHBoxLayout()
        layout.addLayout(video_layout, stretch=3)

        # 卡片样式生成函数
        def build_card(widget):
            widget.setStyleSheet("""
                background: white;
                border-radius: 18px;
                font-size: 18px;
            """)
            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(20)
            shadow.setXOffset(0)
            shadow.setYOffset(3)
            shadow.setColor(QColor(180, 180, 180, 180))
            widget.setGraphicsEffect(shadow)
            return widget

        # VLC 区域（示范）
        self.label_demo = QLabel("示范动作区域")
        self.label_demo.setAlignment(Qt.AlignCenter)
        card_demo = build_card(self.label_demo)
        video_layout.addWidget(card_demo)

        # 摄像头区域
        self.label_camera = QLabel("摄像头区域")
        self.label_camera.setAlignment(Qt.AlignCenter)
        card_cam = build_card(self.label_camera)
        video_layout.addWidget(card_cam)

        # ------------ 中间按钮 ------------
        self.btn_start = QPushButton("开始训练")
        layout.addWidget(self.btn_start)
        self.btn_start.clicked.connect(self.start_training)
        self.btn_start.setMinimumHeight(48)

        # ------------ 底部文字反馈区域 ------------
        self.text_feedback = QTextEdit()
        layout.addWidget(self.text_feedback, stretch=2)

        # ------------ 居中提示条 ------------
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color: rgba(0,0,0,180); color:white; font-size:22px; padding:12px; border-radius:12px;"
        )
        self.status_label.hide()


    # 居中提示条位置
    def resizeEvent(self, event):
        super().resizeEvent(event)
        w = self.width()
        h = self.height()
        self.status_label.setGeometry(w//2 - 200, h//2 - 40, 400, 80)


    def show_status(self, msg):
        self.status_label.setText(msg)
        self.status_label.show()

    def hide_status(self):
        self.status_label.hide()


    # -------------------- 开始训练 --------------------
    def start_training(self):
        self.text_feedback.clear()

        # 播放标准示范视频（带声音）
        media = self.vlc_instance.media_new(self.demo_video_path)
        self.vlc_player.set_media(media)
        self.vlc_player.set_hwnd(int(self.label_demo.winId()))
        self.vlc_player.play()

        # 摄像头
        if not self.cap.isOpened():
            self.cap.open(0)
        self.timer_cam.start(30)

        # 启动录制
        self.user_seq = []
        self.recording = True
        self.record_start = time.time()

        self.show_status("正在录制动作……请保持动作")
        self.btn_start.setEnabled(False)


    # -------------------- 摄像头更新 --------------------
    def update_camera(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)

        # 录制骨架
        if self.recording:
            if time.time() - self.record_start <= self.record_duration:
                pts = self.pose_detector.detect(frame)
                if pts and len(pts) == 33:
                    self.user_seq.append({"pose": pts})
            else:
                self.recording = False
                self.show_status("正在分析，请稍候…")
                self._analyze()
                self.hide_status()
                self.btn_start.setEnabled(True)

        # 显示摄像头画面
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.display_to_label(self.label_camera, rgb)


    # -------------------- 动作分析 + LLM --------------------
    def _analyze(self):
        with open("user_pose.json", "w") as f:
            json.dump(self.user_seq, f, indent=2)

        result = self.evaluator.evaluate("raise_arm", self.user_seq)

        try:
            llm_out = self.llm_agent.generate_feedback(result)
        except Exception as e:
            llm_out = f"肘关节可以再抬高，与肩部保持平行tfc ygfcvx ：{e}"

        self.text_feedback.clear()
        self.text_feedback.append("【系统结果】")
        self.text_feedback.append(f"完成度：{result['accuracy']:.2f}")
        self.text_feedback.append(f"DTW：{result['dtw_score']:.2f}")
        self.text_feedback.append(f"\n关节误差：{result['joint_errors']}")

        self.text_feedback.append("\n【规则建议】")
        for msg in result["rule_based_comments"]:
            self.text_feedback.append("· " + msg)

        self.text_feedback.append("\n【大模型反馈】")
        self.text_feedback.append(llm_out)


    # -------------------- 辅助函数：显示图像 --------------------
    def display_to_label(self, label, rgb):
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            label.width(), label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        label.setPixmap(pix)


    # -------------------- 关闭事件 --------------------
    def closeEvent(self, event):
        self.timer_cam.stop()
        if self.cap.isOpened():
            self.cap.release()
        self.vlc_player.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RehabWindow()
    win.show()
    sys.exit(app.exec_())
