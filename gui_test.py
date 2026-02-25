# gui_app.py
import sys
import time
import cv2
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

        self.setWindowTitle("上肢康复训练系统 - 原型")
        self.resize(1200, 700)

        self.demo_video_path = "demo.mp4"  # 标准示范视频路径

        # 核心模块
        self.pose_detector = PoseDetector()
        self.evaluator = PoseEvaluator()
        self.llm_agent = LLM_Agent()

        self._build_ui()

        # 摄像头
        self.cap = cv2.VideoCapture(0)
        self.timer_camera = QTimer()
        self.timer_camera.timeout.connect(self.update_camera)

        # 示范视频
        self.demo_cap = None
        self.timer_demo = QTimer()
        self.timer_demo.timeout.connect(self.update_demo_video)

        # 录制相关
        self.recording = False
        self.user_seq = []
        self.record_start_time = 0
        self.record_duration = 5  # 录制 5 秒

    # ------------ 构建 GUI ------------
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # 上面两个视频区域
        video_layout = QHBoxLayout()
        main_layout.addLayout(video_layout, stretch=3)

        # 左侧示范动作
        self.label_demo = QLabel("示范动作区域")
        self.label_demo.setAlignment(Qt.AlignCenter)
        self.label_demo.setStyleSheet("background:#444; color:white; font-size:18px;")
        video_layout.addWidget(self.label_demo)

        # 右侧摄像头区域
        self.label_camera = QLabel("摄像头区域")
        self.label_camera.setAlignment(Qt.AlignCenter)
        self.label_camera.setStyleSheet("background:#222; color:white; font-size:18px;")
        video_layout.addWidget(self.label_camera)

        # 中间按键
        self.btn_start = QPushButton("开始训练")
        self.btn_start.clicked.connect(self.start_training)
        main_layout.addWidget(self.btn_start)

        # 下方反馈文本
        self.text_feedback = QTextEdit()
        self.text_feedback.setPlaceholderText("这里会显示大模型生成的反馈")
        main_layout.addWidget(self.text_feedback, stretch=1)

        # ==== 中间提示标签（覆盖在最上层） ====
        self.status_label = QLabel("", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        # 半透明黑底 + 白字
        self.status_label.setStyleSheet(
            "background-color: rgba(0, 0, 0, 160);"
            "color: white;"
            "font-size: 18px;"
            "padding: 12px;"
            "border-radius: 8px;"
        )
        self.status_label.hide()  # 默认隐藏

    # 让提示条始终居中
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_status_label_geometry()

    def _update_status_label_geometry(self):
        w = self.width()
        h = self.height()
        box_w = 480
        box_h = 80
        x = (w - box_w) // 2
        y = (h - box_h) // 2
        self.status_label.setGeometry(x, y, box_w, box_h)

    def show_status(self, text: str):
        self.status_label.setText(text)
        self._update_status_label_geometry()
        self.status_label.show()

    def hide_status(self):
        self.status_label.hide()

    # ------------ 按钮点击：开始训练 ------------
    def start_training(self):
        self.text_feedback.clear()
        self.text_feedback.setPlainText("提示：先播放标准示范，请认真观察，然后系统会录制你 5 秒的动作。")

        # 播放标准示范视频
        if self.demo_cap is not None:
            self.demo_cap.release()
        self.demo_cap = cv2.VideoCapture(self.demo_video_path)
        self.timer_demo.start(33)  # 大约 30fps

        # 开启摄像头显示
        if not self.cap.isOpened():
            self.cap.open(0)
        self.timer_camera.start(30)

        # 准备录制用户动作（立即开始录制）
        self.user_seq = []
        self.recording = True
        self.record_start_time = time.time()

        # 👉 屏幕中间提示：正在录制
        self.show_status("正在录制动作（约 5 秒），请尽量模仿示范动作…")

        # 防止重复点击
        self.btn_start.setEnabled(False)

    # ------------ 摄像头画面更新 ------------
    def update_camera(self):
        if not self.cap.isOpened():
            self.text_feedback.append("无法打开摄像头")
            self.timer_camera.stop()
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        # 镜像翻转，让患者看起来更自然
        frame = cv2.flip(frame, 1)

        # 如果正在录制骨架，则调用 PoseDetector
        if self.recording:
            now = time.time()
            if now - self.record_start_time <= self.record_duration:
                pose_dict = self.pose_detector.detect(frame)
                self.user_seq.append(pose_dict)
            else:
                # 录制结束，只做一次评估 & 调用大模型
                self.recording = False
                # 👉 切换成“正在分析”的提示
                self.show_status("正在分析动作并生成评价，请稍候…")
                # 在主线程里直接进行分析
                self._analyze_and_feedback()
                # 分析结束后，隐藏提示 & 允许再次训练
                self.hide_status()
                self.btn_start.setEnabled(True)

        # 显示摄像头画面
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._set_label_image(self.label_camera, rgb)

    # ------------ 示范视频播放 ------------
    def update_demo_video(self):
        if self.demo_cap is None:
            return

        ret, frame = self.demo_cap.read()
        if not ret:
            # 播放完一次就停掉（也可以改成循环）
            self.timer_demo.stop()
            self.demo_cap.release()
            self.demo_cap = None
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._set_label_image(self.label_demo, rgb)

    # ------------ 计算差异 + 调大模型 ------------
    def _analyze_and_feedback(self):
        if not self.user_seq:
            QMessageBox.warning(self, "提示", "未采集到用户动作数据，请重试。")
            return

        # 1. 评估动作
        evaluation = self.evaluator.evaluate("raise_arm", self.user_seq)

        # 2. 调用大模型生成文字反馈
        try:
            feedback_text = self.llm_agent.generate_feedback(evaluation)
        except Exception as e:
            feedback_text = f"调用大模型失败：{e}"

        # 3. 显示在下方文本框
        self.text_feedback.clear()
        self.text_feedback.append("【系统分析结果】")
        self.text_feedback.append(f"整体完成度：{evaluation['accuracy']:.2f}")
        self.text_feedback.append(f"DTW 距离：{evaluation.get('dtw_score', 0.0):.2f}")
        self.text_feedback.append("\n【规则初步建议】")
        for c in evaluation.get("rule_based_comments", []):
            self.text_feedback.append("- " + c)
        self.text_feedback.append("\n【大模型综合反馈】\n")
        self.text_feedback.append(feedback_text)

    # ------------ 工具函数：显示图像到 QLabel ------------
    def _set_label_image(self, label, rgb):
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pix = QPixmap.fromImage(qimg).scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        label.setPixmap(pix)

    # ------------ 清理资源 ------------
    def closeEvent(self, event):
        self.timer_camera.stop()
        if self.cap.isOpened():
            self.cap.release()

        self.timer_demo.stop()
        if self.demo_cap is not None:
            self.demo_cap.release()

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = RehabWindow()
    win.show()
    sys.exit(app.exec_())
