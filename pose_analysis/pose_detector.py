# pose_analysis/pose_detector.py
import mediapipe as mp
import cv2

class PoseDetector:
    def __init__(self):
        self.pose = mp.solutions.pose.Pose()

    def detect(self, frame):
        """返回 33 个关键点，每个 (x, y, z)，未检测则返回 None"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)

        if not result.pose_landmarks:
            return None

        pts = []
        for lm in result.pose_landmarks.landmark:
            pts.append([lm.x, lm.y, lm.z])  # 33 点，每点 3D

        return pts
