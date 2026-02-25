import cv2
import mediapipe as mp

# 初始化模型
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1, enable_segmentation=False)
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# 画图样式
pose_style = mp_pose.POSE_CONNECTIONS
hand_style = mp_hands.HAND_CONNECTIONS

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # === 1. Pose 33 点检测 ===
    pose_result = pose.process(img_rgb)

    if pose_result.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            pose_style,
            mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_draw.DrawingSpec(color=(0, 128, 255), thickness=2)
        )

    # === 2. Hands 21 点检测（左手 + 右手）===
    hand_result = hands.process(img_rgb)

    if hand_result.multi_hand_landmarks:
        for hand_lms in hand_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame,
                hand_lms,
                hand_style,
                mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    cv2.imshow("Full Body + Hands Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
        break

cap.release()
cv2.destroyAllWindows()
