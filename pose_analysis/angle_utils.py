# pose_analysis/angle_utils.py
import math
import numpy as np

def angle(a, b, c):
    """计算三点夹角（二维平面投影）"""
    ba = np.array([a[0] - b[0], a[1] - b[1]])
    bc = np.array([c[0] - b[0], c[1] - b[1]])

    if np.linalg.norm(ba) == 0 or np.linalg.norm(bc) == 0:
        return 0.0

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(math.degrees(math.acos(cosine)))


def compute_joint_angles(user_seq, std_seq):
    """基于肘部、肩部、手腕角度对齐并对比"""
    joints = {
        "elbow":  (12, 14, 16),  # 右肩-右肘-右腕
        "shoulder": (11, 12, 14),  # 左肩-右肩-右肘
        "wrist": (14, 16, 20),  # 右肘-右腕-右手
    }

    frames = min(len(user_seq), len(std_seq))
    if frames == 0:
        return {j: 0.0 for j in joints.keys()}

    errors = {}

    for name, (a, b, c) in joints.items():
        u_list, s_list = [], []
        for i in range(frames):
            up = user_seq[i]["pose"]
            sp = std_seq[i]["pose"]

            ua = up[a]; ub = up[b]; uc = up[c]
            sa = sp[a]; sb = sp[b]; sc = sp[c]

            u_list.append(angle(ua, ub, uc))
            s_list.append(angle(sa, sb, sc))

        errors[name] = abs(sum(u_list) / len(u_list) - sum(s_list) / len(s_list))

    return errors
