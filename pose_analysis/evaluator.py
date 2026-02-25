from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# ============== 小工具 ==============

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _safe_acos(x: float) -> float:
    return math.acos(_clamp(x, -1.0, 1.0))


def _vec(a: List[float], b: List[float]) -> Tuple[float, float]:
    # 2D vector b-a
    return (b[0] - a[0], b[1] - a[1])


def _norm(v: Tuple[float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1])

def _dist2(a: List[float], b: List[float]) -> float:
    dx = float(a[0] - b[0])
    dy = float(a[1] - b[1])
    return math.sqrt(dx * dx + dy * dy)


def _angle_between(v1: Tuple[float, float], v2: Tuple[float, float]) -> float:
    n1 = _norm(v1)
    n2 = _norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cosv = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return _safe_acos(cosv) * 180.0 / math.pi  # degrees


def _dtw(seq_a: List[List[float]], seq_b: List[List[float]]) -> float:
    """
    简单 DTW：代价 = 欧式距离
    seq_a: [T, D], seq_b: [S, D]
    """
    if not seq_a or not seq_b:
        return 1e9

    T = len(seq_a)
    S = len(seq_b)

    INF = 1e18
    dp = [[INF] * (S + 1) for _ in range(T + 1)]
    dp[0][0] = 0.0

    def dist(x: List[float], y: List[float]) -> float:
        return math.sqrt(sum((x[i] - y[i]) ** 2 for i in range(len(x))))

    for i in range(1, T + 1):
        for j in range(1, S + 1):
            c = dist(seq_a[i - 1], seq_b[j - 1])
            dp[i][j] = c + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return float(dp[T][S])


# ============== 关键点索引（MediaPipe Pose 33 点） ==============
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
L_SHOULDER = 11
R_SHOULDER = 12
L_ELBOW = 13
R_ELBOW = 14
L_WRIST = 15
R_WRIST = 16
L_HIP = 23
R_HIP = 24


@dataclass
class Frame:
    pose: List[List[float]]  # 33 x (x,y,z)  (前端给的是归一化坐标)
    left_hand: Optional[List[List[float]]] = None  # 21 x (x,y,z) 或 None
    right_hand: Optional[List[List[float]]] = None


class PoseEvaluator:
    """
    方案A：先保证链路稳定 & 返回值合理：
    - 接受 user_seq 或 frames
    - 标准序列：内置 raise_arm 模板（角度随时间变化）
    - 评估特征：肩关节外展角 + 肘关节屈伸角（2D）
    """

    def __init__(self, standard_dir: Optional[str] = None):
        self.standard_dir = standard_dir or os.getenv("REHAB_STANDARD_DIR", "")
        # 你后面要做"标准视频→标准骨架"，再把标准序列存文件即可；现在先用模板跑通。
        self.normalize_before_compare = True
        self.normalize_eps = 1e-6

    # --------- 关键点归一化（消除分辨率/尺度差异）---------
    def _normalize_pose(self, pose: List[List[float]]) -> List[List[float]]:
        """
        按帧归一化（平移 + 缩放）：
        - 参考点 c_t：左右髋中点；若髋缺失则用左右肩中点
        - 尺度 s_t：肩宽（优先）；若肩缺失则用髋宽；都不可用则不缩放
        返回新的 pose（不修改原对象）。
        """
        if not pose or len(pose) < 33:
            return pose

        # center
        c = None
        try:
            lhip, rhip = pose[L_HIP], pose[R_HIP]
            c = [(lhip[0] + rhip[0]) / 2.0, (lhip[1] + rhip[1]) / 2.0, (lhip[2] + rhip[2]) / 2.0]
        except Exception:
            c = None

        if c is None:
            try:
                ls, rs = pose[L_SHOULDER], pose[R_SHOULDER]
                c = [(ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0, (ls[2] + rs[2]) / 2.0]
            except Exception:
                c = [0.0, 0.0, 0.0]

        # scale
        s = 0.0
        try:
            s = _dist2(pose[L_SHOULDER], pose[R_SHOULDER])
        except Exception:
            s = 0.0
        if s <= self.normalize_eps:
            try:
                s = _dist2(pose[L_HIP], pose[R_HIP])
            except Exception:
                s = 0.0
        if s <= self.normalize_eps:
            s = 1.0

        inv = 1.0 / (float(s) + float(self.normalize_eps))
        out: List[List[float]] = []
        for p in pose:
            # 保守：点结构异常就原样塞回，避免整个序列崩
            try:
                out.append([(p[0] - c[0]) * inv, (p[1] - c[1]) * inv, (p[2] - c[2]) * inv])
            except Exception:
                out.append(p)
        return out

    # --------- 从 pose 计算每帧特征 ---------
    def _pick_arm_side(self, pose: List[List[float]]) -> str:
        """
        自动选“抬得更明显”的一侧：看 wrist 相对 shoulder 的抬高程度（y 越小越靠上）
        """
        try:
            ls = pose[L_SHOULDER]
            rs = pose[R_SHOULDER]
            lw = pose[L_WRIST]
            rw = pose[R_WRIST]
            left_lift = (ls[1] - lw[1])  # >0 表示腕在肩上方
            right_lift = (rs[1] - rw[1])
            return "left" if left_lift >= right_lift else "right"
        except Exception:
            return "left"

    def _frame_angles(self, pose: List[List[float]], side: str) -> Tuple[float, float]:
        """
        返回 (shoulder_abduction_like_angle, elbow_flexion_angle)
        - shoulder: upper-arm vs torso
        - elbow: upper-arm vs forearm
        """
        if side == "left":
            sh, el, wr, hip = pose[L_SHOULDER], pose[L_ELBOW], pose[L_WRIST], pose[L_HIP]
        else:
            sh, el, wr, hip = pose[R_SHOULDER], pose[R_ELBOW], pose[R_WRIST], pose[R_HIP]

        torso = _vec(sh, hip)      # shoulder -> hip
        upper = _vec(sh, el)       # shoulder -> elbow
        fore  = _vec(el, wr)       # elbow -> wrist

        shoulder_angle = _angle_between(upper, torso)          # 越大越"抬起"
        elbow_angle    = _angle_between(tuple(-upper[i] for i in range(2)), fore)  # 上臂反向 vs 前臂

        # elbow_angle 正常接近 160~180(伸直) 或更小(弯曲)，按你动作定义可再调
        return shoulder_angle, elbow_angle

    def _seq_features(self, frames: List[Frame]) -> Tuple[List[List[float]], Dict[str, float], str]:
        """
        把序列变成 DTW 特征序列，并给出一个简单 joint_errors（基于角度波动）
        """
        if not frames:
            return [], {"shoulder": 1e9, "elbow": 1e9}, "left"

        first_pose = frames[0].pose
        if self.normalize_before_compare:
            first_pose = self._normalize_pose(first_pose)
        side = self._pick_arm_side(first_pose)

        feats: List[List[float]] = []
        shoulders: List[float] = []
        elbows: List[float] = []

        for fr in frames:
            if not fr.pose or len(fr.pose) < 33:
                continue
            pose = fr.pose
            if self.normalize_before_compare:
                pose = self._normalize_pose(pose)
            sa, ea = self._frame_angles(pose, side)
            feats.append([sa, ea])
            shoulders.append(sa)
            elbows.append(ea)

        if not feats:
            return [], {"shoulder": 1e9, "elbow": 1e9}, side

        # 用"序列幅度是否足够"当作一个 error：raise_arm 肩角应该有明显上升
        shoulder_range = (max(shoulders) - min(shoulders)) if shoulders else 0.0
        elbow_range = (max(elbows) - min(elbows)) if elbows else 0.0

        joint_errors = {
            # range 太小，说明几乎没动 -> error 大
            "shoulder": float(max(0.0, 35.0 - shoulder_range)),  # 期望至少变化 ~35°
            "elbow": float(max(0.0, 25.0 - elbow_range)),        # raise_arm 若肘基本不动，这项会偏大；可后面再调
        }
        return feats, joint_errors, side

    # --------- 内置标准模板（先跑通） ---------
    def _standard_template(self, action: str, length: int) -> List[List[float]]:
        length = max(5, length)
        if action == "raise_arm":
            # 肩角从 ~15° 增到 ~85°；肘角保持相对伸直 ~165°
            out = []
            for t in range(length):
                a = t / (length - 1)
                shoulder = 15.0 + 70.0 * a
                elbow = 165.0
                out.append([shoulder, elbow])
            return out

        # 其他动作先给个"平坦模板"，避免崩
        return [[30.0, 165.0] for _ in range(length)]

    # --------- 规则与得分 ---------
    def _accuracy_from(self, dtw_score: float, joint_errors: Dict[str, float]) -> float:
        # dtw 越小越好；joint_errors 越小越好
        # 这个是“先能用”的平滑函数：不会动不动 0
        dtw_part = 1.0 / (1.0 + dtw_score / 25.0)  # 25 可调
        err = joint_errors.get("shoulder", 0.0) + 0.5 * joint_errors.get("elbow", 0.0)
        err_part = 1.0 / (1.0 + err / 15.0)
        acc = 0.65 * dtw_part + 0.35 * err_part
        return float(_clamp(acc, 0.0, 1.0))

    def _rule_comments(self, feats: List[List[float]], joint_errors: Dict[str, float], side: str) -> List[str]:
        comments: List[str] = []
        if not feats:
            return ["未检测到有效骨架序列，请确保上半身入镜、光线充足。"]

        shoulder_err = joint_errors.get("shoulder", 0.0)
        elbow_err = joint_errors.get("elbow", 0.0)

        if shoulder_err > 10:
            comments.append("整体动作幅度偏小（手臂抬高不够）。")
        else:
            comments.append("手臂抬高幅度基本达标。")

        if elbow_err > 10:
            comments.append("肘部变化不明显/姿态不稳定，可尝试更标准地完成动作。")

        comments.append(f"当前主要评估侧：{('左臂' if side=='left' else '右臂')}。")
        return comments

    # --------- 对外主函数 ---------
    def evaluate(
        self,
        action: str,
        user_frames: List[Frame],
        standard_frames: Optional[List[Frame]] = None,
    ) -> Dict[str, Any]:
        # 兼容性处理：如果传入的是字典列表，转换为 Frame 对象
        def _to_frame(obj: Any) -> Frame:
            if isinstance(obj, Frame):
                return obj
            elif isinstance(obj, dict):
                return Frame(
                    pose=obj.get("pose", []),
                    left_hand=obj.get("left_hand"),
                    right_hand=obj.get("right_hand"),
                )
            else:
                # 尝试作为对象访问属性
                return Frame(
                    pose=getattr(obj, "pose", []),
                    left_hand=getattr(obj, "left_hand", None),
                    right_hand=getattr(obj, "right_hand", None),
                )

        # 转换用户序列
        user_frames = [_to_frame(f) for f in user_frames]
        
        # 转换标准序列（如果提供）
        if standard_frames:
            standard_frames = [_to_frame(f) for f in standard_frames]

        # 用户序列 -> 特征
        user_feats, joint_errors, side = self._seq_features(user_frames)

        # 标准序列：如果你传了标准骨架，就用它；否则用模板
        if standard_frames:
            std_feats, _, _ = self._seq_features(standard_frames)
            if not std_feats:
                std_feats = self._standard_template(action, len(user_feats) or len(user_frames))
        else:
            std_feats = self._standard_template(action, len(user_feats) or len(user_frames))

        # DTW
        dtw_score = _dtw(user_feats, std_feats)

        # accuracy
        accuracy = self._accuracy_from(dtw_score, joint_errors)

        # comments
        comments = self._rule_comments(user_feats, joint_errors, side)

        return {
            "action": action,
            "dtw_score": float(dtw_score),
            "accuracy": float(accuracy),
            "joint_errors": joint_errors,
            "rule_based_comments": comments,
        }
