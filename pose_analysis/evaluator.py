from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import math
import os


@dataclass
class Frame:
    pose: List[List[float]]
    left_hand: Optional[List[List[float]]] = None
    right_hand: Optional[List[List[float]]] = None


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class PoseEvaluator:
    def __init__(self, standard_dir: Optional[str] = None):
        self.standard_dir = standard_dir or os.getenv("REHAB_STANDARD_DIR", "")
        self.normalize_before_compare = True
        self.normalize_eps = 1e-6

        # ===== 多轮训练相关默认参数 =====
        self.error_norm_min = float(os.getenv("ERROR_NORM_MIN", "0.0"))
        self.error_norm_max = float(os.getenv("ERROR_NORM_MAX", "40.0"))

        # 整体误差加权
        self.joint_error_weights = {
            "shoulder": float(os.getenv("SHOULDER_ERROR_WEIGHT", "0.7")),
            "elbow": float(os.getenv("ELBOW_ERROR_WEIGHT", "0.3")),
        }

        # 规则达标判定
        self.default_pass_accuracy = float(os.getenv("DEFAULT_PASS_ACCURACY", "0.75"))
        self.default_pass_error = float(os.getenv("DEFAULT_PASS_ERROR", "15.0"))

        # 关键点索引（MediaPipe Pose）
        self.LEFT_SHOULDER = 11
        self.RIGHT_SHOULDER = 12
        self.LEFT_ELBOW = 13
        self.RIGHT_ELBOW = 14
        self.LEFT_WRIST = 15
        self.RIGHT_WRIST = 16
        self.LEFT_HIP = 23
        self.RIGHT_HIP = 24

    # --------- 基础工具 ---------
    def _safe_xy(self, pose: List[List[float]], idx: int) -> Optional[List[float]]:
        if not pose or idx < 0 or idx >= len(pose):
            return None
        p = pose[idx]
        if not isinstance(p, list) or len(p) < 2:
            return None
        return [float(p[0]), float(p[1])]

    def _torso_center_and_scale(self, pose: List[List[float]]) -> Optional[Dict[str, float]]:
        l_sh = self._safe_xy(pose, self.LEFT_SHOULDER)
        r_sh = self._safe_xy(pose, self.RIGHT_SHOULDER)
        l_hip = self._safe_xy(pose, self.LEFT_HIP)
        r_hip = self._safe_xy(pose, self.RIGHT_HIP)
        if not (l_sh and r_sh and l_hip and r_hip):
            return None

        cx = (l_sh[0] + r_sh[0] + l_hip[0] + r_hip[0]) / 4.0
        cy = (l_sh[1] + r_sh[1] + l_hip[1] + r_hip[1]) / 4.0

        sh_cx = (l_sh[0] + r_sh[0]) / 2.0
        sh_cy = (l_sh[1] + r_sh[1]) / 2.0
        hip_cx = (l_hip[0] + r_hip[0]) / 2.0
        hip_cy = (l_hip[1] + r_hip[1]) / 2.0

        torso_len = math.sqrt((sh_cx - hip_cx) ** 2 + (sh_cy - hip_cy) ** 2)
        scale = max(self.normalize_eps, torso_len)
        return {"cx": cx, "cy": cy, "scale": scale}

    def _normalize_pose(self, pose: List[List[float]]) -> List[List[float]]:
        if not self.normalize_before_compare:
            return [[float(p[0]), float(p[1]), 0.0] for p in pose]

        info = self._torso_center_and_scale(pose)
        if info is None:
            return [[float(p[0]), float(p[1]), 0.0] for p in pose]

        cx, cy, s = info["cx"], info["cy"], info["scale"]
        out: List[List[float]] = []
        for p in pose:
            if len(p) >= 2:
                x = (float(p[0]) - cx) / s
                y = (float(p[1]) - cy) / s
                out.append([x, y, 0.0])
            else:
                out.append([0.0, 0.0, 0.0])
        return out

    def _frame_distance(self, a: Frame, b: Frame) -> float:
        pa = self._normalize_pose(a.pose or [])
        pb = self._normalize_pose(b.pose or [])
        if not pa or not pb:
            return 1.0

        n = min(len(pa), len(pb))
        if n <= 0:
            return 1.0

        total = 0.0
        for i in range(n):
            dx = pa[i][0] - pb[i][0]
            dy = pa[i][1] - pb[i][1]
            total += math.sqrt(dx * dx + dy * dy)
        return total / n

    # --------- DTW ---------
    def _dtw_distance(self, seq_a: List[Frame], seq_b: List[Frame]) -> float:
        if not seq_a or not seq_b:
            return 1.0

        n, m = len(seq_a), len(seq_b)
        dp = [[float("inf")] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0.0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = self._frame_distance(seq_a[i - 1], seq_b[j - 1])
                dp[i][j] = cost + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

        return dp[n][m] / max(1, (n + m) / 2.0)

    # --------- 关节误差 ---------
    def _mean_point(self, pts: List[Optional[List[float]]]) -> Optional[List[float]]:
        valid = [p for p in pts if p is not None]
        if not valid:
            return None
        return [
            sum(p[0] for p in valid) / len(valid),
            sum(p[1] for p in valid) / len(valid),
        ]

    def _joint_errors(self, user_frames: List[Frame], std_frames: List[Frame]) -> Dict[str, float]:
        if not user_frames or not std_frames:
            return {"shoulder": 0.0, "elbow": 0.0}

        n = min(len(user_frames), len(std_frames))
        if n <= 0:
            return {"shoulder": 0.0, "elbow": 0.0}

        shoulder_err = 0.0
        elbow_err = 0.0
        shoulder_cnt = 0
        elbow_cnt = 0

        for i in range(n):
            up = self._normalize_pose(user_frames[i].pose or [])
            sp = self._normalize_pose(std_frames[i].pose or [])

            # shoulder
            u_sh = self._mean_point([
                self._safe_xy(up, self.LEFT_SHOULDER),
                self._safe_xy(up, self.RIGHT_SHOULDER),
            ])
            s_sh = self._mean_point([
                self._safe_xy(sp, self.LEFT_SHOULDER),
                self._safe_xy(sp, self.RIGHT_SHOULDER),
            ])
            if u_sh and s_sh:
                dx = u_sh[0] - s_sh[0]
                dy = u_sh[1] - s_sh[1]
                shoulder_err += math.sqrt(dx * dx + dy * dy)
                shoulder_cnt += 1

            # elbow
            u_el = self._mean_point([
                self._safe_xy(up, self.LEFT_ELBOW),
                self._safe_xy(up, self.RIGHT_ELBOW),
            ])
            s_el = self._mean_point([
                self._safe_xy(sp, self.LEFT_ELBOW),
                self._safe_xy(sp, self.RIGHT_ELBOW),
            ])
            if u_el and s_el:
                dx = u_el[0] - s_el[0]
                dy = u_el[1] - s_el[1]
                elbow_err += math.sqrt(dx * dx + dy * dy)
                elbow_cnt += 1

        return {
            "shoulder": shoulder_err / shoulder_cnt if shoulder_cnt > 0 else 0.0,
            "elbow": elbow_err / elbow_cnt if elbow_cnt > 0 else 0.0,
        }

    def _build_rule_comments(
        self,
        accuracy: float,
        joint_errors: Dict[str, float],
        overall_error: float,
    ) -> List[str]:
        comments: List[str] = []

        if accuracy < 0.6:
            comments.append("整体动作与标准差异较大，建议放慢节奏重新练习。")
        elif accuracy < 0.75:
            comments.append("动作基本完成，但稳定性和一致性仍需提升。")

        if joint_errors.get("shoulder", 0.0) > 0.18:
            comments.append("肩部抬举幅度或左右对称性不足，建议注意肩部高度。")
        if joint_errors.get("elbow", 0.0) > 0.20:
            comments.append("肘部姿态控制不够稳定，建议注意屈伸角度。")
        if overall_error > self.default_pass_error:
            comments.append("整体误差偏大，建议减慢动作并保持轨迹稳定。")

        return comments[:6]

    # --------- 多轮训练：整体误差/归一化/规则达标 ---------
    def _overall_error(self, joint_errors: Dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0

        for k, w in self.joint_error_weights.items():
            total += float(joint_errors.get(k, 0.0)) * float(w)
            weight_sum += float(w)

        if weight_sum <= 1e-8:
            return 0.0
        return float(total / weight_sum)

    def _normalized_error(self, overall_error: float) -> float:
        lo = self.error_norm_min
        hi = self.error_norm_max
        if hi <= lo:
            return 0.0
        x = (float(overall_error) - lo) / (hi - lo)
        return float(_clamp(x, 0.0, 1.0))

    def _pass_by_rule(self, accuracy: float, overall_error: float) -> bool:
        return bool(
            float(accuracy) >= self.default_pass_accuracy
            and float(overall_error) <= self.default_pass_error
        )

    # --------- 主接口 ---------
    def evaluate(
        self,
        action: str,
        user_frames: List[Frame],
        standard_frames: Optional[List[Frame]] = None,
    ) -> Dict[str, Any]:
        if not user_frames:
            raise ValueError("user_frames is empty")
        if not standard_frames:
            raise ValueError("standard_frames is empty")

        dtw_distance = self._dtw_distance(user_frames, standard_frames)

        # 把距离映射成 0~1 准确度
        # 距离越小越接近 1
        accuracy = float(math.exp(-dtw_distance / 0.35))
        accuracy = float(_clamp(accuracy, 0.0, 1.0))

        joint_errors = self._joint_errors(user_frames, standard_frames)
        overall_error = self._overall_error(joint_errors)
        normalized_error = self._normalized_error(overall_error)
        pass_by_rule = self._pass_by_rule(accuracy, overall_error)
        comments = self._build_rule_comments(accuracy, joint_errors, overall_error)

        return {
            "action": action,
            "dtw_score": float(dtw_distance),
            "accuracy": float(accuracy),
            "joint_errors": joint_errors,
            "overall_error": float(overall_error),
            "normalized_error": float(normalized_error),
            "pass_by_rule": bool(pass_by_rule),
            "rule_based_comments": comments,
        }