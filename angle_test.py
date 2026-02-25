import json
import numpy as np
from pose_analysis.evaluator import PoseEvaluator

ev = PoseEvaluator()

# 读取 JSON
def load_json_as_vectors(path):
    seq = json.load(open(path, "r"))

    vec_seq = []
    for frame_dict in seq:   # 每帧是 dict
        # pose + hand 拼成向量
        vec = []
        for part in ["pose", "left", "right"]:
            pts = frame_dict.get(part, [])
            for (x, y, z) in pts:
                vec.extend([x, y, z])
        vec_seq.append(np.array(vec, dtype=float))

    return vec_seq


user = load_json_as_vectors("user.json")
std = load_json_as_vectors("raise_arm.json")

result = ev.evaluate("result", user)

print(result)
