# pose_analysis/dtw_utils.py
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def dtw_distance(user_seq, std_seq):
    """使用关键点向量作为输入进行 DTW"""
    if len(user_seq) == 0 or len(std_seq) == 0:
        return 9999.0

    u = []
    s = []

    for frame in user_seq:
        pts = np.array(frame["pose"]).flatten()
        u.append(pts)

    for frame in std_seq:
        pts = np.array(frame["pose"]).flatten()
        s.append(pts)

    dist, _ = fastdtw(u, s, dist=euclidean)
    return float(dist)
