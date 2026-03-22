import os
from pathlib import Path
import requests

# ========= 1) 按需修改 =========
MUSEPOSE_API = "http://127.0.0.1:19001/infer"   # 改成你的服务器地址
PHOTO_PATH = "D:\谷歌下载\system-main\system-main\stand.jpg"               # 改成你的照片路径
POSE_VIDEO_PATH = "D:\谷歌下载\system-main\system-main\dance.mp4"      # 改成你的姿势视频路径
ACTION_NAME = "raise_arm"
OUT_PATH = "D:\谷歌下载\system-main\system-main\musepose_result.mp4"       # 返回视频保存位置

# 可选分辨率，显存紧张就 512
WIDTH = 512
HEIGHT = 512


def main():
    photo_path = Path(PHOTO_PATH)
    pose_video_path = Path(POSE_VIDEO_PATH)
    out_path = Path(OUT_PATH)

    if not photo_path.exists():
        raise FileNotFoundError(f"照片不存在: {photo_path}")
    if not pose_video_path.exists():
        raise FileNotFoundError(f"姿势视频不存在: {pose_video_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("开始上传到 MusePose 服务器...")
    print(f"照片: {photo_path}")
    print(f"姿势视频: {pose_video_path}")
    print(f"输出: {out_path}")

    with open(photo_path, "rb") as f_img, open(pose_video_path, "rb") as f_vid:
        files = {
            "photo": (photo_path.name, f_img, "image/jpeg"),
            "pose_video": (pose_video_path.name, f_vid, "video/mp4"),
        }
        data = {
            "action": ACTION_NAME,
            "width": str(WIDTH),
            "height": str(HEIGHT),
        }

        resp = requests.post(
            MUSEPOSE_API,
            files=files,
            data=data,
            timeout=3600
        )

    print(f"服务器返回状态码: {resp.status_code}")

    if resp.status_code != 200:
        print("返回失败，响应内容如下：")
        print(resp.text)
        raise RuntimeError("MusePose 调用失败")

    with open(out_path, "wb") as f:
        f.write(resp.content)

    print("生成成功")
    print(f"结果视频已保存到: {out_path}")


if __name__ == "__main__":
    main()