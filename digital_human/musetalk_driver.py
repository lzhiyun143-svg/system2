# digital_human/musetalk_driver.py
import requests
import os

class MuseTalkAvatar:
    def __init__(self, server_url: str):
        # 这里填写你的服务器 Musetalk API 地址
        self.server_url = server_url

        # 本地数字人头像文件
        self.avatar_image = "./digital_human/avatar.jpg"

    def speak(self, text: str) -> str:
        """
        调用服务器上的 Musetalk，将文本转语音，然后合成数字人视频
        """
        # 第一步：让服务器生成语音（不用 OpenAI，不用 Dashscope）
        tts_resp = requests.post(
            f"{self.server_url}/tts",
            json={"text": text}
        )
        tts_resp.raise_for_status()
        audio_path_server = tts_resp.json()["audio_path"]

        # 第二步：将头像发给服务器生成数字人视频
        files = {
            "image": open(self.avatar_image, "rb")
        }

        data = {
            "audio_path": audio_path_server   # 服务器内部路径，服务器自己读
        }

        video_resp = requests.post(
            f"{self.server_url}/musetalk",
            files=files,
            data=data
        )

        video_resp.raise_for_status()

        # 保存生成的视频到本地
        output_path = "./digital_human/avatar_feedback.mp4"
        with open(output_path, "wb") as f:
            f.write(video_resp.content)

        return output_path
