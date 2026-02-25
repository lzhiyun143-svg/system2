# demo_action/action_player.py
import os

class ActionPlayer:
    def __init__(self):
        self.base_dir = "./demo_action/action_library"

    def get_video_path(self, action_name: str):
        path = os.path.join(self.base_dir, f"{action_name}.mp4")
        if os.path.exists(path):
            return path
        return None
