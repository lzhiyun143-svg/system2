import pyttsx3

def text_to_wav(text, output="output.wav"):
    engine = pyttsx3.init()

    # 可选：调节声音属性
    engine.setProperty("rate", 180)   # 语速
    engine.setProperty("volume", 1.0) # 音量 (0~1)

    voices = engine.getProperty("voices")
    # 尝试选择中文女声（Windows 下可用）
    for v in voices:
        if "Chinese" in v.name or "ZH" in v.id:
            engine.setProperty('voice', v.id)

    engine.save_to_file(text, output)
    engine.runAndWait()
    print("生成成功：", output)


if __name__ == "__main__":
    text_to_wav("你的姿势很正确，继续保持哦")
