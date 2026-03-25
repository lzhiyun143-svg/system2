from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import time
import inspect
import re
import json
import base64
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from pose_analysis.evaluator import PoseEvaluator, Frame
import sys
import subprocess
import tempfile
import shutil

from openai import OpenAI
import uuid
import threading
import requests
from fastapi import UploadFile, File, Form, Query
from fastapi.staticfiles import StaticFiles


# =========================
# Config (ENV)
# =========================
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")
DASHSCOPE_VL_MODEL = os.getenv("DASHSCOPE_VL_MODEL", "qwen-vl-plus")

STANDARD_DIR = os.getenv("STANDARD_DIR", "./standards")
MUSETALK_API_BASE = os.getenv("MUSETALK_API_BASE", "http://127.0.0.1:19000")

# MimicMotion (Remote API)
MIMICMOTION_API_BASE = os.getenv("MIMICMOTION_API_BASE", "http://127.0.0.1:19002").rstrip("/")
MIMICMOTION_TIMEOUT = int(os.getenv("MIMICMOTION_TIMEOUT", "7200"))
MIMICMOTION_LOCK = threading.Lock()

# Request params passed to MimicMotion service
MIMICMOTION_NUM_FRAMES = int(os.getenv("MIMICMOTION_NUM_FRAMES", "72"))
MIMICMOTION_RESOLUTION = int(os.getenv("MIMICMOTION_RESOLUTION", "576"))
MIMICMOTION_FPS = int(os.getenv("MIMICMOTION_FPS", "15"))

GENERATED_DIR = Path(os.getenv("GENERATED_DIR", "./generated")).resolve()
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

USER_VIDEO_DIR = Path(os.getenv("USER_VIDEO_DIR", "./user_templates")).resolve()
USER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

DEMO_ACTION_DIR = Path(os.getenv("DEMO_ACTION_DIR", "./demo_action")).resolve()
DEMO_ACTION_DIR.mkdir(parents=True, exist_ok=True)

MUSETALK_LOCK = threading.Lock()

MAX_RULE_COMMENTS = int(os.getenv("MAX_RULE_COMMENTS", "6"))
MAX_COMMENT_CHARS = int(os.getenv("MAX_COMMENT_CHARS", "120"))

COMPARE_FRAMES_DIR = Path(r"D:\system\system2_compare_frames")
COMPARE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# =========================
# FastAPI
# =========================
app = FastAPI(title="Rehab Web Server", version="0.5.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/generated", StaticFiles(directory=str(GENERATED_DIR)), name="generated")
app.mount("/user_templates", StaticFiles(directory=str(USER_VIDEO_DIR)), name="user_templates")

if DEMO_ACTION_DIR.exists():
    app.mount("/demo_action", StaticFiles(directory=str(DEMO_ACTION_DIR)), name="demo_action")


# =========================
# Build evaluator safely
# =========================
def build_evaluator() -> PoseEvaluator:
    try:
        sig = inspect.signature(PoseEvaluator)
        params = list(sig.parameters.values())

        if any(p.name == "standard_dir" for p in params):
            return PoseEvaluator(standard_dir=STANDARD_DIR)

        required_positional = [
            p
            for p in params
            if p.default is inspect._empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(required_positional) >= 1:
            return PoseEvaluator(STANDARD_DIR)

        return PoseEvaluator()

    except Exception as e:
        try:
            return PoseEvaluator()
        except Exception:
            raise RuntimeError(f"Failed to init PoseEvaluator: {type(e).__name__}: {e}")


evaluator = build_evaluator()


# =========================
# LLM client
# =========================
def get_llm_client() -> OpenAI:
    if not DASHSCOPE_API_KEY:
        raise RuntimeError("DASHSCOPE_API_KEY is not set in environment variables.")
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def llm_ping() -> Dict[str, Any]:
    t0 = time.time()
    client = get_llm_client()

    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[
            {"role": "system", "content": "You are a connectivity test."},
            {"role": "user", "content": "Reply with OK only."},
        ],
        temperature=0,
        max_tokens=5,
    )

    dt = time.time() - t0
    content = resp.choices[0].message.content if resp.choices else ""
    return {
        "ok": True,
        "model": DASHSCOPE_MODEL,
        "base_url": DASHSCOPE_BASE_URL,
        "latency_sec": round(dt, 3),
        "reply": (content or "").strip(),
    }


def _trim_rule_comments(rule_comments: List[str]) -> List[str]:
    out: List[str] = []
    for c in rule_comments[:MAX_RULE_COMMENTS]:
        s = (c or "").strip()
        if len(s) > MAX_COMMENT_CHARS:
            s = s[:MAX_COMMENT_CHARS] + "…"
        if s:
            out.append(s)
    return out


def llm_action_feedback(action: str, metric_out: Dict[str, Any]) -> str:
    client = get_llm_client()

    dtw_score = metric_out.get("dtw_score", None)
    acc = metric_out.get("accuracy", None)
    joint_errors = metric_out.get("joint_errors", {})
    rule_comments = _trim_rule_comments(metric_out.get("rule_based_comments", []) or [])

    prompt = f"""
你是康复训练动作评估教练。用户在做动作：{action}

系统指标（仅供你参考）：
- dtw_score: {dtw_score}
- accuracy: {acc}
- joint_errors: {joint_errors}
- rule_based_comments: {rule_comments}

请用中文给出：
1) 一句话总体评价（是否接近标准）
2) 2-4条可执行的纠正建议（尽量具体到关节/方向/幅度）
要求：
- 不要提“DTW/accuracy”等术语
- 像真人教练一样说
""".strip()

    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[
            {"role": "system", "content": "You are a rehab motion evaluation coach."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=256,
    )
    return (resp.choices[0].message.content or "").strip()


def llm_confirm_judge(action: str, eval_result: Dict[str, Any]) -> Dict[str, Any]:
    client = get_llm_client()

    dtw_score = eval_result.get("dtw_score", None)
    acc = eval_result.get("accuracy", None)
    joint_errors = eval_result.get("joint_errors", {})
    rule_comments = _trim_rule_comments(eval_result.get("rule_based_comments", []) or [])

    prompt = f"""
你是康复训练动作的“复核官”。现在要复核动作：{action}

系统评估摘要（可信，但可能偏严/偏松）：
- dtw_score: {dtw_score}
- accuracy: {acc}
- joint_errors: {joint_errors}
- rule_based_comments: {rule_comments}

请你输出严格 JSON（只输出 JSON，不要任何多余文本），格式如下：
{{
  "is_pass": true/false,
  "confidence": 0-1,
  "overall": "一句话总结",
  "key_issues": ["问题1","问题2"],
  "tips": ["建议1","建议2","建议3"]
}}

要求：
- 不要出现“DTW/accuracy”等词
- 如果信息不足，也要给出保守判断，并在 overall 里说明“建议补采更稳定动作”
""".strip()

    t0 = time.time()
    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    latency = time.time() - t0
    text = (resp.choices[0].message.content or "").strip()

    try:
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        data = json.loads(text)
        data["model"] = DASHSCOPE_MODEL
        data["latency_sec"] = round(latency, 3)
        return data
    except Exception:
        return {
            "is_pass": None,
            "confidence": 0.0,
            "overall": "LLM 返回格式异常，建议查看 raw。",
            "key_issues": [],
            "tips": [],
            "model": DASHSCOPE_MODEL,
            "latency_sec": round(latency, 3),
            "raw": text,
        }


def tts_text_to_wav(text: str, out_wav: Path) -> None:
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Empty TTS text")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    tmp_mp3 = out_wav.with_suffix(".mp3")
    try:
        cmd = [
            sys.executable, "-m", "edge_tts",
            "--text", text,
            "--voice", os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
            "--write-media", str(tmp_mp3),
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if p.returncode != 0 or (not tmp_mp3.exists()):
            raise RuntimeError(p.stdout[-2000:])
    except Exception as e:
        raise RuntimeError(
            f"TTS failed. Please `pip install edge-tts` and ensure network OK. Detail: {type(e).__name__}: {e}"
        )

    ff = os.getenv("FFMPEG_BIN", "ffmpeg")
    cmd_ff = [ff, "-y", "-i", str(tmp_mp3), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    p2 = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p2.returncode != 0 or (not out_wav.exists()):
        raise RuntimeError(f"ffmpeg convert failed: {p2.stdout[-2000:]}")


def llm_confirm_judge_by_images(
    action: str,
    standard_images: List[str],
    user_images: List[str],
    eval_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = get_llm_client()

    standard_images = (standard_images or [])[:6]
    user_images = (user_images or [])[:6]

    if len(standard_images) < 2 or len(user_images) < 2:
        return {
            "is_pass": None,
            "confidence": 0.0,
            "overall": "关键帧数量不足，建议至少提供每侧2-4张关键帧。",
            "key_issues": ["关键帧不足或未成功采集"],
            "tips": ["重新采集动作", "确保人物完整入镜", "网络稳定后重试"],
            "mode": "vision_keyframes",
            "model": DASHSCOPE_VL_MODEL,
        }

    helper_text = ""
    if eval_result:
        helper_text = f"\n规则评估辅助信息（可选参考）: {json.dumps(eval_result, ensure_ascii=False)[:700]}"

    prompt_text = f"""
你是康复训练动作评估复核官。
任务：对比“标准动作关键帧”和“用户动作关键帧”，判断用户动作是否基本达标，并给出简洁、可执行建议。

动作名称：{action}

输入说明：
- 先给你【标准动作关键帧】（按时间顺序）
- 再给你【用户动作关键帧】（按时间顺序）
- 重点比较：动作幅度、抬起高度、左右对称性、时序一致性、是否有明显代偿
{helper_text}

输出要求（严格 JSON，仅 JSON，不要任何多余文字）：
{{
  "is_pass": true,
  "confidence": 0.0,
  "overall": "一句话总结，50字左右",
  "key_issues": ["问题1", "问题2"],
  "tips": ["建议1", "建议2", "建议3"]
}}

要求：
- 中文输出
- confidence 在 0~1
- 如果图像不清楚或看不全，请保守判断并在 overall / key_issues 里说明
- 不要提模型、分辨率、像素等技术词
""".strip()

    content_items = [{"type": "text", "text": prompt_text}]

    content_items.append({"type": "text", "text": "下面是【标准动作关键帧】（按时间顺序）"})
    for i, img in enumerate(standard_images, 1):
        content_items.append({"type": "text", "text": f"标准帧{i}"})
        content_items.append({"type": "image_url", "image_url": {"url": img}})

    content_items.append({"type": "text", "text": "下面是【用户动作关键帧】（按时间顺序）"})
    for i, img in enumerate(user_images, 1):
        content_items.append({"type": "text", "text": f"用户帧{i}"})
        content_items.append({"type": "image_url", "image_url": {"url": img}})

    t0 = time.time()
    resp = client.chat.completions.create(
        model=DASHSCOPE_VL_MODEL,
        messages=[
          {"role": "system", "content": "Return valid JSON only."},
          {"role": "user", "content": content_items},
        ],
        temperature=0.1,
        max_tokens=512,
    )
    latency = time.time() - t0

    text = (resp.choices[0].message.content or "").strip()

    try:
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()

        data = json.loads(text)
        data["mode"] = "vision_keyframes"
        data["model"] = DASHSCOPE_VL_MODEL
        data["latency_sec"] = round(latency, 3)
        return data
    except Exception:
        return {
            "is_pass": None,
            "confidence": 0.0,
            "overall": "LLM 返回格式异常（非JSON），请查看 raw。",
            "key_issues": ["返回格式解析失败"],
            "tips": ["稍后重试", "检查视觉模型配置"],
            "mode": "vision_keyframes",
            "model": DASHSCOPE_VL_MODEL,
            "latency_sec": round(latency, 3),
            "raw": text,
        }


def _save_dataurl_image(data_url: str, out_path: Path) -> bool:
    try:
        if not data_url or "," not in data_url:
            return False

        header, b64data = data_url.split(",", 1)
        if "base64" not in header:
            return False

        raw = base64.b64decode(b64data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(raw)
        return True
    except Exception as e:
        print(f"[save_dataurl_image] failed: {e}")
        return False


def save_compare_keyframes(
    action: str,
    standard_images: List[str],
    user_images: List[str],
) -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_action = re.sub(r"[^a-zA-Z0-9_\-]", "_", action or "unknown")
    session_dir = COMPARE_FRAMES_DIR / f"{ts}_{safe_action}"

    std_dir = session_dir / "standard"
    usr_dir = session_dir / "user"
    std_dir.mkdir(parents=True, exist_ok=True)
    usr_dir.mkdir(parents=True, exist_ok=True)

    saved_std = []
    saved_usr = []

    for i, data_url in enumerate(standard_images or [], 1):
        ext = ".jpg"
        if data_url.startswith("data:image/png"):
            ext = ".png"
        out_path = std_dir / f"std_{i:02d}{ext}"
        if _save_dataurl_image(data_url, out_path):
            saved_std.append(str(out_path))

    for i, data_url in enumerate(user_images or [], 1):
        ext = ".jpg"
        if data_url.startswith("data:image/png"):
            ext = ".png"
        out_path = usr_dir / f"user_{i:02d}{ext}"
        if _save_dataurl_image(data_url, out_path):
            saved_usr.append(str(out_path))

    return {
        "session_dir": str(session_dir),
        "saved_standard_count": len(saved_std),
        "saved_user_count": len(saved_usr),
        "saved_standard_files": saved_std,
        "saved_user_files": saved_usr,
    }


def save_json_file(path: Path, data: Dict[str, Any]):
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[save_json_file] failed: {e}")


def normalize_user_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("name is empty")
    safe = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_-]", "_", name)
    safe = safe.strip("_") or "user"
    return safe[:50]


def get_user_dir(name: str) -> Path:
    return USER_VIDEO_DIR / normalize_user_name(name)


def get_user_video_path(name: str) -> Path:
    return get_user_dir(name) / "avatar.mp4"


def get_user_meta_path(name: str) -> Path:
    return get_user_dir(name) / "meta.json"


def get_user_avatar_path(user_name: str) -> Path:
    return get_user_dir(user_name) / "avatar.mp4"


def get_user_photo_path(user_name: str) -> Path:
    return get_user_dir(user_name) / "photo.jpg"


def get_user_standard_dir(user_name: str) -> Path:
    p = get_user_dir(user_name) / "standard_videos"
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_user_standard_action_dir(user_name: str, action: str) -> Path:
    p = get_user_standard_dir(user_name) / action
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_upload_to_path(src: UploadFile, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        shutil.copyfileobj(src.file, f)


def transcode_to_mp4(src_path: Path, dst_path: Path):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src_path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-movflags", "+faststart",
        str(dst_path),
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it is in PATH.")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg transcode failed: {err[:1000]}")


def image_to_jpg(src_path: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)


def write_user_meta(user_name: str, meta: dict) -> None:
    meta_path = get_user_meta_path(user_name)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def read_user_meta(user_name: str) -> dict:
    p = get_user_meta_path(user_name)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def build_public_user_avatar_url(user_name: str) -> str:
    safe = normalize_user_name(user_name)
    return f"/user_templates/{safe}/avatar.mp4"


def build_public_user_photo_url(user_name: str) -> str:
    safe = normalize_user_name(user_name)
    return f"/user_templates/{safe}/photo.jpg"


def build_public_user_standard_url(user_name: str, action: str, filename: str) -> str:
    safe = normalize_user_name(user_name)
    return f"/user_templates/{safe}/standard_videos/{action}/{filename}"


def build_user_profile(name: str) -> Dict[str, Any]:
    user_dir = get_user_dir(name)
    video_path = get_user_video_path(name)
    meta_path = get_user_meta_path(name)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    return {
        "ok": True,
        "user_name": name,
        "safe_name": normalize_user_name(name),
        "exists": video_path.exists(),
        "video_path": str(video_path) if video_path.exists() else None,
        "user_dir": str(user_dir),
        "meta": meta,
    }


def _is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTS


def get_demo_action_video_path(action: str) -> Path:
    flat = DEMO_ACTION_DIR / f"{action}.mp4"
    if flat.exists():
        return flat

    action_dir = DEMO_ACTION_DIR / action
    if action_dir.exists() and action_dir.is_dir():
        candidates = [p for p in sorted(action_dir.iterdir()) if _is_video_file(p)]
        if candidates:
            return candidates[0]

    raise FileNotFoundError(f"demo action video not found for action={action}")


def get_demo_action_video_paths(action: str) -> List[Path]:
    out: List[Path] = []

    flat = DEMO_ACTION_DIR / f"{action}.mp4"
    if flat.exists() and _is_video_file(flat):
        out.append(flat)

    action_dir = DEMO_ACTION_DIR / action
    if action_dir.exists() and action_dir.is_dir():
        for p in sorted(action_dir.iterdir()):
            if _is_video_file(p):
                out.append(p)

    uniq = []
    seen = set()
    for p in out:
        key = str(p.resolve())
        if key not in seen:
            uniq.append(p)
            seen.add(key)
    return uniq


def get_demo_action_video_public_url(action: str, video_path: Path) -> str:
    rel = video_path.relative_to(DEMO_ACTION_DIR).as_posix()
    return f"/demo_action/{rel}"


def get_demo_action_video_id(video_path: Path) -> str:
    return video_path.stem


def get_user_generated_standard_path(user_name: str, action: str, filename: str) -> Path:
    return get_user_standard_action_dir(user_name, action) / filename


def build_generated_standard_item(user_name: str, action: str, out_path: Path) -> Dict[str, Any]:
    return {
        "file_name": out_path.name,
        "video_url": build_public_user_standard_url(user_name, action, out_path.name),
        "local_path": str(out_path),
    }




def call_mimicmotion_with_photo_and_video(
    photo_path: Path,
    pose_video_path: Path,
    action: str,
    num_frames: int = MIMICMOTION_NUM_FRAMES,
    resolution: int = MIMICMOTION_RESOLUTION,
    fps: int = MIMICMOTION_FPS,
) -> bytes:
    if not photo_path.exists():
        raise FileNotFoundError(f"photo not found: {photo_path}")
    if not pose_video_path.exists():
        raise FileNotFoundError(f"pose video not found: {pose_video_path}")

    url = f"{MIMICMOTION_API_BASE}/infer"
    print(f"[MimicMotion] request start -> action={action}, pose_video={pose_video_path.name}, url={url}")

    with MIMICMOTION_LOCK:
        with open(photo_path, "rb") as f_img, open(pose_video_path, "rb") as f_vid:
            files = {
                "ref_image": (photo_path.name, f_img, "image/jpeg"),
                "ref_video": (pose_video_path.name, f_vid, "video/mp4"),
            }
            data = {
                "action": action,
                "num_frames": str(int(num_frames)),
                "resolution": str(int(resolution)),
                "fps": str(int(fps)),
            }

            try:
                resp = requests.post(url, files=files, data=data, timeout=MIMICMOTION_TIMEOUT)
            except Exception as e:
                raise RuntimeError(f"request error for {pose_video_path.name}: {type(e).__name__}: {e}")

            print(f"[MimicMotion] response <- pose_video={pose_video_path.name}, status={resp.status_code}, content_length={len(resp.content)}")

            if resp.status_code != 200:
                raise RuntimeError(
                    f"MimicMotion infer failed for {pose_video_path.name}: "
                    f"{resp.status_code} {resp.text[:3000]}"
                )

            if not resp.content:
                raise RuntimeError(f"MimicMotion returned empty content for {pose_video_path.name}")

            return resp.content


# =========================
# Request/Response Models
# =========================
class FrameIn(BaseModel):
    pose: List[List[float]] = Field(..., description="33 x 3 pose landmarks")
    left_hand: Optional[List[List[float]]] = None
    right_hand: Optional[List[List[float]]] = None


class EvalRequest(BaseModel):
    action: str
    frames: Optional[List[FrameIn]] = None
    user_seq: Optional[List[FrameIn]] = None
    standard_seq: Optional[List[FrameIn]] = None
    use_llm: Optional[bool] = False


class ConfirmRequest(BaseModel):
    action: str
    frames: List[FrameIn]
    standard_seq: Optional[List[FrameIn]] = None
    eval_result: Dict[str, Any]
    standard_images: Optional[List[str]] = None
    user_images: Optional[List[str]] = None


class CoachVideoRequest(BaseModel):
    user_name: str
    text: str
    version: Optional[str] = "v1.5"
    mode: Optional[str] = "normal"


class BuildStandardVideoRequest(BaseModel):
    user_name: str
    action: str
    force: Optional[bool] = False


class EvaluateAutoRequest(BaseModel):
    action: str
    user_name: str
    frames: Optional[List[FrameIn]] = None
    user_seq: Optional[List[FrameIn]] = None
    standard_seq: Optional[List[FrameIn]] = None
    standard_images: Optional[List[str]] = None
    user_images: Optional[List[str]] = None
    use_llm: Optional[bool] = True


# =========================
# Helper for evaluate_auto
# =========================
def build_coach_script(confirm_out: Dict[str, Any], eval_out: Optional[Dict[str, Any]] = None) -> str:
    overall = str((confirm_out or {}).get("overall") or "").strip()
    key_issues = (confirm_out or {}).get("key_issues") or []
    tips = (confirm_out or {}).get("tips") or []

    if not overall and eval_out:
        overall = str(eval_out.get("llm_feedback") or "").strip()

    lines: List[str] = []

    if overall:
        lines.append(f"本次动作评估结果：{overall}")

    if key_issues:
        clean_issues = [str(x).strip() for x in key_issues if str(x).strip()]
        if clean_issues:
            lines.append("主要问题有：" + "；".join(clean_issues[:3]) + "。")

    if tips:
        clean_tips = [str(x).strip() for x in tips if str(x).strip()]
        if clean_tips:
            lines.append("建议你这样调整：" + "；".join(clean_tips[:3]) + "。")

    if not lines:
        lines.append("本次动作已完成评估。请继续保持动作稳定，注意幅度和节奏一致。")

    return "\n".join(lines)


def generate_musetalk_feedback_video(
    user_name: str,
    text: str,
    version: str = "v1.5",
    mode: str = "normal",
) -> str:
    if not text.strip():
        raise RuntimeError("feedback text is empty")

    video_path = get_user_video_path(user_name)
    if not video_path.exists():
        raise RuntimeError("user template video not found, please register first")

    uid = uuid.uuid4().hex[:10]
    safe_name = normalize_user_name(user_name)
    out_mp4 = GENERATED_DIR / f"coach_{safe_name}_{uid}.mp4"

    with tempfile.TemporaryDirectory(prefix="coach_video_") as td:
        td = Path(td)
        tts_wav = td / "tts.wav"

        tts_text_to_wav(text, tts_wav)

        with MUSETALK_LOCK:
            url = MUSETALK_API_BASE.rstrip("/") + "/infer"
            video_fp = open(video_path, "rb")
            audio_fp = open(tts_wav, "rb")
            files = {
                "video": (video_path.name, video_fp, "video/mp4"),
                "audio": (tts_wav.name, audio_fp, "audio/wav"),
            }
            data = {"version": version, "mode": mode}
            try:
                r = requests.post(url, data=data, files=files, timeout=1800)
            finally:
                try:
                    video_fp.close()
                except Exception:
                    pass
                try:
                    audio_fp.close()
                except Exception:
                    pass

        if r.status_code != 200:
            raise RuntimeError(f"MuseTalk failed: {r.text[:1000]}")

        with open(out_mp4, "wb") as f:
            f.write(r.content)

    return f"/generated/{out_mp4.name}"


def run_evaluate_core(req: EvalRequest) -> Dict[str, Any]:
    user_in = req.frames if req.frames is not None else req.user_seq
    if not user_in:
        raise HTTPException(status_code=422, detail="Field required: frames or user_seq")

    user_frames = [
        Frame(pose=f.pose, left_hand=f.left_hand, right_hand=f.right_hand)
        for f in user_in
        if f.pose is not None
    ]

    std_frames = None
    if req.standard_seq:
        std_frames = [
            Frame(pose=f.pose, left_hand=f.left_hand, right_hand=f.right_hand)
            for f in req.standard_seq
            if f.pose is not None
        ]

    if len(user_frames) < 3:
        raise HTTPException(status_code=400, detail="Too few valid frames (<3).")

    out = evaluator.evaluate(
        action=req.action,
        user_frames=user_frames,
        standard_frames=std_frames,
    )

    out["_meta"] = {
        "user_frames": len(user_frames),
        "standard_frames": len(std_frames) if std_frames is not None else 0,
        "use_standard": bool(std_frames),
        "use_llm_feedback": bool(req.use_llm),
    }

    if req.use_llm:
        try:
            out["llm_feedback"] = llm_action_feedback(req.action, out)
            out["llm_used"] = True
            out["llm_model"] = DASHSCOPE_MODEL
        except Exception as e:
            out["llm_used"] = False
            out["llm_error"] = f"{type(e).__name__}: {e}"

    return out


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/profile/check")
def api_profile_check(name: str) -> Dict[str, Any]:
    try:
        return build_user_profile(name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"profile check failed: {type(e).__name__}: {e}")


@app.get("/api/profile/info_legacy")
def api_profile_info_legacy(name: str) -> Dict[str, Any]:
    try:
        return build_user_profile(name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"profile info failed: {type(e).__name__}: {e}")


@app.post("/api/profile/register")
async def api_profile_register(
    name: str = Form(...),
    video: UploadFile = File(...),
) -> Dict[str, Any]:
    try:
        user_dir = get_user_dir(name)
        user_dir.mkdir(parents=True, exist_ok=True)

        raw_ext = Path(video.filename or "avatar.webm").suffix or ".webm"
        raw_path = user_dir / f"raw{raw_ext}"
        mp4_path = user_dir / "avatar.mp4"

        save_upload_to_path(video, raw_path)
        transcode_to_mp4(raw_path, mp4_path)

        meta = {
            "user_name": name,
            "safe_name": normalize_user_name(name),
            "video_path": str(mp4_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_filename": video.filename,
        }
        get_user_meta_path(name).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return {
            "ok": True,
            "message": "profile registered",
            "video_path": str(mp4_path),
            "profile": build_user_profile(name),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"profile register failed: {type(e).__name__}: {e}")


@app.get("/api/llm_ping")
def api_llm_ping() -> Dict[str, Any]:
    try:
        return llm_ping()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM ping failed: {type(e).__name__}: {e}",
        )


@app.get("/api/confirm")
def api_confirm_get() -> Dict[str, Any]:
    try:
        return llm_ping()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM confirm GET failed: {type(e).__name__}: {e}",
        )


@app.post("/api/confirm")
def api_confirm_post(req: ConfirmRequest) -> Dict[str, Any]:
    try:
        if req.frames is not None and len(req.frames) < 3:
            raise HTTPException(status_code=400, detail="Too few valid frames (<3).")

        if req.standard_images and req.user_images:
            std_imgs = (req.standard_images or [])[:6]
            usr_imgs = (req.user_images or [])[:6]

            save_info = save_compare_keyframes(
                action=req.action,
                standard_images=std_imgs,
                user_images=usr_imgs,
            )

            llm_out = llm_confirm_judge_by_images(
                action=req.action,
                standard_images=std_imgs,
                user_images=usr_imgs,
                eval_result=req.eval_result,
            )

            try:
                save_json_file(Path(save_info["session_dir"]) / "llm_result.json", llm_out)
            except Exception:
                pass

            return {
                "action": req.action,
                "confirm_mode": "vision_keyframes",
                "llm_confirm": llm_out,
                "saved_frames": {
                    "session_dir": save_info["session_dir"],
                    "saved_standard_count": save_info["saved_standard_count"],
                    "saved_user_count": save_info["saved_user_count"],
                },
            }

        llm_out = llm_confirm_judge(req.action, req.eval_result or {})

        return {
            "action": req.action,
            "confirm_mode": "json_eval",
            "llm_confirm": llm_out,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM confirm POST failed: {type(e).__name__}: {e}",
        )


@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest) -> Dict[str, Any]:
    try:
        return run_evaluate_core(req)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")


@app.post("/api/evaluate_auto")
def api_evaluate_auto(req: EvaluateAutoRequest) -> Dict[str, Any]:
    try:
        eval_req = EvalRequest(
            action=req.action,
            frames=req.frames,
            user_seq=req.user_seq,
            standard_seq=req.standard_seq,
            use_llm=req.use_llm,
        )
        eval_out = run_evaluate_core(eval_req)

        confirm_out: Dict[str, Any]
        save_info: Optional[Dict[str, Any]] = None

        std_imgs = (req.standard_images or [])[:6]
        usr_imgs = (req.user_images or [])[:6]

        if req.use_llm and len(std_imgs) >= 2 and len(usr_imgs) >= 2:
            save_info = save_compare_keyframes(
                action=req.action,
                standard_images=std_imgs,
                user_images=usr_imgs,
            )

            confirm_out = llm_confirm_judge_by_images(
                action=req.action,
                standard_images=std_imgs,
                user_images=usr_imgs,
                eval_result=eval_out,
            )

            try:
                save_json_file(Path(save_info["session_dir"]) / "llm_result.json", confirm_out)
            except Exception:
                pass
        elif req.use_llm:
            confirm_out = llm_confirm_judge(req.action, eval_out)
        else:
            confirm_out = {
                "is_pass": None,
                "confidence": 0.0,
                "overall": "已完成动作评估。",
                "key_issues": eval_out.get("rule_based_comments", [])[:3],
                "tips": eval_out.get("rule_based_comments", [])[:3],
                "mode": "rule_only",
            }

        feedback_text = build_coach_script(confirm_out, eval_out)

        coach_video_url = ""
        coach_video_error = ""
        try:
            coach_video_url = generate_musetalk_feedback_video(
                user_name=req.user_name,
                text=feedback_text,
                version="v1.5",
                mode="normal",
            )
        except Exception as e:
            coach_video_error = f"{type(e).__name__}: {e}"

        return {
            "ok": True,
            "action": req.action,
            "user_name": req.user_name,
            "evaluate": eval_out,
            "confirm": confirm_out,
            "feedback_text": feedback_text,
            "coach_video_url": coach_video_url,
            "coach_video_error": coach_video_error,
            "saved_frames": {
                "session_dir": save_info["session_dir"],
                "saved_standard_count": save_info["saved_standard_count"],
                "saved_user_count": save_info["saved_user_count"],
            } if save_info else None,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluate_auto failed: {type(e).__name__}: {e}")


@app.post("/api/coach_video_v2")
def api_coach_video_v2(req: CoachVideoRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    try:
        url = generate_musetalk_feedback_video(
            user_name=req.user_name,
            text=req.text,
            version=req.version or "v1.5",
            mode=req.mode or "normal",
        )
        return {
            "ok": True,
            "url": url,
            "text": req.text,
            "user_name": req.user_name,
            "template_video": str(get_user_video_path(req.user_name)),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"coach_video_v2 failed: {type(e).__name__}: {e}")


@app.post("/api/coach_video")
def api_coach_video(
    video: UploadFile = File(...),
    text: str = Form(...),
    version: str = Form("v1.5"),
    mode: str = Form("normal"),
) -> Dict[str, Any]:
    if not text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    uid = uuid.uuid4().hex[:10]
    out_mp4 = GENERATED_DIR / f"coach_{uid}.mp4"

    with tempfile.TemporaryDirectory(prefix="coach_video_") as td:
        td = Path(td)
        in_video = td / (video.filename or "standard.mp4")
        tts_wav = td / "tts.wav"

        with open(in_video, "wb") as f:
            shutil.copyfileobj(video.file, f)

        tts_text_to_wav(text, tts_wav)

        with MUSETALK_LOCK:
            url = MUSETALK_API_BASE.rstrip("/") + "/infer"
            files = {
                "video": (in_video.name, open(in_video, "rb"), "video/mp4"),
                "audio": (tts_wav.name, open(tts_wav, "rb"), "audio/wav"),
            }
            data = {"version": version, "mode": mode}

            try:
                r = requests.post(url, data=data, files=files, timeout=1800)
            finally:
                for _, fp, *_ in files.values():
                    try:
                        fp.close()
                    except Exception:
                        pass

        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"MuseTalk failed: {r.text[:1000]}")

        with open(out_mp4, "wb") as f:
            f.write(r.content)

    return {
        "ok": True,
        "url": f"/generated/{out_mp4.name}",
        "text": text,
    }


@app.post("/api/profile/register_v2")
async def api_profile_register_v2(
    name: str = Form(...),
    video: UploadFile = File(...),
    photo: UploadFile = File(...),
):
    try:
        safe_name = normalize_user_name(name)
        user_dir = get_user_dir(name)
        user_dir.mkdir(parents=True, exist_ok=True)

        video_suffix = Path(video.filename or "avatar.webm").suffix or ".webm"
        raw_video_path = user_dir / f"avatar_raw{video_suffix}"
        final_video_path = get_user_avatar_path(name)

        save_upload_to_path(video, raw_video_path)
        transcode_to_mp4(raw_video_path, final_video_path)

        photo_suffix = Path(photo.filename or "photo.jpg").suffix or ".jpg"
        raw_photo_path = user_dir / f"photo_raw{photo_suffix}"
        final_photo_path = get_user_photo_path(name)

        save_upload_to_path(photo, raw_photo_path)
        image_to_jpg(raw_photo_path, final_photo_path)

        old_meta = read_user_meta(name)
        standard_videos = old_meta.get("standard_videos", {})

        meta = {
            "user_name": name,
            "safe_name": safe_name,
            "created_at": old_meta.get("created_at") or datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "avatar_path": str(final_video_path),
            "photo_path": str(final_photo_path),
            "avatar_url": build_public_user_avatar_url(name),
            "photo_url": build_public_user_photo_url(name),
            "standard_videos": standard_videos,
        }
        write_user_meta(name, meta)

        return {
            "ok": True,
            "message": "profile registered",
            "profile": meta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"register_v2 failed: {type(e).__name__}: {e}")


@app.get("/api/profile/info")
def api_profile_info(user_name: str = Query(...)):
    try:
        meta = read_user_meta(user_name)
        if not meta:
            return {"ok": False, "exists": False}
        return {"ok": True, "exists": True, "profile": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"profile info failed: {type(e).__name__}: {e}")


@app.get("/api/standard_video/info")
def api_standard_video_info(
    user_name: str = Query(...),
    action: str = Query(...),
):
    try:
        demo_video_paths = get_demo_action_video_paths(action)
        if not demo_video_paths:
            raise FileNotFoundError(f"no demo videos found for action={action}")

        generated_dir = get_user_standard_action_dir(user_name, action)
        generated_items = []
        for p in sorted(generated_dir.glob("*")):
            if _is_video_file(p):
                generated_items.append(build_generated_standard_item(user_name, action, p))

        demo_items = []
        for p in demo_video_paths:
            demo_items.append({
                "id": get_demo_action_video_id(p),
                "file_name": p.name,
                "demo_video_url": get_demo_action_video_public_url(action, p),
                "demo_video_path": str(p),
            })

        display_video_url = generated_items[0]["video_url"] if generated_items else demo_items[0]["demo_video_url"]

        return {
            "ok": True,
            "user_name": user_name,
            "action": action,
            "demo_count": len(demo_items),
            "generated_count": len(generated_items),
            "all_generated": len(generated_items) == len(demo_items) and len(demo_items) > 0,
            "demo_videos": demo_items,
            "generated_videos": generated_items,
            "generated_exists": len(generated_items) > 0,
            "display_video_url": display_video_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"standard_video info failed: {type(e).__name__}: {e}")


@app.post("/api/standard_video/build")
def api_standard_video_build(req: BuildStandardVideoRequest):
    try:
        user_name = req.user_name
        action = req.action
        force = bool(req.force)

        photo_path = get_user_photo_path(user_name)
        if not photo_path.exists():
            raise HTTPException(status_code=404, detail="user photo not found, please register first")

        demo_video_paths = get_demo_action_video_paths(action)
        if not demo_video_paths:
            raise HTTPException(status_code=404, detail=f"no demo videos found for action={action}")

        print(f"[standard_video/build] user={user_name}, action={action}, force={force}")
        print(f"[standard_video/build] demo_video_count={len(demo_video_paths)}")
        for idx, p in enumerate(demo_video_paths, 1):
            print(f"[standard_video/build] demo[{idx}] = {p}")

        out_dir = get_user_standard_action_dir(user_name, action)
        out_dir.mkdir(parents=True, exist_ok=True)

        if force:
            for old in out_dir.glob("*"):
                if _is_video_file(old):
                    try:
                        old.unlink()
                    except Exception as e:
                        print(f"[standard_video/build] remove old file failed: {old} -> {e}")

        results = []
        errors = []

        for i, demo_video_path in enumerate(demo_video_paths, 1):
            demo_id = get_demo_action_video_id(demo_video_path)
            out_name = f"{i:03d}_mimicmotion_{demo_id}.mp4"
            out_path = get_user_generated_standard_path(user_name, action, out_name)

            print(f"[standard_video/build] start {i}/{len(demo_video_paths)} -> demo_id={demo_id}, out={out_path.name}")

            if out_path.exists() and not force:
                print(f"[standard_video/build] skip cached -> {out_path.name}")
                results.append({
                    "id": demo_id,
                    "source_demo_video_url": get_demo_action_video_public_url(action, demo_video_path),
                    "cached": True,
                    **build_generated_standard_item(user_name, action, out_path),
                })
                continue

            try:
                content = call_mimicmotion_with_photo_and_video(
                    photo_path=photo_path,
                    pose_video_path=demo_video_path,
                    action=action,
                    num_frames=MIMICMOTION_NUM_FRAMES,
                    resolution=MIMICMOTION_RESOLUTION,
                    fps=MIMICMOTION_FPS,
                )

                with open(out_path, "wb") as f:
                    f.write(content)

                print(f"[standard_video/build] success -> {out_path.name}, bytes={out_path.stat().st_size}")

                results.append({
                    "id": demo_id,
                    "source_demo_video_url": get_demo_action_video_public_url(action, demo_video_path),
                    "cached": False,
                    **build_generated_standard_item(user_name, action, out_path),
                })

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                print(f"[standard_video/build] failed -> demo_id={demo_id}, error={err_msg}")
                errors.append({
                    "id": demo_id,
                    "file_name": demo_video_path.name,
                    "error": err_msg,
                })

        meta = read_user_meta(user_name)
        standard_videos = meta.get("standard_videos", {})
        standard_videos[action] = {
            "action": action,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "demo_count": len(demo_video_paths),
            "success_count": len(results),
            "error_count": len(errors),
            "items": results,
            "errors": errors,
        }
        meta["standard_videos"] = standard_videos
        meta["updated_at"] = datetime.now().isoformat(timespec="seconds")
        write_user_meta(user_name, meta)

        fully_ok = (len(results) == len(demo_video_paths) and len(demo_video_paths) > 0)

        return {
            "ok": len(results) > 0,
            "fully_ok": fully_ok,
            "user_name": user_name,
            "action": action,
            "demo_count": len(demo_video_paths),
            "success_count": len(results),
            "error_count": len(errors),
            "results": results,
            "errors": errors,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"standard_video build failed: {type(e).__name__}: {e}")


@app.get("/api/actions")
def api_actions():
    try:
        if not DEMO_ACTION_DIR.exists():
            return {"ok": True, "actions": []}

        action_map: Dict[str, Dict[str, Any]] = {}

        for p in sorted(DEMO_ACTION_DIR.glob("*.mp4")):
            action_map[p.stem] = {
                "action": p.stem,
                "video_count": 1,
                "cover_video_url": f"/demo_action/{p.name}",
                "videos": [
                    {
                        "id": p.stem,
                        "file_name": p.name,
                        "video_url": f"/demo_action/{p.name}",
                    }
                ],
            }

        for d in sorted(DEMO_ACTION_DIR.iterdir()):
            if not d.is_dir():
                continue
            vids = [p for p in sorted(d.iterdir()) if _is_video_file(p)]
            if not vids:
                continue

            action_map[d.name] = {
                "action": d.name,
                "video_count": len(vids),
                "cover_video_url": f"/demo_action/{d.name}/{vids[0].name}",
                "videos": [
                    {
                        "id": p.stem,
                        "file_name": p.name,
                        "video_url": f"/demo_action/{d.name}/{p.name}",
                    }
                    for p in vids
                ],
            }

        actions = list(action_map.values())
        return {"ok": True, "actions": actions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"list actions failed: {type(e).__name__}: {e}")