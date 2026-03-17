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

# OpenAI SDK（兼容 DashScope compatible-mode）
from openai import OpenAI
import uuid
import threading
import requests
from fastapi import UploadFile, File, Form
from fastapi.staticfiles import StaticFiles

# =========================
# Config (ENV)
# =========================
DASHSCOPE_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")  # 必须在环境变量里设置
DASHSCOPE_MODEL = os.getenv("DASHSCOPE_MODEL", "qwen-plus")  # 可换 qwen-max
DASHSCOPE_VL_MODEL = os.getenv("DASHSCOPE_VL_MODEL", "qwen-vl-plus")
STANDARD_DIR = os.getenv("STANDARD_DIR", "./standards")
MUSETALK_API_BASE = os.getenv("MUSETALK_API_BASE", "http://127.0.0.1:19000")  # 你本地转发后的 MuseTalk API
GENERATED_DIR = Path(os.getenv("GENERATED_DIR", "./generated"))
GENERATED_DIR.mkdir(parents=True, exist_ok=True)
USER_VIDEO_DIR = Path(os.getenv("USER_VIDEO_DIR", "./user_templates"))
USER_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

# ⚠️ 你现在 MuseTalk 服务是固定槽位 data/video/output.mp4 + data/audio/output.wav，不支持并发
MUSETALK_LOCK = threading.Lock()

# 安全：限制喂给 LLM 的最大 comment 数量/长度，避免 payload 过大
MAX_RULE_COMMENTS = int(os.getenv("MAX_RULE_COMMENTS", "6"))
MAX_COMMENT_CHARS = int(os.getenv("MAX_COMMENT_CHARS", "120"))

# 图片保存路径
COMPARE_FRAMES_DIR = Path("D:\system\system2_compare_frames")
COMPARE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# FastAPI
# =========================
app = FastAPI(title="Rehab Web Server", version="0.3.0")

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

# =========================
# Build evaluator safely
# - 兼容 PoseEvaluator() / PoseEvaluator(standard_dir=...) / PoseEvaluator(STANDARD_DIR)
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
        # 不要把 key 写在代码里；统一用环境变量
        raise RuntimeError("DASHSCOPE_API_KEY is not set in environment variables.")
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def llm_ping() -> Dict[str, Any]:
    """用于确认：服务端是否真的能连上大模型（以及耗时）。"""
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
    """
    把规则/DTW 输出交给大模型，生成更像“教练”的自然语言反馈。
    """
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
    """
    LLM 复核：基于 evaluator 的结果，输出更“确定”的结论（是否合格/主要问题/建议）。
    注意：不直接把全帧序列喂给 LLM，避免 payload 太大/泄露。
    """
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
  "is_pass": true/false,          // 是否基本达标（允许轻微误差）
  "confidence": 0-1,              // 你对结论的把握
  "overall": "一句话总结",
  "key_issues": ["问题1","问题2"], // 0-3条
  "tips": ["建议1","建议2","建议3"] // 2-5条，尽量可执行
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

    # 尝试解析 JSON（如果模型偶尔包了 ```json，也尽量兜底）
    import json
    try:
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        data = json.loads(text)
        data["model"] = DASHSCOPE_MODEL
        data["latency_sec"] = round(latency, 3)
        return data
    except Exception:
        # 解析失败：返回 raw，便于你排查 prompt
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
    """
    文本 -> wav(16k/mono, pcm_s16le)
    依赖：edge-tts + ffmpeg
    安装：pip install edge-tts
    """
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Empty TTS text")

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # 1) edge-tts 先生成 mp3
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

    # 2) mp3 -> wav 16k mono
    ff = os.getenv("FFMPEG_BIN", "ffmpeg")
    cmd_ff = [ff, "-y", "-i", str(tmp_mp3), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    p2 = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p2.returncode != 0 or (not out_wav.exists()):
        raise RuntimeError(f"ffmpeg convert failed: {p2.stdout[-2000:]}")
    

# 视觉调用
def llm_confirm_judge_by_images(
    action: str,
    standard_images: List[str],
    user_images: List[str],
    eval_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    client = get_llm_client()  # 你原来已有的方法，保留

    # 后端兜底限制，防止 payload 太大
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
        # 可选：把规则评估结果做辅助提示（截断避免太长）
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
  "overall": "一句话总结",
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

    # 标准帧
    content_items.append({"type": "text", "text": "下面是【标准动作关键帧】（按时间顺序）"})
    for i, img in enumerate(standard_images, 1):
        content_items.append({"type": "text", "text": f"标准帧{i}"})
        content_items.append({"type": "image_url", "image_url": {"url": img}})

    # 用户帧
    content_items.append({"type": "text", "text": "下面是【用户动作关键帧】（按时间顺序）"})
    for i, img in enumerate(user_images, 1):
        content_items.append({"type": "text", "text": f"用户帧{i}"})
        content_items.append({"type": "image_url", "image_url": {"url": img}})

    t0 = time.time()
    resp = client.chat.completions.create(
        model=DASHSCOPE_VL_MODEL,  # 注意这里用视觉模型
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
    """
    支持 data:image/jpeg;base64,... / data:image/png;base64,...
    """
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
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg and ensure it is in PATH.")
    except subprocess.CalledProcessError as e:
        err = (e.stderr or b"").decode("utf-8", errors="ignore")
        raise RuntimeError(f"ffmpeg transcode failed: {err[:1000]}")


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

    # 是否调用大模型给自然语言反馈
    use_llm: Optional[bool] = False


class ConfirmRequest(BaseModel):
    action: str
    # 你前端会传 frames
    frames: List[FrameIn]
    # 可选：标准序列（你前端现在会传 standard_seq）
    standard_seq: Optional[List[FrameIn]] = None
    # 前端把 /api/evaluate 的结果带上（推荐）
    eval_result: Dict[str, Any]

    standard_images: Optional[List[str]] = None
    user_images: Optional[List[str]] = None


class CoachVideoRequest(BaseModel):
    user_name: str
    text: str
    version: Optional[str] = "v1.5"
    mode: Optional[str] = "normal"


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/profile/check")
def api_profile_check(name: str) -> Dict[str, Any]:
    try:
        return build_user_profile(name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"profile check failed: {type(e).__name__}: {e}")


@app.get("/api/profile/info")
def api_profile_info(name: str) -> Dict[str, Any]:
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


# =========================
# LLM connectivity endpoints
# =========================
@app.get("/api/llm_ping")
def api_llm_ping() -> Dict[str, Any]:
    try:
        return llm_ping()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM ping failed: {type(e).__name__}: {e}",
        )


# GET /api/confirm：保持“连通性确认”（避免你浏览器直接点返回 404）
@app.get("/api/confirm")
def api_confirm_get() -> Dict[str, Any]:
    try:
        return llm_ping()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM confirm GET failed: {type(e).__name__}: {e}",
        )


# POST /api/confirm：真正做“LLM 复核”
@app.post("/api/confirm")
def api_confirm_post(req: ConfirmRequest) -> Dict[str, Any]:
    try:
        # 旧字段校验（兼容你的现有逻辑）
        if req.frames is not None and len(req.frames) < 3:
            raise HTTPException(status_code=400, detail="Too few valid frames (<3).")

        # =========================
        # 新模式：关键帧图像复核（优先）
        # =========================
        if req.standard_images and req.user_images:
            # 后端兜底限制，避免 payload 太大
            std_imgs = (req.standard_images or [])[:6]
            usr_imgs = (req.user_images or [])[:6]

            # 1) 保存关键帧到专门文件夹
            save_info = save_compare_keyframes(
                action=req.action,
                standard_images=std_imgs,
                user_images=usr_imgs,
            )

            # 2) 视觉模型复核（看图对比）
            llm_out = llm_confirm_judge_by_images(
                action=req.action,
                standard_images=std_imgs,
                user_images=usr_imgs,
                eval_result=req.eval_result,  # 可选辅助
            )

            # 3) 保存结果（可选）
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

        # =========================
        # 旧模式：JSON 复核（兼容）
        # =========================
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
    
    
# =========================
# Main evaluate endpoint
# =========================
@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest) -> Dict[str, Any]:
    user_in = req.frames if req.frames is not None else req.user_seq
    if not user_in:
        raise HTTPException(status_code=422, detail="Field required: frames or user_seq")

    # user frames
    user_frames = [
        Frame(pose=f.pose, left_hand=f.left_hand, right_hand=f.right_hand)
        for f in user_in
        if f.pose is not None
    ]

    # standard frames（来自前端叠加抽帧的 3 秒全帧）
    std_frames = None
    if req.standard_seq:
        std_frames = [
            Frame(pose=f.pose, left_hand=f.left_hand, right_hand=f.right_hand)
            for f in req.standard_seq
            if f.pose is not None
        ]

    if len(user_frames) < 3:
        raise HTTPException(status_code=400, detail="Too few valid frames (<3).")

    try:
        out = evaluator.evaluate(
            action=req.action,
            user_frames=user_frames,
            standard_frames=std_frames,
        )

        # ✅ 追加一些 meta，帮助你确认“是否真的用上了 standard_seq”
        out["_meta"] = {
            "user_frames": len(user_frames),
            "standard_frames": len(std_frames) if std_frames is not None else 0,
            "use_standard": bool(std_frames),
            "use_llm_feedback": bool(req.use_llm),
        }

        # 可选：让 LLM 生成教练反馈（和 confirm 不同：confirm 是“复核/达标判断”）
        if req.use_llm:
            try:
                out["llm_feedback"] = llm_action_feedback(req.action, out)
                out["llm_used"] = True
                out["llm_model"] = DASHSCOPE_MODEL
            except Exception as e:
                out["llm_used"] = False
                out["llm_error"] = f"{type(e).__name__}: {e}"

        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")

@app.post("/api/coach_video_v2")
def api_coach_video_v2(req: CoachVideoRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    try:
        video_path = get_user_video_path(req.user_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid user name: {e}")

    if not video_path.exists():
        raise HTTPException(status_code=404, detail="user template video not found, please register first")

    uid = uuid.uuid4().hex[:10]
    safe_name = normalize_user_name(req.user_name)
    out_mp4 = GENERATED_DIR / f"coach_{safe_name}_{uid}.mp4"

    with tempfile.TemporaryDirectory(prefix="coach_video_") as td:
        td = Path(td)
        tts_wav = td / "tts.wav"

        tts_text_to_wav(req.text, tts_wav)

        with MUSETALK_LOCK:
            url = MUSETALK_API_BASE.rstrip("/") + "/infer"
            video_fp = open(video_path, "rb")
            audio_fp = open(tts_wav, "rb")
            files = {
                "video": (video_path.name, video_fp, "video/mp4"),
                "audio": (tts_wav.name, audio_fp, "audio/wav"),
            }
            data = {"version": req.version or "v1.5", "mode": req.mode or "normal"}
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
            raise HTTPException(status_code=500, detail=f"MuseTalk failed: {r.text[:1000]}")

        with open(out_mp4, "wb") as f:
            f.write(r.content)

    return {
        "ok": True,
        "url": f"/generated/{out_mp4.name}",
        "text": req.text,
        "user_name": req.user_name,
        "template_video": str(video_path),
    }


@app.post("/api/coach_video")
def api_coach_video(
    video: UploadFile = File(...),     # 标准视频（前端把 demoVideoSrc fetch 成文件上传）
    text: str = Form(...),             # LLM 生成的教练反馈文本
    version: str = Form("v1.5"),
    mode: str = Form("normal"),
) -> Dict[str, Any]:
    """
    1) text -> wav
    2) video + wav -> MuseTalk
    3) 保存 mp4 -> /generated/xxx.mp4
    4) 返回 url
    """
    if not text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    # 生成一个唯一文件名
    uid = uuid.uuid4().hex[:10]
    out_mp4 = GENERATED_DIR / f"coach_{uid}.mp4"

    with tempfile.TemporaryDirectory(prefix="coach_video_") as td:
        td = Path(td)
        in_video = td / (video.filename or "standard.mp4")
        tts_wav = td / "tts.wav"

        # 保存上传的视频
        with open(in_video, "wb") as f:
            shutil.copyfileobj(video.file, f)

        # TTS -> wav
        tts_text_to_wav(text, tts_wav)

        # 调 MuseTalk（加锁，避免固定槽位覆盖）
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
                # 关闭文件句柄
                for _, fp, *_ in files.values():
                    try:
                        fp.close()
                    except Exception:
                        pass

        if r.status_code != 200:
            raise HTTPException(status_code=500, detail=f"MuseTalk failed: {r.text[:1000]}")

        # 保存结果
        with open(out_mp4, "wb") as f:
            f.write(r.content)

    return {
        "ok": True,
        "url": f"/generated/{out_mp4.name}",
        "text": text,
    }