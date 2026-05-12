from __future__ import annotations

from typing import Any, Dict, List, Optional
import os
import time
import inspect
import re
import json
import base64
import math
import statistics
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

MIMICMOTION_API_BASE = os.getenv("MIMICMOTION_API_BASE", "http://127.0.0.1:19002").rstrip("/")
MIMICMOTION_TIMEOUT = int(os.getenv("MIMICMOTION_TIMEOUT", "7200"))
MIMICMOTION_LOCK = threading.Lock()

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
# Logging helper
# =========================
def stage_log(tag: str, message: str) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] [{tag}] {message}", flush=True)


# =========================
# Timing statistics
# =========================
TIMING_STATS_LOCK = threading.Lock()
TIMING_STAGE_KEYS = [
    "pose_evaluation",
    "llm_feedback_generation",
    "digital_human_synthesis",
    "total_per_round",
]
TIMING_STAGE_LABELS = {
    "pose_evaluation": "Pose evaluation",
    "llm_feedback_generation": "LLM feedback generation",
    "digital_human_synthesis": "Digital human synthesis",
    "total_per_round": "Total per round",
}
TIMING_HISTORY: Dict[str, List[float]] = {k: [] for k in TIMING_STAGE_KEYS}
TIMING_PENDING_ROUNDS: List[Dict[str, Any]] = []
TIMING_MAX_HISTORY = int(os.getenv("TIMING_MAX_HISTORY", "200"))
TIMING_MAX_PENDING = int(os.getenv("TIMING_MAX_PENDING", "50"))


def _timing_trim_list(items: List[float], max_len: int = TIMING_MAX_HISTORY) -> List[float]:
    if len(items) > max_len:
        del items[:-max_len]
    return items


def _timing_stats(values: List[float]) -> Dict[str, float]:
    vals = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not vals:
        return {"count": 0, "avg": 0.0, "std": 0.0}
    avg = sum(vals) / len(vals)
    std = statistics.stdev(vals) if len(vals) >= 2 else 0.0
    return {"count": len(vals), "avg": avg, "std": std}


def _timing_print_summary_unlocked() -> None:
    lines = []
    border = "-" * 66
    lines.append(border)
    lines.append(f"{'Module':<28} {'Avg. Time (s)':>14} {'Std. Dev.':>12} {'Count':>8}")
    lines.append(border)
    for key in TIMING_STAGE_KEYS:
        s = _timing_stats(TIMING_HISTORY.get(key, []))
        lines.append(
            f"{TIMING_STAGE_LABELS[key]:<28} {s['avg']:>14.3f} {s['std']:>12.3f} {s['count']:>8d}"
        )
    lines.append(border)
    stage_log("TimingStats", "\n" + "\n".join(lines))


def _timing_add_stage(stage_key: str, elapsed_sec: float) -> None:
    with TIMING_STATS_LOCK:
        TIMING_HISTORY.setdefault(stage_key, []).append(float(elapsed_sec))
        _timing_trim_list(TIMING_HISTORY[stage_key])


def _timing_find_pending_for_confirm() -> Optional[Dict[str, Any]]:
    for item in TIMING_PENDING_ROUNDS:
        if item.get("pose_evaluation") is not None and item.get("llm_feedback_generation") is None:
            return item
    return None


def _timing_find_pending_for_coach() -> Optional[Dict[str, Any]]:
    for item in TIMING_PENDING_ROUNDS:
        if item.get("pose_evaluation") is not None and item.get("llm_feedback_generation") is not None and item.get("digital_human_synthesis") is None:
            return item
    return None


def _timing_register_stage(stage_key: str, elapsed_sec: float) -> None:
    elapsed_sec = float(elapsed_sec)
    if not math.isfinite(elapsed_sec):
        return

    with TIMING_STATS_LOCK:
        TIMING_HISTORY.setdefault(stage_key, []).append(elapsed_sec)
        _timing_trim_list(TIMING_HISTORY[stage_key])

        if stage_key == "pose_evaluation":
            TIMING_PENDING_ROUNDS.append({
                "id": uuid.uuid4().hex[:8],
                "created_at": time.time(),
                "pose_evaluation": elapsed_sec,
                "llm_feedback_generation": None,
                "digital_human_synthesis": None,
            })
            if len(TIMING_PENDING_ROUNDS) > TIMING_MAX_PENDING:
                del TIMING_PENDING_ROUNDS[:-TIMING_MAX_PENDING]
            stage_log("Timing", f"Pose evaluation: {elapsed_sec:.3f}s")
            return

        if stage_key == "llm_feedback_generation":
            pending = _timing_find_pending_for_confirm()
            if pending is None:
                pending = {
                    "id": uuid.uuid4().hex[:8],
                    "created_at": time.time(),
                    "pose_evaluation": None,
                    "llm_feedback_generation": None,
                    "digital_human_synthesis": None,
                }
                TIMING_PENDING_ROUNDS.append(pending)
            pending["llm_feedback_generation"] = elapsed_sec
            stage_log("Timing", f"LLM feedback generation: {elapsed_sec:.3f}s")
            return

        if stage_key == "digital_human_synthesis":
            pending = _timing_find_pending_for_coach()
            if pending is None:
                pending = {
                    "id": uuid.uuid4().hex[:8],
                    "created_at": time.time(),
                    "pose_evaluation": None,
                    "llm_feedback_generation": None,
                    "digital_human_synthesis": None,
                }
                TIMING_PENDING_ROUNDS.append(pending)
            pending["digital_human_synthesis"] = elapsed_sec
            stage_log("Timing", f"Digital human synthesis: {elapsed_sec:.3f}s")

            p = pending.get("pose_evaluation")
            c = pending.get("llm_feedback_generation")
            g = pending.get("digital_human_synthesis")
            if all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in [p, c, g]):
                total = float(p) + float(c) + float(g)
                TIMING_HISTORY.setdefault("total_per_round", []).append(total)
                _timing_trim_list(TIMING_HISTORY["total_per_round"])
                stage_log(
                    "Timing",
                    f"Round {pending.get('id')} total: {total:.3f}s (pose={float(p):.3f}s, llm={float(c):.3f}s, digital_human={float(g):.3f}s)",
                )
                try:
                    TIMING_PENDING_ROUNDS.remove(pending)
                except ValueError:
                    pass
                _timing_print_summary_unlocked()
            return


def _timing_seconds(start_t: float) -> float:
    return max(0.0, time.perf_counter() - float(start_t))


# =========================
# FastAPI
# =========================
app = FastAPI(title="Rehab Web Server", version="0.7.0")

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
            stage_log("Init", f"PoseEvaluator init with standard_dir={STANDARD_DIR}")
            return PoseEvaluator(standard_dir=STANDARD_DIR)

        required_positional = [
            p for p in params
            if p.default is inspect._empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(required_positional) >= 1:
            stage_log("Init", f"PoseEvaluator init with positional standard_dir={STANDARD_DIR}")
            return PoseEvaluator(STANDARD_DIR)

        stage_log("Init", "PoseEvaluator init with default constructor")
        return PoseEvaluator()

    except Exception as e:
        stage_log("Init", f"PoseEvaluator first init failed: {type(e).__name__}: {e}")
        try:
            return PoseEvaluator()
        except Exception:
            raise RuntimeError(f"Failed to init PoseEvaluator: {type(e).__name__}: {e}")


evaluator = build_evaluator()


# =========================
# Score helpers
# =========================
def _normalize_percent(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    if not math.isfinite(v):
        return None

    if 0.0 <= v <= 1.0:
        return max(0.0, min(100.0, v * 100.0))
    return max(0.0, min(100.0, v))


def _normalize_dtw_to_score(dtw_score: Any) -> float:
    try:
        v = float(dtw_score)
    except Exception:
        return 70.0

    if not math.isfinite(v):
        return 70.0

    if v < 0:
        v = 0.0

    if 0.0 <= v <= 1.0:
        return max(0.0, min(100.0, v * 100.0))

    score = 100.0 * math.exp(-v / 0.60)
    return max(0.0, min(100.0, score))


def _normalize_joint_errors_to_score(joint_errors: Any) -> float:
    if not isinstance(joint_errors, dict) or not joint_errors:
        return 75.0

    vals: List[float] = []
    for _, val in joint_errors.items():
        try:
            x = float(val)
            if math.isfinite(x):
                vals.append(max(0.0, x))
        except Exception:
            pass

    if not vals:
        return 75.0

    mean_err = sum(vals) / len(vals)
    score = 100.0 * math.exp(-mean_err / 0.30)
    return max(0.0, min(100.0, score))


def compute_rehab_score(metric_out: Dict[str, Any]) -> Dict[str, Any]:
    acc_score = _normalize_percent(metric_out.get("accuracy"))
    if acc_score is None:
        acc_score = 65.0

    dtw_score_norm = _normalize_dtw_to_score(metric_out.get("dtw_score"))
    joint_score = _normalize_joint_errors_to_score(metric_out.get("joint_errors"))

    raw_score = (
        0.60 * acc_score +
        0.15 * dtw_score_norm +
        0.25 * joint_score
    )

    rule_comments = metric_out.get("rule_based_comments", []) or []
    penalty = min(8.0, 1.0 * len(rule_comments))

    final_score = raw_score - penalty

    user_frames = metric_out.get("_meta", {}).get("user_frames", 0) if isinstance(metric_out.get("_meta"), dict) else 0
    if user_frames >= 3:
        final_score = max(30.0, final_score)

    final_score = max(0.0, min(100.0, final_score))

    if final_score >= 85:
        level = "优秀"
    elif final_score >= 70:
        level = "良好"
    elif final_score >= 60:
        level = "合格"
    else:
        level = "待提升"

    return {
        "score": round(final_score, 1),
        "score_breakdown": {
            "accuracy_score": round(acc_score, 1),
            "dtw_score_norm": round(dtw_score_norm, 1),
            "joint_score": round(joint_score, 1),
            "rule_penalty": round(penalty, 1),
        },
        "score_level": level,
    }


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

def llm_shorten_for_digital_human(full_text: str) -> str:
    text = (full_text or "").strip()
    if not text:
        return "本轮训练已完成，请继续保持动作稳定。"

    # 已经很短就不再压缩
    if len(text) <= 90:
        return text

    try:
        client = get_llm_client()
        prompt = f"""
你是康复训练数字人口播文案助手。

请把下面这段康复反馈，压缩成适合数字人口播的简短版本，要求：
1. 使用中文
2. 保留核心意思，不要漏掉最重要的问题和建议
3. 口语化、自然，像康复教练在说话
4. 长度控制在 40~80 个汉字左右，尽量不要超过 2~3 句话
5. 不要分点，不要编号，不要输出标题，只输出最终口播文本

原始反馈：
{text}
""".strip()

        resp = client.chat.completions.create(
            model=DASHSCOPE_MODEL,
            messages=[
                {"role": "system", "content": "Return plain Chinese text only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )

        short_text = (resp.choices[0].message.content or "").strip()
        short_text = re.sub(r"\s+", " ", short_text)

        if not short_text:
            return text[:80]

        # 再保险截断一下，避免太长
        if len(short_text) > 100:
            short_text = short_text[:100].rstrip("，。；、 ") + "。"

        return short_text

    except Exception:
        # LLM 摘要失败时，退化为规则截断
        simple = re.sub(r"\s+", " ", text)
        if len(simple) > 80:
            simple = simple[:80].rstrip("，。；、 ") + "。"
        return simple
    
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
""".strip()

    resp = client.chat.completions.create(
        model=DASHSCOPE_MODEL,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=256,
    )
    text = (resp.choices[0].message.content or "").strip()

    try:
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        data = json.loads(text)
        data["model"] = DASHSCOPE_MODEL
        return data
    except Exception:
        return {
            "is_pass": None,
            "confidence": 0.0,
            "overall": "LLM 返回格式异常，建议查看 raw。",
            "key_issues": [],
            "tips": [],
            "model": DASHSCOPE_MODEL,
            "raw": text,
        }


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
{helper_text}

输出要求（严格 JSON，仅 JSON，不要任何多余文字）：
{{
  "is_pass": true,
  "confidence": 0.0,
  "overall": "一句话总结，50字左右",
  "key_issues": ["问题1", "问题2"],
  "tips": ["建议1", "建议2", "建议3"]
}}
""".strip()

    content_items = [{"type": "text", "text": prompt_text}]
    content_items.append({"type": "text", "text": "下面是【标准动作关键帧】"})
    for i, img in enumerate(standard_images, 1):
        content_items.append({"type": "text", "text": f"标准帧{i}"})
        content_items.append({"type": "image_url", "image_url": {"url": img}})

    content_items.append({"type": "text", "text": "下面是【用户动作关键帧】"})
    for i, img in enumerate(user_images, 1):
        content_items.append({"type": "text", "text": f"用户帧{i}"})
        content_items.append({"type": "image_url", "image_url": {"url": img}})

    resp = client.chat.completions.create(
        model=DASHSCOPE_VL_MODEL,
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": content_items},
        ],
        temperature=0.1,
        max_tokens=512,
    )

    text = (resp.choices[0].message.content or "").strip()
    try:
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json", "", 1).strip()
        data = json.loads(text)
        data["mode"] = "vision_keyframes"
        data["model"] = DASHSCOPE_VL_MODEL
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
            "raw": text,
        }


def tts_text_to_wav(text: str, out_wav: Path) -> None:
    text = (text or "").strip()
    if not text:
        raise RuntimeError("Empty TTS text")

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp3 = out_wav.with_suffix(".mp3")

    cmd = [
        sys.executable, "-m", "edge_tts",
        "--text", text,
        "--voice", os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural"),
        "--write-media", str(tmp_mp3),
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or (not tmp_mp3.exists()):
        raise RuntimeError(f"TTS failed: {p.stdout[-2000:]}")

    ff = os.getenv("FFMPEG_BIN", "ffmpeg")
    cmd_ff = [ff, "-y", "-i", str(tmp_mp3), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    p2 = subprocess.run(cmd_ff, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p2.returncode != 0 or (not out_wav.exists()):
        raise RuntimeError(f"ffmpeg convert failed: {p2.stdout[-2000:]}")


def _build_generated_public_url(path: Path) -> str:
    path = path.resolve()
    try:
        rel = path.relative_to(GENERATED_DIR).as_posix()
        return f"/generated/{rel}"
    except Exception:
        raise RuntimeError(f"generated file is outside GENERATED_DIR: {path}")


def _musetalk_abs_url(base: str, maybe_url: str) -> str:
    base = (base or "").rstrip("/")
    maybe_url = (maybe_url or "").strip()
    if not maybe_url:
        return ""
    if maybe_url.startswith("http://") or maybe_url.startswith("https://"):
        return maybe_url
    if maybe_url.startswith("/"):
        return f"{base}{maybe_url}"
    return f"{base}/{maybe_url}"


def _write_video_bytes(dst: Path, content: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f:
        f.write(content)


def _download_to_generated(maybe_url: str, dst: Path) -> bool:
    url = _musetalk_abs_url(MUSETALK_API_BASE, maybe_url)
    if not url:
        return False
    try:
        resp = requests.get(url, timeout=300)
        if resp.status_code == 200 and resp.content:
            _write_video_bytes(dst, resp.content)
            return True
    except Exception:
        return False
    return False


def _collect_candidate_paths(payload: Dict[str, Any]) -> List[str]:
    candidates: List[str] = []
    for key in [
        "url", "video_url", "download_url", "path", "output_path", "result_path",
        "file_path", "saved_path", "filename", "output", "result",
    ]:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            candidates.append(val.strip())
    data = payload.get("data")
    if isinstance(data, dict):
        for key in [
            "url", "video_url", "download_url", "path", "output_path", "result_path",
            "file_path", "saved_path", "filename", "output", "result",
        ]:
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                candidates.append(val.strip())
    return candidates


def _try_resolve_musetalk_response(resp: requests.Response, out_path: Path) -> Optional[str]:
    ctype = (resp.headers.get("content-type") or "").lower()

    if resp.status_code == 200 and ("video/" in ctype or out_path.suffix.lower() in [".mp4", ".webm"] and resp.content[:16]):
        _write_video_bytes(out_path, resp.content)
        return _build_generated_public_url(out_path)

    payload: Optional[Dict[str, Any]] = None
    try:
        payload = resp.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        for candidate in _collect_candidate_paths(payload):
            p = Path(candidate)
            if p.exists() and p.is_file():
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(p, out_path)
                return _build_generated_public_url(out_path)
            if _download_to_generated(candidate, out_path):
                return _build_generated_public_url(out_path)

    return None

def generate_musetalk_feedback_video(
    user_name: str,
    text: str,
    version: str = "v1.5",
    mode: str = "normal",
) -> str:
    text = (text or "").strip()
    if not text:
        raise RuntimeError("empty text for coach video")

    template_video = get_user_video_path(user_name)
    if not template_video.exists():
        raise FileNotFoundError(f"user template video not found: {template_video}")

    user_safe = normalize_user_name(user_name)
    out_dir = GENERATED_DIR / "coach_videos" / user_safe
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    wav_path = out_dir / f"{run_id}.wav"
    out_video_path = out_dir / f"{run_id}.mp4"

    # 1) 生成当前反馈对应的语音
    tts_text_to_wav(text, wav_path)
    if not wav_path.exists():
        raise RuntimeError(f"TTS wav not created: {wav_path}")

    infer_url = f"{MUSETALK_API_BASE.rstrip('/')}/infer"

    # 2) 把“当前用户模板视频 + 当前反馈音频”直接传给 MuseTalk
    with open(template_video, "rb") as f_video, open(wav_path, "rb") as f_audio:
        files = {
            "video": (template_video.name, f_video, "video/mp4"),
            "audio": (wav_path.name, f_audio, "audio/wav"),
        }
        data = {
            "version": version,
            "mode": mode,
            "user_name": user_name,
        }

        resp = requests.post(infer_url, files=files, data=data, timeout=1800)

    if resp.status_code != 200:
        raise RuntimeError(
            f"MuseTalk infer failed: {resp.status_code} {resp.text[:3000]}"
        )

    content_type = (resp.headers.get("content-type") or "").lower()

    # 3) 如果直接返回视频二进制
    if "video" in content_type or resp.content[:12].startswith(b"\x00\x00\x00"):
        out_video_path.write_bytes(resp.content)
        return f"/generated/coach_videos/{user_safe}/{out_video_path.name}"

    # 4) 如果返回 JSON
    try:
        payload = resp.json()
    except Exception:
        raise RuntimeError(f"MuseTalk returned unknown response: {resp.text[:1000]}")

    # 4.1 直接返回可下载 URL
    url = payload.get("url") or payload.get("video_url")
    if url:
        if url.startswith("http://") or url.startswith("https://"):
            return url

        # 相对路径时，补成对当前 MuseTalk 服务可访问的地址
        return f"{MUSETALK_API_BASE.rstrip('/')}{url if url.startswith('/') else '/' + url}"

    # 4.2 返回本地文件路径
    local_path = payload.get("local_path") or payload.get("output") or payload.get("video_path")
    if local_path:
        src = Path(local_path)
        if not src.exists():
            raise RuntimeError(f"MuseTalk output path not found: {src}")
        shutil.copyfile(src, out_video_path)
        return f"/generated/coach_videos/{user_safe}/{out_video_path.name}"

    raise RuntimeError(f"MuseTalk response missing url/local_path: {payload}")


# =========================
# Utility: images / storage
# =========================
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
    except Exception:
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
        out_path = std_dir / f"std_{i:02d}.jpg"
        if _save_dataurl_image(data_url, out_path):
            saved_std.append(str(out_path))

    for i, data_url in enumerate(user_images or [], 1):
        out_path = usr_dir / f"user_{i:02d}.jpg"
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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =========================
# User/profile/path helpers
# =========================
def normalize_user_name(name: str) -> str:
    name = (name or "").strip()
    if not name:
        raise ValueError("name is empty")
    safe = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9_-]", "_", name)
    safe = safe.strip("_") or "user"
    return safe[:50]


def normalize_video_id(video_id: str) -> str:
    video_id = (video_id or "").strip()
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", video_id)
    return safe or "default"


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


def get_user_plan_path(name: str, action: str, video_id: str) -> Path:
    safe_video_id = normalize_video_id(video_id)
    return get_user_dir(name) / "train_plans" / action / f"{safe_video_id}.json"


def default_train_plan() -> Dict[str, Any]:
    return {
        "current_round": 1,
        "max_rounds": 4,
        "current_target": 0.75,
        "current_threshold": 15.0,
        "target_min": 0.60,
        "target_max": 0.95,
        "threshold_min": 5.0,
        "threshold_max": 30.0,
        "goal_step": 0.08,
        "threshold_step": 3.0,
        "alpha": 0.40,
        "beta": 0.25,
        "gamma": 0.25,
        "delta": 0.10,
        "base_value": 0.50,
        "joint_weights": {
            "shoulder": 0.7,
            "elbow": 0.3,
        },
        "history": [],
        "is_finished": False,
    }


def load_train_plan(name: str, action: str, video_id: str) -> Dict[str, Any]:
    path = get_user_plan_path(name, action, video_id)
    if not path.exists():
        plan = default_train_plan()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        return plan

    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(plan, dict):
            return default_train_plan()
        if "max_rounds" not in plan:
            plan["max_rounds"] = 4
        if "is_finished" not in plan:
            plan["is_finished"] = False
        return plan
    except Exception:
        return default_train_plan()


def save_train_plan(name: str, action: str, video_id: str, plan: Dict[str, Any]) -> None:
    path = get_user_plan_path(name, action, video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


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
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)


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


# =========================
# Demo/standard video helpers
# =========================
def _is_video_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in ALLOWED_VIDEO_EXTS


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

            resp = requests.post(url, files=files, data=data, timeout=MIMICMOTION_TIMEOUT)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"MimicMotion infer failed for {pose_video_path.name}: "
                    f"{resp.status_code} {resp.text[:3000]}"
                )
            if not resp.content:
                raise RuntimeError(f"MimicMotion returned empty content for {pose_video_path.name}")
            return resp.content


# =========================
# Training plan helpers
# =========================
def compute_overall_error_from_joint_errors(
    joint_errors: Dict[str, Any],
    joint_weights: Optional[Dict[str, Any]] = None,
) -> float:
    weights = joint_weights or {"shoulder": 0.7, "elbow": 0.3}
    total = 0.0
    weight_sum = 0.0
    for k, w in weights.items():
        total += float(joint_errors.get(k, 0.0)) * float(w)
        weight_sum += float(w)
    if weight_sum <= 1e-8:
        return 0.0
    return float(total / weight_sum)


def normalize_error(overall_error: float, err_min: float = 0.0, err_max: float = 40.0) -> float:
    if err_max <= err_min:
        return 0.0
    x = (float(overall_error) - float(err_min)) / (float(err_max) - float(err_min))
    return max(0.0, min(1.0, x))


def compute_training_state(
    accuracy: float,
    normalized_error_value: float,
    pass_flag: int,
    key_issue_count: int,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> float:
    state = (
        float(alpha) * float(accuracy)
        + float(beta) * float(1.0 - normalized_error_value)
        + float(gamma) * float(pass_flag)
        - float(delta) * float(key_issue_count)
    )
    return float(state)


def update_goal_and_threshold(
    current_target: float,
    current_threshold: float,
    state_value: float,
    base_value: float,
    goal_step: float,
    threshold_step: float,
    target_min: float,
    target_max: float,
    threshold_min: float,
    threshold_max: float,
) -> Dict[str, float]:
    next_target = float(current_target) + float(goal_step) * (float(state_value) - float(base_value))
    next_threshold = float(current_threshold) - float(threshold_step) * (float(state_value) - float(base_value))

    next_target = max(float(target_min), min(float(target_max), next_target))
    next_threshold = max(float(threshold_min), min(float(threshold_max), next_threshold))

    return {
        "next_target": float(next_target),
        "next_threshold": float(next_threshold),
    }


# =========================
# Request Models
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


class TrainPlanUpdateRequest(BaseModel):
    user_name: str
    action: str
    video_id: str
    eval_result: Dict[str, Any]
    llm_confirm: Optional[Dict[str, Any]] = None


class TrainPlanResetRequest(BaseModel):
    user_name: str
    action: str
    video_id: str


# =========================
# Core evaluation
# =========================
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

    pose_eval_t0 = time.perf_counter()
    out = evaluator.evaluate(
        action=req.action,
        user_frames=user_frames,
        standard_frames=std_frames,
    )
    pose_eval_sec = _timing_seconds(pose_eval_t0)

    out["_meta"] = {
        "user_frames": len(user_frames),
        "standard_frames": len(std_frames) if std_frames is not None else 0,
        "use_standard": bool(std_frames),
        "use_llm_feedback": bool(req.use_llm),
    }
    print("accuracy =", out.get("accuracy"))
    print("dtw_score =", out.get("dtw_score"))
    print("joint_errors =", out.get("joint_errors"))
    print("rule_based_comments =", out.get("rule_based_comments"))
    
    score_info = compute_rehab_score(out)
    print("score_breakdown =", score_info["score_breakdown"])
    print("final_score =", score_info["score"])
    out["score"] = score_info["score"]
    out["score_breakdown"] = score_info["score_breakdown"]
    out["score_level"] = score_info["score_level"]
    out.setdefault("_timing", {})["pose_evaluation_sec"] = round(pose_eval_sec, 6)

    if req.use_llm:
        try:
            llm_t0 = time.perf_counter()
            out["llm_feedback"] = llm_action_feedback(req.action, out)
            out["llm_used"] = True
            out["llm_model"] = DASHSCOPE_MODEL
            out.setdefault("_timing", {})["evaluate_side_llm_feedback_sec"] = round(_timing_seconds(llm_t0), 6)
        except Exception as e:
            out["llm_used"] = False
            out["llm_error"] = f"{type(e).__name__}: {e}"

    return out


# =========================
# Routes
# =========================
@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/profile/check")
def api_profile_check(name: str) -> Dict[str, Any]:
    try:
        return build_user_profile(name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"profile check failed: {type(e).__name__}: {e}")


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
            "profile": build_user_profile(name),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"register_v2 failed: {type(e).__name__}: {e}")


@app.get("/api/profile/info")
def api_profile_info(user_name: str = Query(...)):
    try:
        meta = read_user_meta(user_name)
        if not meta:
            return {"ok": False, "exists": False}
        return {"ok": True, "exists": True, "profile": build_user_profile(user_name)}
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

        out_dir = get_user_standard_action_dir(user_name, action)
        out_dir.mkdir(parents=True, exist_ok=True)

        if force:
            for old in out_dir.glob("*"):
                if _is_video_file(old):
                    try:
                        old.unlink()
                    except Exception:
                        pass

        results = []
        errors = []

        for i, demo_video_path in enumerate(demo_video_paths, 1):
            demo_id = get_demo_action_video_id(demo_video_path)
            out_name = f"{i:03d}_mimicmotion_{demo_id}.mp4"
            out_path = get_user_generated_standard_path(user_name, action, out_name)

            if out_path.exists() and not force:
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

                results.append({
                    "id": demo_id,
                    "source_demo_video_url": get_demo_action_video_public_url(action, demo_video_path),
                    "cached": False,
                    **build_generated_standard_item(user_name, action, out_path),
                })
            except Exception as e:
                errors.append({
                    "id": demo_id,
                    "file_name": demo_video_path.name,
                    "error": f"{type(e).__name__}: {e}",
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


@app.post("/api/evaluate")
def api_evaluate(req: EvalRequest) -> Dict[str, Any]:
    try:
        result = run_evaluate_core(req)
        pose_eval_sec = None
        try:
            pose_eval_sec = float(result.get("_timing", {}).get("pose_evaluation_sec"))
        except Exception:
            pose_eval_sec = None

        if pose_eval_sec is None or (not math.isfinite(pose_eval_sec)):
            pose_eval_sec = None

        if pose_eval_sec is not None:
            _timing_register_stage("pose_evaluation", pose_eval_sec)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {type(e).__name__}: {e}")


@app.post("/api/confirm")
def api_confirm_post(req: ConfirmRequest) -> Dict[str, Any]:
    t0 = time.perf_counter()
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

            resp = {
                "action": req.action,
                "confirm_mode": "vision_keyframes",
                "llm_confirm": llm_out,
                "saved_frames": {
                    "session_dir": save_info["session_dir"],
                    "saved_standard_count": save_info["saved_standard_count"],
                    "saved_user_count": save_info["saved_user_count"],
                },
            }
            _timing_register_stage("llm_feedback_generation", _timing_seconds(t0))
            return resp

        llm_out = llm_confirm_judge(req.action, req.eval_result or {})
        resp = {
            "action": req.action,
            "confirm_mode": "json_eval",
            "llm_confirm": llm_out,
        }
        _timing_register_stage("llm_feedback_generation", _timing_seconds(t0))
        return resp

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM confirm POST failed: {type(e).__name__}: {e}")

@app.post("/api/coach_video_v2")
def api_coach_video_v2(req: CoachVideoRequest) -> Dict[str, Any]:
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text is empty")

    t0 = time.perf_counter()
    try:
        display_text = req.text.strip()
        speech_text = llm_shorten_for_digital_human(display_text)

        url = generate_musetalk_feedback_video(
            user_name=req.user_name,
            text=speech_text,
            version=req.version or "v1.5",
            mode=req.mode or "normal",
        )

        resp = {
            "ok": True,
            "url": url,
            "text": display_text,          # 页面仍显示完整反馈
            "speech_text": speech_text,    # 数字人实际使用的短文本
            "user_name": req.user_name,
            "template_video": str(get_user_video_path(req.user_name)),
        }

        _timing_register_stage("digital_human_synthesis", _timing_seconds(t0))
        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"coach_video_v2 failed: {type(e).__name__}: {e}")

@app.get("/api/training/plan/get")
def api_training_plan_get(name: str, action: str, video_id: str) -> Dict[str, Any]:
    try:
        plan = load_train_plan(name, action, video_id)
        return {
            "ok": True,
            "user_name": name,
            "action": action,
            "video_id": video_id,
            "train_plan": plan,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"training plan get failed: {type(e).__name__}: {e}")


@app.post("/api/training/plan/reset")
def api_training_plan_reset(req: TrainPlanResetRequest) -> Dict[str, Any]:
    try:
        plan = default_train_plan()
        save_train_plan(req.user_name, req.action, req.video_id, plan)
        return {
            "ok": True,
            "user_name": req.user_name,
            "action": req.action,
            "video_id": req.video_id,
            "train_plan": plan,
            "message": "当前视频训练计划已重置为第1轮",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"training plan reset failed: {type(e).__name__}: {e}")


@app.post("/api/training/plan/update")
def api_training_plan_update(req: TrainPlanUpdateRequest) -> Dict[str, Any]:
    try:
        plan = load_train_plan(req.user_name, req.action, req.video_id)

        round_id = int(plan.get("current_round", 1))
        max_rounds = int(plan.get("max_rounds", 4))

        if round_id > max_rounds or bool(plan.get("is_finished", False)):
            plan["is_finished"] = True
            save_train_plan(req.user_name, req.action, req.video_id, plan)
            return {
                "ok": True,
                "user_name": req.user_name,
                "action": req.action,
                "video_id": req.video_id,
                "message": "该视频训练已达到最大轮次，不再继续更新。",
                "round_finished": max_rounds,
                "train_plan": plan,
            }

        eval_result = req.eval_result or {}
        llm_confirm = req.llm_confirm or {}

        accuracy = float(eval_result.get("accuracy", 0.0))
        joint_errors = eval_result.get("joint_errors", {}) or {}

        overall_error = eval_result.get("overall_error", None)
        if overall_error is None:
            overall_error = compute_overall_error_from_joint_errors(
                joint_errors,
                joint_weights=plan.get("joint_weights", {"shoulder": 0.7, "elbow": 0.3}),
            )
        overall_error = float(overall_error)

        normalized_error_value = eval_result.get("normalized_error", None)
        if normalized_error_value is None:
            normalized_error_value = normalize_error(overall_error, 0.0, 40.0)
        normalized_error_value = float(normalized_error_value)

        current_target = float(plan.get("current_target", 0.75))
        current_threshold = float(plan.get("current_threshold", 15.0))

        pass_flag = int(
            (accuracy >= current_target) and (overall_error <= current_threshold)
        )

        if llm_confirm.get("is_pass") is True:
            pass_flag = max(pass_flag, 1)

        key_issues = llm_confirm.get("key_issues", []) or []
        if not isinstance(key_issues, list):
            key_issues = []
        key_issue_count = len(key_issues)

        state_value = compute_training_state(
            accuracy=accuracy,
            normalized_error_value=normalized_error_value,
            pass_flag=pass_flag,
            key_issue_count=key_issue_count,
            alpha=float(plan.get("alpha", 0.40)),
            beta=float(plan.get("beta", 0.25)),
            gamma=float(plan.get("gamma", 0.25)),
            delta=float(plan.get("delta", 0.10)),
        )

        updated = update_goal_and_threshold(
            current_target=current_target,
            current_threshold=current_threshold,
            state_value=state_value,
            base_value=float(plan.get("base_value", 0.50)),
            goal_step=float(plan.get("goal_step", 0.08)),
            threshold_step=float(plan.get("threshold_step", 3.0)),
            target_min=float(plan.get("target_min", 0.60)),
            target_max=float(plan.get("target_max", 0.95)),
            threshold_min=float(plan.get("threshold_min", 5.0)),
            threshold_max=float(plan.get("threshold_max", 30.0)),
        )

        history_item = {
            "round": round_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": req.action,
            "video_id": req.video_id,
            "score": float(eval_result.get("score", 0.0)),
            "score_level": str(eval_result.get("score_level", "")),
            "accuracy": accuracy,
            "joint_errors": joint_errors,
            "overall_error": overall_error,
            "normalized_error": normalized_error_value,
            "pass_flag": pass_flag,
            "key_issue_count": key_issue_count,
            "key_issues": key_issues,
            "state_value": state_value,
            "current_target": current_target,
            "current_threshold": current_threshold,
            "next_target": updated["next_target"],
            "next_threshold": updated["next_threshold"],
        }

        history = plan.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append(history_item)

        plan["history"] = history
        plan["current_round"] = round_id + 1
        plan["current_target"] = updated["next_target"]
        plan["current_threshold"] = updated["next_threshold"]
        plan["is_finished"] = int(plan["current_round"]) > max_rounds

        save_train_plan(req.user_name, req.action, req.video_id, plan)

        return {
            "ok": True,
            "user_name": req.user_name,
            "action": req.action,
            "video_id": req.video_id,
            "round_finished": round_id,
            "accuracy": accuracy,
            "overall_error": overall_error,
            "normalized_error": normalized_error_value,
            "pass_flag": pass_flag,
            "key_issue_count": key_issue_count,
            "state_value": state_value,
            "current_target": current_target,
            "current_threshold": current_threshold,
            "next_target": updated["next_target"],
            "next_threshold": updated["next_threshold"],
            "train_plan": plan,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"training plan update failed: {type(e).__name__}: {e}")