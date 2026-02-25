# api_server.py
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist

from pose_analysis.evaluator import PoseEvaluator


# -----------------------------
# App
# -----------------------------
app = FastAPI(title="Rehab Platform API", version="0.1.0")

# 先允许本地前端（Next.js 默认 3000）访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent  # D:\system
STANDARD_DIR = BASE_DIR / "demo_action" / "standard_pose"

# ✅ 关键：给 PoseEvaluator 传 standard_dir（解决你截图报错）
evaluator = PoseEvaluator(standard_dir=STANDARD_DIR)


# -----------------------------
# Schemas
# -----------------------------
Point3 = conlist(float, min_length=3, max_length=3)          # [x,y,z]
Pose33 = conlist(Point3, min_length=33, max_length=33)       # 33 points

class Frame(BaseModel):
    pose: Pose33 = Field(..., description="33 landmarks, each [x,y,z]")
    left_hand: Optional[list] = Field(default_factory=list)
    right_hand: Optional[list] = Field(default_factory=list)

class EvaluateRequest(BaseModel):
    action: str = Field("raise_arm", description="action name, e.g. raise_arm")
    user_seq: List[Frame] = Field(..., description="list of frames")

class EvaluateResponse(BaseModel):
    action: str
    dtw_score: float
    accuracy: float
    joint_errors: Dict[str, float]
    rule_based_comments: List[str]


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/actions")
def list_actions():
    """返回 standard_pose 目录下可用动作名（不含 .json 后缀）"""
    if not STANDARD_DIR.exists():
        raise HTTPException(status_code=500, detail=f"STANDARD_DIR not found: {STANDARD_DIR}")
    actions = sorted([p.stem for p in STANDARD_DIR.glob("*.json")])
    return {"actions": actions}

@app.post("/api/evaluate", response_model=EvaluateResponse)
def api_evaluate(req: EvaluateRequest):
    if not req.user_seq:
        raise HTTPException(status_code=400, detail="user_seq is empty")

    # pydantic v2 用 model_dump；为兼容 v1/v2，这里做个安全转换
    user_seq = []
    for f in req.user_seq:
        if hasattr(f, "model_dump"):
            user_seq.append(f.model_dump())
        else:
            user_seq.append(f.dict())

    try:
        result = evaluator.evaluate(req.action, user_seq)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"evaluate failed: {e}")

    # ✅ 注意：返回 dict，不要 json.dumps，否则 Swagger 会显示 "string"
    return result
