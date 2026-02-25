# llm/llm_agent.py
import os
import json
from typing import Any, Dict, Optional

from openai import OpenAI


class LLM_Agent:
    """
    支持两种方式（任选其一）：

    A) DashScope(Qwen) OpenAI兼容接口（你现有方式）
       - set DASHSCOPE_API_KEY=xxx
       - (可选) set LLM_MODEL=qwen2.5-32b-instruct
       - base_url 默认 dashscope

    B) OpenAI 官方
       - set OPENAI_API_KEY=xxx
       - (可选) set LLM_MODEL=gpt-4o-mini
       - base_url 不设置（走默认）
    注意：API Key 必须放在后端环境变量里，不要放前端。
    """

    def __init__(self):
        # 优先读取通用变量；否则兼容你原来的 DASHSCOPE_API_KEY
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "未找到 API Key。请设置 LLM_API_KEY / OPENAI_API_KEY / DASHSCOPE_API_KEY 之一。"
            )

        base_url = os.getenv("LLM_BASE_URL")
        if not base_url and os.getenv("DASHSCOPE_API_KEY"):
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        # 默认模型：DashScope 用 qwen；OpenAI 用 gpt-4o-mini（你也可以改）
        model_name = os.getenv("LLM_MODEL")
        if not model_name:
            if base_url:
                model_name = "qwen2.5-32b-instruct"
            else:
                model_name = "gpt-4o-mini"

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name

    def judge(self, evaluation: Dict[str, Any], user_profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        输入：你的数值评估 evaluation（DTW/accuracy/joint_errors/comments）
        输出：LLM 的结构化评估（用于前端展示）
        """
        # 兼容旧字段
        auto_comments = evaluation.get("rule_based_comments") or evaluation.get("comments") or []
        evaluation = dict(evaluation)
        evaluation["rule_based_comments"] = auto_comments

        schema_hint = {
            "pass": True,
            "score_0_100": 85,
            "summary": "总体完成较好，但肘部略弯曲，建议抬臂到耳旁并停顿2秒。",
            "key_problems": ["肘部伸展不足", "抬臂高度略低"],
            "drills": ["每次抬到耳旁停2秒，做8次", "对镜练习保持肩不过度耸起"],
            "safety_note": "如出现肩痛或麻木，请停止并咨询医生/治疗师。",
            "encouragement": "很好，坚持练习会越来越稳！"
        }

        prompt = f"""
你是一名专业的上肢康复治疗师 + 动作评测员。
我会给你一份“系统数值评估结果”（包含 accuracy、dtw_score、joint_errors、规则提示）。
请你输出【严格 JSON】（不要 markdown，不要多余文字），字段必须包含：

- pass: boolean（是否合格）
- score_0_100: int（0~100）
- summary: string（2~3句话，口语化、可执行）
- key_problems: string[]（1~4条，具体问题）
- drills: string[]（2~4条，具体训练方法/节奏/停顿）
- safety_note: string（安全提示一句）
- encouragement: string（鼓励一句）

判定建议：
- accuracy >= 0.80 通常 pass；accuracy 0.65~0.80 视 joint_errors 与 comments 决定；<0.65 通常不合格。
- 反馈要温和鼓励、可执行，避免空话。

下面是你的输入 evaluation JSON（只用于参考，不要原样复述）：
{json.dumps(evaluation, ensure_ascii=False)}
"""

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()

        # 尝试解析 JSON；解析失败则降级返回纯文本
        try:
            data = json.loads(text)
            # 基本兜底
            for k in ["pass", "score_0_100", "summary", "key_problems", "drills", "safety_note", "encouragement"]:
                if k not in data:
                    raise ValueError(f"Missing key: {k}")
            return {"ok": True, "coach": data, "raw": text}
        except Exception:
            return {"ok": False, "coach": None, "raw": text}
