# -*- coding: utf-8 -*-
"""LLM module - recommendation ranking with JSON output."""

import json as _json
import re

import pandas as pd
from openai import OpenAI


RECOMMENDATION_SYSTEM_PROMPT = (
    "你是一名中央党校干部培训专家，负责干部教育培训的学员遴选与推荐工作。"
    "请仅依据：岗位/干部层级/培训经历/课程名称/累计学时，分析每人是否适合参加。"
    "请不要重新筛选或质疑筛选结果。"
    "请严格输出纯 JSON 数组，不要包含任何 Markdown 标记、不要解释、不要额外文字。"
)

SCORING_GUIDE = (
    "评分标准：90-100=★★★★★ 强烈推荐，"
    "80-89=★★★★☆ 推荐，"
    "70-79=★★★☆☆ 可考虑，"
    "60-69=★★☆☆☆ 匹配度低，"
    "0-59=★☆☆☆☆ 不推荐。"
    "score 必须是 0-100 整数。priority/follow_up/backup 必须是布尔值 true/false。"
)


def _score_to_level(score):
    if score >= 90: return "★★★★★"
    if score >= 80: return "★★★★☆"
    if score >= 70: return "★★★☆☆"
    if score >= 60: return "★★☆☆☆"
    return "★☆☆☆☆"


def build_recommendation_prompt(candidates, training_name, training_goal, max_candidates_in_prompt=50):
    """Build recommendation prompt with per-candidate training history."""
    display = candidates[:max_candidates_in_prompt]
    total = len(candidates)

    rows = []
    for idx, c in enumerate(display, 1):
        # Build recent training summary (last 3 courses)
        recent_summary = ""
        if c.recent_trainings:
            recent_items = []
            for t in c.recent_trainings[:3]:
                recent_items.append(
                    f"{t.get('课程', '')} "
                    f"({t.get('培训方式', '')}, "
                    f"{t.get('开始时间', '')})"
                )
            recent_summary = "; ".join(recent_items)

        rows.append(
            f"{idx}. {c.name} | "
            f"编码:{c.employee_code} | "
            f"级别:{c.cadre_level} | "
            f"单位:{c.unit} | "
            f"部门:{c.department} | "
            f"职务:{c.position} | "
            f"累计学时:{c.total_hours:.0f} | "
            f"近期培训:{recent_summary if recent_summary else '无'}"
        )

    candidate_text = "\n".join(rows)

    return f"""## 培训班信息
- 名称：{training_name}
- 目标：{training_goal}

## 筛选说明
以下 {total} 名候选人已通过 Python 硬规则筛选
（级别过滤、排除已参加同名培训、近5年类似培训过滤、学时排序）。
当前仅展示前 {len(display)} 名，请仅对这些候选人评分。
请不要重新筛选或质疑筛选结果。

## 候选人列表
{candidate_text}

## 输出要求
输出纯 JSON 数组（不要 Markdown 包裹），每人一个对象：

[
  {{
    "employee_code": "编码",
    "name": "姓名",
    "score": 85,
    "level": "★★★★☆",
    "reason": "推荐理由（1-2句中文）",
    "risk": "不推荐理由（1-2句中文，如无则写无）",
    "priority": true,
    "follow_up": false,
    "backup": false
  }}
]

{SCORING_GUIDE}
输出必须是有效 JSON，不要代码块包裹。"""


def ask_ai_recommendation(client, candidates, training_name, training_goal, model="deepseek-v4-pro", timeout=180):
    """Call DeepSeek for recommendation ranking, returns raw JSON text."""
    if not candidates:
        return "[]"

    prompt = build_recommendation_prompt(candidates, training_name, training_goal)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RECOMMENDATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        timeout=timeout,
    )
    return response.choices[0].message.content


def parse_recommendation_json(raw_text):
    """Parse AI JSON response into sortable/filterable DataFrame."""
    text = raw_text.strip()

    # Extract JSON from markdown code block if present
    m = re.search(r"`(?:json)?\s*([\s\S]*?)`", text)
    if m:
        text = m.group(1).strip()

    # Find JSON array boundaries
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        text = text[arr_start:arr_end + 1]

    # Parse
    try:
        data = _json.loads(text)
    except _json.JSONDecodeError:
        try:
            data = _json.loads(text.replace("'", '"'))
        except _json.JSONDecodeError:
            return pd.DataFrame({"error": ["JSON parse failed"], "raw": [raw_text[:500]]})

    if not isinstance(data, list):
        return pd.DataFrame({"error": ["Not an array"], "raw": [raw_text[:500]]})

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        score = int(item.get("score", 0))
        rows.append({
            "employee_code": str(item.get("employee_code", "")),
            "name": str(item.get("name", "")),
            "score": score,
            "level": str(item.get("level", _score_to_level(score))),
            "reason": str(item.get("reason", "")),
            "risk": str(item.get("risk", "")),
            "priority": bool(item.get("priority", False)),
            "follow_up": bool(item.get("follow_up", False)),
            "backup": bool(item.get("backup", False)),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)
    return df
