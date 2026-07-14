# -*- coding: utf-8 -*-
"""SQL Agent — multi-turn database exploration + retrieval.

LLM explores the database structure before generating the final SQL.
Python only executes read-only SQL; all business reasoning is in the LLM.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from modules.sqlite_manager import TrainingDatabase
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_AGENT_TURNS = 8  # prevent infinite loops

# Read-only SQL whitelist
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "ATTACH", "DETACH", "REPLACE",
]


def validate_sql(sql: str) -> tuple[bool, str]:
    """Ensure SQL is read-only (SELECT or PRAGMA).

    Returns:
        (is_safe, reason) tuple.
    """
    upper = sql.strip().upper()

    # PRAGMA is read-only for our purposes
    if upper.startswith("PRAGMA"):
        return True, ""

    if not upper.startswith("SELECT"):
        return False, "仅允许 SELECT 或 PRAGMA 语句"

    for kw in _FORBIDDEN_KEYWORDS:
        # Use word-boundary check
        if re.search(rf"\b{kw}\b", upper):
            return False, f"禁止操作: {kw}"

    return True, ""


class SQLAgent:
    """Multi-turn agent for database exploration and SQL generation.

    Usage:
        agent = SQLAgent()
        agent.run(user_question, llm_client)  # blocks until done
    """

    def __init__(self):
        self._db: Optional[TrainingDatabase] = None
        self._history: list[dict] = []  # {role, content, sql?, result?}

    def run(
        self,
        question: str,
        llm_client,  # LLMClient instance
        db: TrainingDatabase,
        status_writer=None,  # optional st.status() for Streamlit progress
    ) -> dict:
        """Run the full agent loop.

        Args:
            question: User's natural language question.
            llm_client: Configured LLMClient instance.
            db: Connected TrainingDatabase instance.
            status_writer: Optional callback for progress messages.

        Returns:
            dict with keys: sql (final SQL), result (DataFrame),
            explanation (str), turns (int), history (list).
        """
        self._db = db
        self._history = []

        # Step 0: Get schema
        schema_text = self._get_schema()
        self._log(status_writer, "🔍 Agent 启动，探索数据库中...")

        # Build conversation context
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"## 数据库 Schema\n{schema_text}\n\n## 用户问题\n{question}",
            },
        ]

        for turn in range(1, MAX_AGENT_TURNS + 1):
            self._log(status_writer, f"🔄 第 {turn} 轮对话...")

            # Call LLM
            try:
                response = llm_client._client.chat.completions.create(
                    model=llm_client.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=180,
                )
                content = response.choices[0].message.content or ""
            except Exception as exc:
                raise RuntimeError(f"LLM 调用失败 (第{turn}轮): {exc}") from exc

            # Parse: does it contain SQL?
            sql = _extract_sql(content)

            if sql:
                # Validate
                is_safe, reason = validate_sql(sql)
                if not is_safe:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"SQL 校验失败: {reason}。请重新生成只读 SQL。",
                    })
                    self._log(status_writer, f"⚠️ 校验失败: {reason}")
                    continue

                # Execute
                try:
                    result = db.query_to_df(sql)
                    result_summary = (
                        f"查询结果: {len(result)} 行 × {len(result.columns)} 列\n"
                        f"列: {', '.join(list(result.columns)[:10])}\n"
                    )
                    if len(result) <= 20:
                        result_summary += result.to_markdown(index=False)
                    else:
                        result_summary += (
                            f"(仅展示前10行)\n"
                            f"{result.head(10).to_markdown(index=False)}"
                        )

                    self._log(
                        status_writer,
                        f"⚡ SQL 执行: {len(result)} 行 × {len(result.columns)} 列",
                    )
                    self._history.append({
                        "turn": turn, "sql": sql, "rows": len(result),
                        "exploratory": not content.strip().upper().startswith("FINAL:"),
                    })
                except Exception as exc:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": f"SQL 执行失败: {exc}\n" + _diagnose_cte_error(sql) + "\n请修正 SQL 并重试。",
                    })
                    self._log(status_writer, f"❌ 执行失败: {exc}")
                    continue

                # Check if this is the FINAL response
                content_upper = content.strip().upper()
                is_final = (
                    content_upper.startswith("FINAL:")
                    or content_upper.startswith("最终")
                    or "FINAL SQL" in content_upper
                )

                if is_final:
                    self._log(status_writer, "✅ Agent 完成，已生成最终 SQL")
                    # Get explanation
                    explanation = self._get_explanation(
                        llm_client, question, result_summary
                    )
                    return {
                        "sql": sql,
                        "result": result,
                        "explanation": explanation,
                        "turns": turn,
                        "history": self._history,
                    }

                # Not final — feed result back for further exploration
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": f"SQL 执行成功。\n{result_summary}\n\n"
                               f"请继续探索或生成最终 SQL。如果已经获取了足够信息，"
                               f"请以 'FINAL:' 开头输出最终检索 SQL。",
                })
            else:
                # No SQL in response — LLM is thinking/explaining
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "请生成一条 SQL 查询（SELECT 或 PRAGMA）。"
                               "如果已完成探索，请以 'FINAL:' 开头输出最终检索 SQL。",
                })

        # Max turns reached
        self._log(status_writer, "⚠️ 达到最大轮次，返回最后一次结果")
        return {
            "sql": "",
            "result": pd.DataFrame({"提示": ["Agent 未能在最大轮次内完成分析"]}),
            "explanation": "分析超时，请尝试更具体的问题。",
            "turns": MAX_AGENT_TURNS,
            "history": self._history,
        }

    def _get_schema(self) -> str:
        """Build a compact database schema for the LLM."""
        if self._db is None:
            return "Database not connected"

        # Get schema via PRAGMA
        persons_schema = self._db.query_to_df("PRAGMA table_info(persons)")
        training_schema = self._db.query_to_df("PRAGMA table_info(training_records)")

        lines = ["## persons 表"]
        for _, row in persons_schema.iterrows():
            lines.append(f"- {row['name']} ({row['type']})")
        lines.append("")
        lines.append("## training_records 表")
        for _, row in training_schema.iterrows():
            lines.append(f"- {row['name']} ({row['type']})")
        lines.append("")
        lines.append("两表通过 employee_code 关联。")

        # Quick stats
        stats = self._db.get_stats()
        lines.append(f"\n当前数据量: {stats}")

        return "\n".join(lines)

    def _get_explanation(self, llm_client, question: str, summary: str) -> str:
        """Ask LLM to explain the final result."""
        prompt = f"## 用户问题\n{question}\n\n## 分析结果\n{summary}\n\n请用自然语言总结（2-4句话中文）。不编造数据。"
        try:
            return llm_client.chat(
                user_message=prompt,
                system_message="你是一位数据分析师。请简洁解释分析结果。",
            )
        except Exception:
            return "AI 解释生成失败"

    def _log(self, writer, msg: str) -> None:
        """Write progress to Streamlit status or logger."""
        logger.info(msg)
        if writer:
            try:
                writer.write(msg)
            except Exception:
                pass


def _extract_sql(text: str) -> str:
    """Extract SQL from LLM response.

    Strips 'FINAL:' prefix, markdown code blocks, and trailing semicolons.
    """
    text = text.strip()

    # Remove FINAL: prefix
    if text.upper().startswith("FINAL:"):
        text = text[6:].strip()

    # Try to extract from markdown code block
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip().rstrip(";")

    # If the entire text looks like a SQL statement
    upper = text.upper()
    if upper.startswith("SELECT") or upper.startswith("PRAGMA"):
        # Find the end of the SQL (stop at blank line or explanation text)
        lines = text.split("\n")
        sql_lines = []
        for line in lines:
            if not line.strip():
                break
            stripped = line.strip()
            # Stop at natural language
            if (
                stripped.startswith("#")
                or stripped.startswith("--")
                or stripped.startswith("说明")
                or stripped.startswith("解释")
                or stripped.startswith("分析")
            ):
                break
            sql_lines.append(stripped)
        return "\n".join(sql_lines).rstrip(";")

    # Last resort: try to find SELECT/PRAGMA in the text
    for keyword in ["SELECT", "PRAGMA"]:
        idx = upper.find(keyword)
        if idx != -1:
            # Find semicolon or end of this SQL statement
            remaining = text[idx:]
            semi_idx = remaining.find(";")
            if semi_idx != -1:
                return remaining[:semi_idx].strip()
            # Return first non-empty lines
            sql_part = []
            for line in remaining.split("\n"):
                if not line.strip():
                    break
                sql_part.append(line.strip())
            return "\n".join(sql_part)

    return text.rstrip(";")


def _diagnose_cte_error(sql: str) -> str:
    """Detect common CTE syntax errors and return targeted fix hint."""
    # Pattern: ) , name AS ( without preceding WITH
    import re
    if re.search(r"\)\s*,\s*\w+\s+AS\s*\(", sql, re.IGNORECASE):
        if not sql.strip().upper().startswith("WITH"):
            return (
                "SQL 语法错误: 缺少 WITH 关键字。"
                "你的 SQL 包含多个 CTE 定义 (xxx AS (...))，"
                "但忘记了在第一个 CTE 前面写 WITH。请修正为:\n"
                "WITH cte1 AS (SELECT ...), cte2 AS (SELECT ...) SELECT ..."
            )
    return ""


# ═══════════════════════════════════════════════════════════
# System Prompt
# ═══════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """你是一名 SQLite 数据分析助手。

## 核心原则
先观察数据库，再推理，最后查询。不要凭空猜测。

## 工作流程
1. 查看数据库结构（PRAGMA table_info）
2. 探索实际数据（SELECT DISTINCT 查看职位、部门、课程等字段的真实值）
3. 根据真实数据推理业务概念（如"班子成员"可能是哪些职位）
4. 生成最终检索 SQL（以 'FINAL:' 开头）

## 规则
- 不确定字段名 → 先 PRAGMA table_info
- 不确定有哪些职位 → 先 SELECT DISTINCT position
- 不确定有哪些部门 → 先 SELECT DISTINCT department
- 不确定课程类别 → 先 SELECT DISTINCT course_name
- 不确定培训方式 → 先 SELECT DISTINCT training_method
- 遇到模糊概念（班子成员、中层干部、年轻干部、数字化课程等）→ 先查数据库
- 宁可多查，不要乱猜
- 宁可扩大召回，不要漏掉相关数据

## 输出格式
- 探索阶段：直接输出 SQL（SELECT 或 PRAGMA）
- 最终阶段：以 'FINAL:' 开头，后跟最终检索 SQL
- 不要包裹在 Markdown 代码块中
- 不要在 SQL 后面写解释文字

## 示例
用户: 统计集团班子成员近三年的培训情况

你的回复:
PRAGMA table_info(persons)

(收到结果后)
SELECT DISTINCT cadre_flag FROM persons

(继续探索)
SELECT DISTINCT position FROM persons WHERE position IS NOT NULL ORDER BY position

(推理：班子成员包括总经理、副总经理、董事长等...)
FINAL:
SELECT p.name, p.position, SUM(t.hours) as total_hours
FROM persons p
JOIN training_records t ON p.employee_code = t.employee_code
WHERE p.position IN ('总经理','副总经理','董事长','执行董事')
AND t.start_date >= '2023-01-01'
GROUP BY p.employee_code
ORDER BY total_hours DESC

## CTE 写法示例
如果查询涉及多个中间步骤，必须使用 WITH 关键字定义 CTE。
正例（正确）：
WITH step1 AS (
  SELECT employee_code, SUM(hours) as total FROM training_records GROUP BY employee_code
), step2 AS (
  SELECT employee_code FROM step1 WHERE total < 440
)
SELECT * FROM training_records WHERE employee_code IN (SELECT employee_code FROM step2)

误例（错误）：
step1 AS (SELECT ...)
-- 缺少 WITH 关键字，SQLite 报: near ")" syntax error

记住：
- CTE 必须以 WITH 开头
- 多个 CTE 用逗号分隔
- 最后一个 CTE 后面紧接主查询 SELECT
- 生成 SQL 后自行检查: 是否有 AS ( 但没有前置 WITH ?"""
