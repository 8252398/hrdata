# -*- coding: utf-8 -*-
"""SQL Agent — multi-turn database exploration + retrieval.

LLM explores the database structure before generating the final SQL.
Python only executes read-only SQL; all business reasoning is in the LLM.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

from modules.prompt_builder import PromptBuilder
from modules.sqlite_manager import TrainingDatabase
from utils.logger import get_logger

logger = get_logger(__name__)

MAX_AGENT_TURNS = 8  # prevent infinite loops

# Fuzzy business concepts that MUST be explored via SELECT DISTINCT before FINAL
_FUZZY_CONCEPTS = [
    "班子成员", "中层干部", "技术人员", "年轻干部", "管理人员",
    "领导干部", "集团领导", "核心人员", "关键岗位", "骨干",
    "干部", "领导", "班子", "管理层", "高管",
]

# Fields that should be explored when fuzzy concepts appear
_EXPLORATION_FIELDS = {
    "cadre_flag": ("persons", "干部标识/身份标签"),
    "organizer": ("training_records", "主办单位/培训机构"),
    "training_type": ("training_records", "培训类型"),
    "department": ("persons", "部门名称"),
    "unit": ("persons", "单位名称"),
}

# Required columns for FINAL SQL result (must have detailed fields for human audit)
_REQUIRED_COLUMNS = {
    "employee_code": "人员编码",
    "name": "姓名",
    "unit": "单位",
    "department": "部门",
    "cadre_flag": "干部标识",
    "course_name": "培训名称",
    "hours": "学时",
    "training_type": "培训类型",
    "organizer": "主办单位",
    "institution": "培训机构",
    "start_date": "开始时间",
    "end_date": "完成时间",
}

# Minimum column count for FINAL SQL to prevent single-field returns
_MIN_FINAL_COLUMNS = 8

# Read-only SQL whitelist
_FORBIDDEN_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER",
    "CREATE", "ATTACH", "DETACH", "REPLACE",
]


def _check_mandatory_exploration(
    question: str,
    history: list[dict],
    current_sql: str,
) -> str:
    """Check whether required exploration has been performed before FINAL.

    Returns:
        Empty string if exploration is sufficient, otherwise a message
        describing what exploration is still required.
    """
    upper_q = question.upper()

    # Detect which fuzzy concepts appear in the question
    matched_concepts = [c for c in _FUZZY_CONCEPTS if c in question]
    if not matched_concepts:
        return ""  # no fuzzy concepts, no mandatory exploration

    # Check what DISTINCT explorations have already been executed
    explored_fields = set()
    for h in history:
        if not h.get("exploratory", False):
            continue
        sql_upper = h.get("sql", "").upper()
        for field in _EXPLORATION_FIELDS:
            if f"DISTINCT {field.upper()}" in sql_upper:
                explored_fields.add(field)

    # Determine which fields need exploration based on matched concepts
    required_fields = set()
    # cadre_flag / 干部相关
    if any(c in question for c in ["班子成员", "中层干部", "干部", "领导", "班子", "管理层", "高管", "集团领导"]):
        required_fields.add("cadre_flag")
    # organizer / 培训单位相关
    if any(c in question for c in ["中组部", "调训", "主办单位", "培训机构", "党校"]):
        required_fields.add("organizer")
    # department / 部门相关
    if any(c in question for c in ["部门"]):
        required_fields.add("department")
    # unit / 单位相关
    if any(c in question for c in ["单位"]):
        required_fields.add("unit")
    # training_type / 培训类型相关
    if any(c in question for c in ["培训类型", "培训方式", "学习类型"]):
        required_fields.add("training_type")

    missing = required_fields - explored_fields
    if not missing:
        return ""

    messages = []
    for field in sorted(missing):
        table, desc = _EXPLORATION_FIELDS[field]
        messages.append(
            f"请先执行: SELECT DISTINCT {field} FROM {table} ORDER BY {field} "
            f"（查看{desc}的真实值）"
        )
    return "; ".join(messages)


def _check_final_columns(result_df, is_final: bool) -> str:
    """Check whether FINAL SQL result contains required detailed columns.

    Returns:
        Empty string if columns are sufficient, otherwise a message
        describing what is missing.
    """
    if not is_final:
        return ""

    cols = [c.lower() for c in result_df.columns]

    # Must have enough columns (prevent single-field return like SELECT cadre_flag)
    if len(cols) < _MIN_FINAL_COLUMNS:
        return (
            f"结果只有 {len(cols)} 列，至少需要 {_MIN_FINAL_COLUMNS} 列以上明细字段。"
            f"当前列: {', '.join(result_df.columns)}"
        )

    missing = []
    for col_key, col_desc in _REQUIRED_COLUMNS.items():
        if col_key not in cols:
            missing.append(f"{col_desc} ({col_key})")

    if missing:
        return f"缺少字段: {', '.join(missing)}"

    return ""


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

    def run_iter(
        self,
        question: str,
        llm_client,  # LLMClient instance
        db: TrainingDatabase,
        status_writer=None,  # optional st.status() for Streamlit progress
    ):
        """Generator-based agent loop with human-in-the-loop support.

        Yields event dicts. The caller drives the loop.

        Event types:
            - {"type": "status", "msg": str}
            - {"type": "sql", "sql": str, "rows": int, "exploratory": bool}
            - {"type": "ask", "text": str}   ← human must answer via .send()
            - {"type": "result", "sql": str, "result": DataFrame,
               "explanation": str, "turns": int, "history": list}

        Usage:
            gen = agent.run_iter(...)
            event = next(gen)                # drive until ask/result
            if event["type"] == "ask":
                answer = get_human_input()
                event = gen.send(answer)     # resume with answer
        """
        self._db = db
        self._history = []

        # Step 0: Get schema and build prompt
        schema_text = self._get_schema()
        yield {"type": "status", "msg": "🔍 Agent 启动，探索数据库中..."}

        prompt_builder = PromptBuilder()
        system_prompt, initial_user = prompt_builder.build_sql_prompt(
            schema_text=schema_text,
            question=question,
        )

        # deepseek-reasoner does not support system role; prepend to first user message
        messages = [
            {
                "role": "user",
                "content": f"[System Instructions]\n{system_prompt}\n\n{initial_user}",
            },
        ]

        _human_answer = None  # holds answer from .send() when agent asks

        for turn in range(1, MAX_AGENT_TURNS + 1):
            yield {"type": "status", "msg": f"🔄 第 {turn} 轮对话..."}

            # Inject human answer if we resumed from an ASK event
            if _human_answer is not None:
                messages.append({"role": "user", "content": _human_answer})
                _human_answer = None

            # Trim conversation if it grows too long
            messages = self._trim_messages(messages)

            # Call LLM
            try:
                content = llm_client.chat_messages(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=4096,
                    timeout=180,
                )
            except Exception as exc:
                raise RuntimeError(f"LLM 调用失败 (第{turn}轮): {exc}") from exc

            # ---- Human-in-the-loop: detect ASK: prefix ----
            ask_match = _extract_ask(content)
            if ask_match:
                yield {"type": "status", "msg": f"❓ AI 提问: {ask_match}"}
                # Yield the question and WAIT for human answer via .send()
                _human_answer = yield {"type": "ask", "text": ask_match}
                # Record the assistant's question + human answer in history
                messages.append({"role": "assistant", "content": content})
                continue

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
                    yield {"type": "status", "msg": f"⚠️ 校验失败: {reason}"}
                    continue

                # Preprocess: fix common LLM-generated syntax issues
                fixed_sql, fix_notes = _preprocess_sql(sql)
                if fixed_sql != sql:
                    sql = fixed_sql
                    for note in fix_notes:
                        yield {"type": "status", "msg": f"🔧 {note}"}

                # Execute (supports multiple ;-separated statements)
                try:
                    result, batch_summary = _execute_batch(db, sql)
                    result_summary = (
                        (batch_summary + "\n" if batch_summary else "")
                        + f"查询结果: {len(result)} 行 x {len(result.columns)} 列\n"
                        + f"列: {', '.join(list(result.columns)[:10])}\n"
                    )
                    if len(result) <= 20:
                        result_summary += result.to_markdown(index=False)
                    else:
                        result_summary += (
                            f"(仅展示前10行)\n"
                            f"{result.head(10).to_markdown(index=False)}"
                        )

                    yield {
                        "type": "status",
                        "msg": f"⚡ SQL 执行: {len(result)} 行 × {len(result.columns)} 列",
                    }
                    self._history.append({
                        "turn": turn, "sql": sql, "rows": len(result),
                        "exploratory": not content.strip().upper().startswith("FINAL:"),
                    })
                    yield {
                        "type": "sql",
                        "sql": sql,
                        "rows": len(result),
                        "exploratory": not content.strip().upper().startswith("FINAL:"),
                    }
                except Exception as exc:
                    error_msg = str(exc)
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"SQL 执行失败: {error_msg}\n"
                            f"请修正 SQL 并重试。常见错误检查清单：\n"
                            f"1. 如果使用 CTE，必须以 WITH 开头，例如：WITH cte1 AS (SELECT ...), cte2 AS (...) SELECT ...\n"
                            f"2. 比较运算符必须使用 <= 和 >=，不要使用 Unicode 符号 ≤ 或 ≥\n"
                            f"3. 字符串值必须用单引号包裹\n"
                            f"4. cadre_flag 字段中可能包含用 '|' 连接的多个身份标签，不要将其拆分，应作为完整字符串匹配\n"
                            f"5. 最终 SQL 必须返回人员编码、姓名、单位、培训名称、学时、主办单位、培训机构、开始/完成时间等明细字段"
                        ),
                    })
                    yield {"type": "status", "msg": f"❌ 执行失败: {exc}"}
                    continue

                # Check if this is the FINAL response
                content_upper = content.strip().upper()
                is_final = content_upper.startswith("FINAL:")

                # ---- Mandatory exploration gate ----
                # If user question contains fuzzy concepts, enforce exploration
                # before accepting any FINAL SQL.
                missing_exploration = _check_mandatory_exploration(
                    question, self._history, sql
                )
                if is_final and missing_exploration:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"⚠️ 强制探索要求：{missing_exploration}\n"
                            f"用户问题中包含模糊业务概念，"
                            f"在输出 FINAL SQL 之前，必须先通过 SELECT DISTINCT 查看数据库中的真实值。"
                            f"请先生成探索性 SQL 查看真实数据，然后再输出 FINAL SQL。"
                        ),
                    })
                    yield {
                        "type": "status",
                        "msg": f"🚫 强制探索: {missing_exploration}",
                    }
                    continue

                # ---- Mandatory result column gate ----
                # If this is FINAL, enforce detailed column requirements.
                missing_cols = _check_final_columns(result, is_final)
                if is_final and missing_cols:
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"⚠️ 最终结果字段不完整：{missing_cols}\n"
                            f"最终 SQL 必须返回完整的明细信息以便人工审核，至少包含：\n"
                            f"- 人员编码 (employee_code)\n"
                            f"- 姓名 (name)\n"
                            f"- 单位 (unit)\n"
                            f"- 部门 (department)\n"
                            f"- 干部标识 (cadre_flag)\n"
                            f"- 培训名称 (course_name)\n"
                            f"- 学时 (hours)\n"
                            f"- 培训类型 (training_type)\n"
                            f"- 主办单位 (organizer)\n"
                            f"- 培训机构 (institution)\n"
                            f"- 开始时间 (start_date)\n"
                            f"- 完成时间 (end_date)\n"
                            f"\n"
                            f"当前结果只有 {len(result.columns)} 列：{', '.join(result.columns)}\n"
                            f"请使用 JOIN 关联 persons 和 training_records 表，"
                            f"在 SELECT 中显式列出上述所有字段，"
                            f"然后重新输出 FINAL SQL。"
                        ),
                    })
                    yield {
                        "type": "status",
                        "msg": f"🚫 字段不完整: {missing_cols}",
                    }
                    continue

                if is_final or turn == MAX_AGENT_TURNS:
                    yield {"type": "status", "msg": "✅ Agent 完成，已生成最终 SQL"}
                    # Get explanation
                    explanation = self._get_explanation(
                        llm_client, question, result_summary
                    )
                    yield {
                        "type": "result",
                        "sql": sql,
                        "result": result,
                        "explanation": explanation,
                        "turns": turn,
                        "history": self._history,
                    }
                    return

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

        # Max turns reached — return last result if any
        yield {"type": "status", "msg": "⚠️ 达到最大轮次，返回最后一次结果"}
        if self._history:
            last = self._history[-1]
            last_sql = last["sql"]
            last_result = db.query_to_df(last_sql)
            last_summary = f"结果: {len(last_result)} 行 x {len(last_result.columns)} 列"
            explanation = self._get_explanation(llm_client, question, last_summary)
            yield {
                "type": "result",
                "sql": last_sql,
                "result": last_result,
                "explanation": explanation,
                "turns": MAX_AGENT_TURNS,
                "history": self._history,
            }
            return
        yield {
            "type": "result",
            "sql": "",
            "result": pd.DataFrame({"提示": ["Agent 未能在最大轮次内完成分析"]}),
            "explanation": "分析超时，请尝试更具体的问题。",
            "turns": MAX_AGENT_TURNS,
            "history": self._history,
        }

    def run(
        self,
        question: str,
        llm_client,  # LLMClient instance
        db: TrainingDatabase,
        status_writer=None,  # optional st.status() for Streamlit progress
    ) -> dict:
        """Run the full agent loop (backward-compatible, blocking).

        Args:
            question: User's natural language question.
            llm_client: Configured LLMClient instance.
            db: Connected TrainingDatabase instance.
            status_writer: Optional callback for progress messages.

        Returns:
            dict with keys: sql (final SQL), result (DataFrame),
            explanation (str), turns (int), history (list).
        """
        gen = self.run_iter(
            question=question,
            llm_client=llm_client,
            db=db,
            status_writer=status_writer,
        )
        result = None
        try:
            while True:
                event = next(gen)
                if event["type"] == "ask":
                    # In blocking mode, auto-reply telling the AI to continue on its own
                    event = gen.send("请继续基于现有信息进行分析，不需要额外人工输入。")
                elif event["type"] == "result":
                    result = event
                    break
        except StopIteration:
            pass
        if result is None:
            return {
                "sql": "",
                "result": pd.DataFrame({"提示": ["Agent 未能在最大轮次内完成分析"]}),
                "explanation": "分析超时，请尝试更具体的问题。",
                "turns": MAX_AGENT_TURNS,
                "history": self._history,
            }
        return result

    def _get_schema(self) -> str:
        """Build a compact database schema for the LLM."""
        if self._db is None:
            return "Database not connected"

        # Column descriptions for the LLM
        _COLUMN_DESC = {
            "employee_code": "集团员工编码（人员唯一标识）",
            "name": "姓名",
            "phone": "商网手机号",
            "unit": "单位名称",
            "department": "部门名称",
            "cadre_flag": "干部标识",
            "id": "培训记录自增ID",
            "course_name": "培训/课程名称（来源信息）",
            "hours": "学时",
            "study_type": "学习类型",
            "training_type": "培训类型",
            "training_method": "培训方式",
            "organizer": "主办单位",
            "institution": "培训机构",
            "start_date": "开始学习时间",
            "end_date": "完成学习时间",
        }

        # Get schema via PRAGMA
        persons_schema = self._db.query_to_df("PRAGMA table_info(persons)")
        training_schema = self._db.query_to_df("PRAGMA table_info(training_records)")

        lines = ["## persons 表"]
        for _, row in persons_schema.iterrows():
            col = row["name"]
            desc = _COLUMN_DESC.get(col, "")
            lines.append(f"- {col} ({row['type']}){f' — {desc}' if desc else ''}")
        lines.append("")
        lines.append("## training_records 表")
        for _, row in training_schema.iterrows():
            col = row["name"]
            desc = _COLUMN_DESC.get(col, "")
            lines.append(f"- {col} ({row['type']}){f' — {desc}' if desc else ''}")
        lines.append("")
        lines.append("两表通过 employee_code 关联。")

        # Quick stats
        stats = self._db.get_stats()
        lines.append(f"\n当前数据量: {stats}")

        return "\n".join(lines)

    def _get_explanation(self, llm_client, question: str, summary: str) -> str:
        """Ask LLM to explain the final result."""
        prompt_builder = PromptBuilder()
        system, user = prompt_builder.build_explanation_prompt(
            question=question,
            summary=summary,
        )
        try:
            return llm_client.chat(
                user_message=user,
                system_message=system,
            )
        except Exception:
            logger.exception("Failed to generate explanation")
            return "AI 解释生成失败"

    def _trim_messages(
        self,
        messages: list[dict],
        max_tokens: int = 6000,
        keep_recent: int = 4,
    ) -> list[dict]:
        """Trim conversation history to avoid token overflow.

        Strategy:
        - Always keep the first message (system instructions + schema + question).
        - If total estimated tokens exceed max_tokens, drop oldest assistant/user
          pairs while keeping the most recent `keep_recent` pairs.
        """
        if len(messages) <= 2:
            return messages

        # Estimate tokens: Chinese-heavy text -> ~3 chars/token, English -> ~4 chars/token.
        # Use a conservative mixed estimate of 3.5 chars per token.
        def _estimate_tokens(msg_list: list[dict]) -> int:
            total_chars = sum(len(m.get("content", "")) for m in msg_list)
            return int(total_chars / 3.5)

        if _estimate_tokens(messages) <= max_tokens:
            return messages

        # Keep first message, then trim from the middle.
        first = [messages[0]]
        rest = messages[1:]

        # Each exploration "turn" contributes an assistant + user pair.
        # Keep the most recent pairs, drop older ones.
        pairs = []
        i = 0
        while i + 1 < len(rest):
            pairs.append((rest[i], rest[i + 1]))
            i += 2

        kept_pairs = pairs[-keep_recent:] if len(pairs) > keep_recent else pairs
        trimmed = first + [msg for pair in kept_pairs for msg in pair]

        # Add a reminder that earlier context was summarized away
        if len(pairs) > keep_recent:
            trimmed.append({
                "role": "user",
                "content": "（前面几轮的探索结果已省略，请基于最近的探索和数据库 Schema 继续分析。）",
            })

        logger.warning(
            "Agent conversation trimmed: %d -> %d messages, est_tokens=%d",
            len(messages), len(trimmed), _estimate_tokens(trimmed),
        )
        return trimmed

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




def _extract_ask(text: str) -> str:
    """Extract human question from LLM response if it starts with ASK:.

    Returns the question text, or empty string if no ASK: prefix found.
    """
    text = text.strip()
    upper = text.upper()
    # Match ASK: at the very beginning, or after a newline
    for prefix in ["ASK:", "问：", "提问:", "QUESTION:", "问题："]:
        idx = upper.find(prefix)
        if idx == 0 or (idx > 0 and text[idx - 1] == "\n"):
            return text[idx + len(prefix):].strip().split("\n")[0].strip()
    return ""


def _split_statements(sql: str) -> list[str]:
    """Split multi-statement SQL into individual statements by semicolon."""
    parts = sql.split(";")
    return [p.strip() for p in parts if p.strip()]


def _preprocess_sql(sql: str) -> tuple[str, list[str]]:
    """Fix common syntax issues in LLM-generated SQL before execution.

    Returns:
        (fixed_sql, list_of_fix_descriptions).
    """
    notes: list[str] = []
    fixed = sql

    # Replace Unicode comparison operators with ASCII equivalents
    if "≤" in fixed or "≥" in fixed:
        fixed = fixed.replace("≤", "<=").replace("≥", ">=")
        notes.append("将 Unicode 比较运算符 ≤/≥ 替换为 <=/>=")

    # Fix missing WITH before CTE definitions.
    # Pattern: "SELECT ... ) , cte_name AS ( ..." means the first SELECT
    # should have been wrapped in a CTE with a WITH prefix.
    stripped = fixed.strip()
    upper = stripped.upper()
    if not upper.startswith("WITH") and not upper.startswith("PRAGMA"):
        # Detect orphaned CTE pattern: ) , name AS (
        m = re.search(r"\)\s*,\s*(\w+)\s+AS\s*\(", stripped, re.IGNORECASE)
        if m:
            closing_paren_pos = m.start()  # position of the first ')'
            # Find the start of the first SELECT
            select_start = upper.find("SELECT")
            if select_start != -1 and select_start < closing_paren_pos:
                first_query = stripped[select_start:closing_paren_pos].strip().rstrip(";")
                # The rest starts after the first ')'
                rest = stripped[closing_paren_pos + 1:].lstrip(",").strip()

                # Try to infer the intended name for the first CTE.
                # Look at the final SELECT's FROM/JOIN aliases; if one alias
                # is not defined as a CTE later, it's likely the first CTE name.
                defined_ctes = {c.lower() for c in re.findall(r"(\w+)\s+AS\s*\(", rest, re.IGNORECASE)}
                final_aliases = set(re.findall(r"\bFROM\s+(\w+)|\bJOIN\s+(\w+)", upper[closing_paren_pos:]))
                # flatten tuples from the regex above
                final_aliases = {a for tup in final_aliases for a in tup if a}
                inferred_name = None
                for alias in final_aliases:
                    if alias.upper() not in {"PERSONS", "TRAINING_RECORDS"} and alias.lower() not in defined_ctes:
                        inferred_name = alias
                        break

                if inferred_name:
                    first_cte_name = inferred_name
                else:
                    # Fallback: use the first CTE name found in the rest
                    name_match = re.match(r"^(\w+)\s+AS\s*\(", rest, re.IGNORECASE)
                    first_cte_name = name_match.group(1) if name_match else "auto_cte_0"

                fixed = f"WITH {first_cte_name} AS (\n  {first_query}\n), {rest}"
                notes.append(f'为首个 CTE 自动补全 WITH 关键字，命名为 "{first_cte_name}"')

    return fixed, notes


def _execute_batch(db, sql: str):
    """Execute potentially multi-statement SQL.
    
    If multiple statements detected, executes each one independently.
    Returns (last_result_df, combined_summary_str) tuple.
    """
    import pandas as pd

    stmts = _split_statements(sql)
    if len(stmts) <= 1:
        result = db.query_to_df(sql)
        return result, ""

    summaries = []
    last_result = None
    for i, stmt in enumerate(stmts, 1):
        try:
            r = db.query_to_df(stmt)
            last_result = r
            summaries.append(
                f"  [{i}] {stmt[:60]}... -> {len(r)} 行 x {len(r.columns)} 列"
            )
        except Exception as exc:
            summaries.append(f"  [{i}] {stmt[:60]}... -> FAIL: {exc}")

    summary = (
        f"Batch executed {len(stmts)} SQLs:\n"
        + "\n".join(summaries)
    )
    return (
        last_result if last_result is not None else pd.DataFrame({"info": ["All done"]}),
        summary,
    )


