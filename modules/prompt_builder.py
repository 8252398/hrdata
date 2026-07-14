# -*- coding: utf-8 -*-
"""Prompt builder — construct prompts from templates and DB schemas."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config.settings import PROMPTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def _load_prompt_file(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.txt"
    if path.exists():
        return path.read_text("utf-8").strip()
    logger.warning("Prompt file not found: %s", path)
    return ""


_DEFAULT_SYSTEM = """你是一名资深 SQL 数据分析师。
你只能生成 SQLite 兼容的 SQL 查询语句。
输出格式：纯 SQL 语句，不要 Markdown 包裹，不要解释。"""

_DEFAULT_SQL_GEN = """## 数据库 Schema
{schema}

## 用户问题
{question}

请生成一条 SQLite SQL 查询。
纯 SQL 语句，不要 Markdown 包裹，不要解释。"""

_DEFAULT_EXPLANATION = """## 用户问题
{question}

## 分析结果摘要
{summary}

请用自然语言总结分析结果。2-4句话。不编造数据。"""


class PromptBuilder:
    """Build prompts for LLM tasks (SQL generation + result explanation)."""

    def __init__(self):
        self.system_prompt = _load_prompt_file("system") or _DEFAULT_SYSTEM
        self.sql_gen_template = _load_prompt_file("code_gen") or _DEFAULT_SQL_GEN
        self.explanation_template = _load_prompt_file("explanation") or _DEFAULT_EXPLANATION

    def build_sql_prompt(
        self,
        schema_text: str,
        question: str,
    ) -> tuple[str, str]:
        """Build (system, user) prompts for SQL generation.

        Args:
            schema_text: Database schema description (tables, columns, types).
            question: User's natural language question.

        Returns:
            (system_prompt, user_prompt) tuple.
        """
        user_prompt = self.sql_gen_template.format(
            schema=schema_text,
            question=question,
        )
        logger.info(
            "SQL prompt built: schema=%d chars, question=%d chars",
            len(schema_text), len(question),
        )
        return self.system_prompt, user_prompt

    def build_explanation_prompt(
        self,
        question: str,
        summary: str,
    ) -> tuple[str, str]:
        """Build prompts for result explanation."""
        system = "你是一位数据分析师。请用自然语言解释分析结果，2-4句话。不编造数据。"
        user_prompt = self.explanation_template.format(
            question=question,
            summary=summary,
        )
        return system, user_prompt
