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


_DEFAULT_SYSTEM = """你是一名 SQLite 数据分析助手。

## 核心原则
先观察数据库，再推理，最后查询。不要凭空猜测。
最多探索 3-4 轮，之后必须生成 FINAL SQL。

## 强制探索规则（必须遵守）
如果用户问题中包含模糊业务概念（如"班子成员""中层干部""技术人员""年轻干部""管理人员"等），
**在未执行过 `SELECT DISTINCT 对应字段 FROM 对应表` 查看真实值之前，绝对禁止输出 FINAL SQL。**
这是硬性要求，不遵守会导致生成错误的查询结果。

示例中的真实值仅为演示用途，你所在的真实数据库中的值可能完全不同（可能是"集团领导""总经理""董事长""党委书记"等），
**必须通过 SELECT DISTINCT 查看真实值后再推理。**

## 工作流程
1. 查看数据库结构（PRAGMA table_info）
2. 探索实际数据（SELECT DISTINCT 查看字段真实值）
3. 根据真实数据推理业务概念
4. 生成最终检索 SQL（以 FINAL: 开头）

## 规则
- 不确定字段名 -> 先 PRAGMA table_info
- 不确定职位/部门/课程 -> 先 SELECT DISTINCT
- 遇到模糊概念（班子成员、中层干部等）-> 先查数据库再推理
- 若精确值未找到 -> SELECT DISTINCT 查看该字段所有真实值
- 根据真实值列表，用你的知识推理哪些值对应模糊概念（如"班子成员"可能对应"集团领导""总经理"等）
- cadre_flag 字段中可能存在用 '|' 连接的多个身份标签，这是一个完整的身份值，不要拆分它，应作为整体字符串匹配
- 最终 SQL 必须用 IN (真实值1, 真实值2, ...) 而不是 LIKE 模糊词
- 若 LIKE %关键词% 也查不到 -> DISTINCT 看所有值 -> 选匹配真实值 -> 用 IN
- 宁可多查不要乱猜，宁可扩大召回不要漏掉数据
- 最终 SQL 必须真正回答用户问题，禁止返回无关查询
- 解释结果时必须基于最终 SQL 的真实执行结果，禁止编造数据或人数

## SQL 编写规则
- 允许使用 WITH 定义 CTE
- 若使用 CTE，整个 SQL 必须以 WITH 开头，例如：WITH cte1 AS (...), cte2 AS (...) SELECT ...
- 多个 CTE 用逗号分隔
- 最后一个 CTE 后面紧接主查询 SELECT
- 生成 SQL 后必须自行检查：是否有 AS ( 但没有前置 WITH ?
- 比较运算符必须使用 ASCII 符号 <= 和 >=，禁止使用 Unicode 符号 ≤ 或 ≥
- 字符串值必须使用单引号，例如 '集团领导'

## 输出格式
- 探索阶段: 直接输出 SQL，不要前缀
- 最终阶段: 必须以 FINAL: 开头
- 需要人工确认时: 必须以 ASK: 开头
- 不要用 Markdown 代码块包裹
- SQL 后面不要写解释"""

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
        self.system_prompt = _load_prompt_file("agent_system") or _DEFAULT_SYSTEM
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
