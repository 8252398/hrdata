# -*- coding: utf-8 -*-
"""Application configuration — single source of truth.

No hardcoded API keys, base URLs, or file paths in business logic.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional

# ---- Project root ----
ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FONTS_DIR = ROOT / "fonts"
PROMPTS_DIR = ROOT / "prompts"
DB_PATH = DATA_DIR / "training.db"

# ---- Agent turn limits ----
MAX_AGENT_TURNS_DEFAULT = 8
MAX_AGENT_TURNS_EXTEND = 8


# ---- LLM defaults (override via env or UI) ----
@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-pro"
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    temperature: float = 1.0
    max_tokens: int = 4096
    timeout: int = 180
    thinking_enabled: bool = True  # deepseek-v4-pro thinking mode


# ---- Supported providers ----
LLM_PROVIDERS = {
    "deepseek": LLMConfig(
        provider="deepseek",
        model="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
    ),
    "openai": LLMConfig(
        provider="openai",
        model="gpt-4o",
        base_url="https://api.openai.com/v1",
    ),
    "glm": LLMConfig(
        provider="glm",
        model="@cf/zai-org/glm-5.2",
        base_url="https://api.cloudflare.com/client/v4/accounts/b71aa7ad5579027a5027c0428d21e500/ai/v1",
    ),
    "ollama": LLMConfig(
        provider="ollama",
        model="qwen3:6b",
        base_url="http://localhost:11434/v1",
    ),
    "custom": LLMConfig(
        provider="custom",
        model="",
        base_url="",
    ),
}


# ---- Prompt limits ----
MAX_SAMPLE_ROWS = 10
MAX_TOKENS_PROMPT = 8000


# ---- Safe executor ----
ALLOWED_IMPORTS = {
    "pandas", "numpy", "plotly", "matplotlib",
    "math", "datetime", "collections", "itertools",
    "statistics", "json", "csv", "io", "typing",
}

FORBIDDEN_IMPORTS = {
    "os", "subprocess", "socket", "requests", "shutil",
    "pathlib", "sys", "importlib", "builtins",
}

FORBIDDEN_CALLS = {"eval", "exec", "compile", "__import__", "open"}
