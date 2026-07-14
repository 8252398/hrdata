# -*- coding: utf-8 -*-
"""Unified LLM client — OpenAI Compatible API for multiple providers."""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

from config.settings import LLMConfig, LLM_PROVIDERS
from utils.logger import get_logger

logger = get_logger(__name__)


class LLMClient:
    """Unified client for OpenAI-compatible API providers.

    Usage:
        client = LLMClient(api_key="sk-...", provider="deepseek")
        response = client.chat("你好")
    """

    def __init__(
        self,
        api_key: str,
        provider: str = "deepseek",
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        provider_config = LLM_PROVIDERS.get(provider, LLM_PROVIDERS["deepseek"])

        self.api_key = api_key
        self.provider = provider
        self.model = model or provider_config.model
        self.base_url = base_url or provider_config.base_url

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )
        logger.info(
            "LLM client created: provider=%s, model=%s", self.provider, self.model
        )

    def chat(
        self,
        user_message: str,
        system_message: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        timeout: int = 180,
    ) -> str:
        """Send a chat completion request.

        Args:
            user_message: User prompt content.
            system_message: System prompt (optional).
            temperature: Sampling temperature.
            max_tokens: Max output tokens.
            timeout: Request timeout in seconds.

        Returns:
            Model response text.

        Raises:
            RuntimeError: If API call fails.
        """
        messages = []
        if system_message:
            # deepseek-reasoner does not support system role; prepend to user
            messages.append({
                "role": "user",
                "content": f"[System Instructions]\n{system_message}\n\n{user_message}",
            })
        else:
            messages.append({"role": "user", "content": user_message})

        prompt_len = len(user_message) + len(system_message)
        logger.info(
            "Chat request: model=%s, prompt_len=%d, timeout=%d",
            self.model, prompt_len, timeout,
        )

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body={"thinking": {"type": "enabled"}},
            )
            content = response.choices[0].message.content or ""
            logger.info("Chat response: %d chars", len(content))
            return content
        except Exception as exc:
            logger.exception("LLM API call failed")
            raise RuntimeError(f"AI 调用失败: {exc}") from exc
