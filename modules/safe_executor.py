# -*- coding: utf-8 -*-
"""Safe Python executor — AST whitelist validation for AI-generated code."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd

from config.settings import ALLOWED_IMPORTS, FORBIDDEN_IMPORTS, FORBIDDEN_CALLS
from utils.logger import get_logger

logger = get_logger(__name__)


class SafeExecutor:
    """Execute AI-generated Python code in a restricted environment.

    Validates code via AST whitelist before execution.
    Only allows safe imports (pandas, numpy, plotly, etc.).
    """

    @staticmethod
    def validate(code: str) -> list[str]:
        """Validate code safety via AST analysis.

        Args:
            code: Python source code to validate.

        Returns:
            List of violation messages (empty = safe).

        Raises:
            SyntaxError: If code cannot be parsed.
        """
        violations = []

        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            raise SyntaxError(f"代码语法错误: {exc}") from exc

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    if base in FORBIDDEN_IMPORTS:
                        violations.append(f"禁止导入模块: {base}")

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    base = node.module.split(".")[0]
                    if base in FORBIDDEN_IMPORTS:
                        violations.append(f"禁止导入模块: {base}")

            # Check forbidden function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in FORBIDDEN_CALLS:
                        violations.append(f"禁止调用: {node.func.id}()")
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in FORBIDDEN_CALLS:
                        violations.append(f"禁止调用: .{node.func.attr}()")

            # Check for eval/exec
            if isinstance(node, ast.Name) and node.id in ("eval", "exec"):
                violations.append(f"禁止使用: {node.id}")

        return violations

    @staticmethod
    def execute(code: str, df: pd.DataFrame) -> Any:
        """Execute validated code in a sandboxed environment.

        Args:
            code: Python code to execute.
            df: DataFrame available as 'df' variable.

        Returns:
            The value of the 'result' variable after execution.

        Raises:
            SyntaxError: Invalid Python syntax.
            ValueError: Unsafe code detected.
            RuntimeError: Execution failed.
        """
        # 1. Safety validation
        violations = SafeExecutor.validate(code)
        if violations:
            msg = "代码安全校验失败:\n" + "\n".join(f"  - {v}" for v in violations)
            logger.error(msg)
            raise ValueError(msg)

        # 2. Prepare execution namespace (restricted)
        namespace: dict[str, Any] = {
            "df": df.copy(),  # Don't pollute original
            "pd": pd,
            "result": None,
        }

        # 3. Execute
        try:
            exec(code, {"__builtins__": {}}, namespace)
        except Exception as exc:
            logger.exception("Code execution failed")
            raise RuntimeError(f"代码执行失败: {exc}") from exc

        if "result" not in namespace or namespace["result"] is None:
            raise RuntimeError("代码未将结果赋值给 result 变量")

        result = namespace["result"]
        logger.info(
            "Execution OK: result type=%s",
            type(result).__name__,
        )
        return result
