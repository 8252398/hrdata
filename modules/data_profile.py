# -*- coding: utf-8 -*-
"""DataFrame profiler — generate schema/statistics for LLM prompts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.settings import MAX_SAMPLE_ROWS
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DataFrameProfile:
    """Compact DataFrame summary for LLM consumption."""

    row_count: int = 0
    column_count: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing: dict[str, int] = field(default_factory=dict)
    sample_rows: list[dict[str, Any]] = field(default_factory=list)
    numeric_summary: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_prompt_text(self) -> str:
        """Serialize profile to a compact prompt fragment."""
        lines = [
            f"- 行数: {self.row_count}",
            f"- 列数: {self.column_count}",
            "",
            "## 字段列表",
        ]
        for col in self.columns:
            dtype = self.dtypes.get(col, "unknown")
            miss = self.missing.get(col, 0)
            lines.append(f"- {col} ({dtype}, 缺失 {miss} 行)")

        if self.numeric_summary:
            lines.append("")
            lines.append("## 数值字段摘要")
            for col, stats in self.numeric_summary.items():
                lines.append(
                    f"- {col}: min={stats.get('min',0):.1f}, "
                    f"max={stats.get('max',0):.1f}, "
                    f"mean={stats.get('mean',0):.1f}"
                )

        lines.append("")
        lines.append(f"## 前 {len(self.sample_rows)} 行样例")
        if self.sample_rows:
            sample_df = pd.DataFrame(self.sample_rows)
            lines.append(sample_df.to_markdown(index=False))

        return "\n".join(lines)


def build_profile(df: pd.DataFrame, max_samples: int = MAX_SAMPLE_ROWS) -> DataFrameProfile:
    """Generate a DataFrameProfile from a DataFrame.

    Args:
        df: Source DataFrame.
        max_samples: Maximum sample rows to include.

    Returns:
        DataFrameProfile instance.
    """
    logger.info("Building profile: %d rows x %d cols", len(df), len(df.columns))

    profile = DataFrameProfile(
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        dtypes={c: str(df[c].dtype) for c in df.columns},
        missing={c: int(df[c].isna().sum()) for c in df.columns},
    )

    # Safe sample extraction
    sample = df.head(max_samples).copy()
    for col in sample.columns:
        if pd.api.types.is_datetime64_any_dtype(sample[col]):
            sample[col] = sample[col].astype(str)
    profile.sample_rows = sample.to_dict(orient="records")

    # Numeric summary
    numeric_cols = df.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        try:
            profile.numeric_summary[col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
            }
        except Exception:
            pass  # Skip if stats fail (e.g., all NaN)

    return profile
