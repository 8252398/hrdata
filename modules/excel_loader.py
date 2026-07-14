# -*- coding: utf-8 -*-
"""Excel file loader — read .xlsx/.csv and return DataFrames."""

from __future__ import annotations

from typing import Optional

import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)


class ExcelLoader:
    """Load Excel/CSV files with multi-sheet auto-detection."""

    @staticmethod
    def load(
        file,
        sheet_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a file into a DataFrame.

        Args:
            file: File object (Streamlit upload) or path string.
            sheet_name: Specific sheet name. Auto-detects largest if None.

        Returns:
            DataFrame with the loaded data.

        Raises:
            ValueError: If the file cannot be read.
        """
        logger.info("Loading file: %s", getattr(file, "name", str(file)))

        try:
            if hasattr(file, "name") and file.name.endswith(".csv"):
                df = pd.read_csv(file)
                logger.info("CSV loaded: %d rows x %d cols", len(df), len(df.columns))
                return df

            if sheet_name:
                df = pd.read_excel(file, sheet_name=sheet_name)
            else:
                xl = pd.ExcelFile(file)
                if len(xl.sheet_names) == 1:
                    df = pd.read_excel(file)
                else:
                    largest = max(
                        xl.sheet_names,
                        key=lambda s: xl.parse(s).shape[0],
                    )
                    df = pd.read_excel(file, sheet_name=largest)
                    logger.info(
                        "Auto-selected sheet '%s' (%d sheets total)",
                        largest, len(xl.sheet_names),
                    )

            logger.info("Excel loaded: %d rows x %d cols", len(df), len(df.columns))
            return df

        except Exception as exc:
            logger.exception("Failed to load file")
            raise ValueError(f"文件读取失败: {exc}") from exc
