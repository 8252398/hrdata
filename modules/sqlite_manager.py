# -*- coding: utf-8 -*-
"""SQLite manager — build personal training-hours database from Excel.

Implements USERNEEDS.md Phase 1 & 2:
  1. Import training records from excel, build per-person database.
  2. Supplement with cadre info from personnel excel.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import DB_PATH
from utils.logger import get_logger

logger = get_logger(__name__)

# Fields to extract from training records
TRAINING_FIELDS = [
    "学时", "学习类型", "培训类型", "培训方式",
    "主办单位", "培训机构", "开始学习时间", "完成学习时间",
]

# Fields for personal info
PERSON_FIELDS = ["姓名", "商网手机号", "单位名称", "部门名称"]

# Field alias mapping: standard name -> possible column names
FIELD_ALIASES = {
    "集团员工编码": ["集团员工编码", "员工编码", "工号"],
    "姓名": ["姓名", "人员姓名"],
    "商网手机号": ["商网手机号", "手机号", "联系电话"],
    "单位名称": ["单位名称", "单位"],
    "部门名称": ["部门名称", "部门"],
    "干部标识": ["干部标识", "干部级别"],
    "学时": ["学时", "培训学时"],
    "来源信息": ["来源信息", "班次名称", "课程名称"],
    "学习类型": ["学习类型"],
    "培训类型": ["培训类型"],
    "培训方式": ["培训方式"],
    "主办单位": ["主办单位"],
    "培训机构": ["培训机构"],
    "开始学习时间": ["开始学习时间", "开始时间"],
    "完成学习时间": ["完成学习时间", "结束时间"],
}


def _resolve_column(df: pd.DataFrame, aliases: list[str]) -> Optional[str]:
    """Find a column in df matching any of the aliases (case-insensitive)."""
    df_cols_lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        key = alias.lower().strip()
        if key in df_cols_lower:
            return df_cols_lower[key]
    return None


def _normalize_date_series(series: pd.Series) -> pd.Series:
    """Normalize a date-like series to ISO 'YYYY-MM-DD' strings.

    Returns the original string representation for values that cannot be
    parsed, so callers can report invalid dates separately.
    """
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    normalized = parsed.dt.strftime("%Y-%m-%d")
    # Preserve original values where parsing failed
    normalized[parsed.isna() & series.notna()] = series[parsed.isna() & series.notna()].astype(str)
    return normalized


@dataclass
class ImportReport:
    """Report after database build/supplement."""

    total_training_rows: int = 0
    total_persons_created: int = 0
    duplicate_codes: int = 0
    missing_cadre_count: int = 0
    missing_training_count: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "培训记录总行数": self.total_training_rows,
            "创建人员记录数": self.total_persons_created,
            "重复编码数": self.duplicate_codes,
            "缺少干部标识人数": self.missing_cadre_count,
            "缺少学时记录人数": self.missing_training_count,
            "错误": self.errors,
        }


class TrainingDatabase:
    """SQLite database for per-person training records."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self.conn is None:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.row_factory = sqlite3.Row
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA foreign_keys=ON")
        return self.conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def reset(self) -> None:
        """Drop and recreate all tables."""
        conn = self.connect()
        conn.execute("DROP TABLE IF EXISTS training_records")
        conn.execute("DROP TABLE IF EXISTS persons")
        conn.commit()

    def init_schema(self) -> None:
        """Create tables if not exist."""
        conn = self.connect()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS persons (
                employee_code TEXT PRIMARY KEY,
                name TEXT,
                phone TEXT,
                unit TEXT,
                department TEXT,
                cadre_flag TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS training_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_code TEXT NOT NULL,
                course_name TEXT,
                hours REAL,
                study_type TEXT,
                training_type TEXT,
                training_method TEXT,
                organizer TEXT,
                institution TEXT,
                start_date TEXT,
                end_date TEXT,
                FOREIGN KEY (employee_code) REFERENCES persons(employee_code)
            );

            CREATE INDEX IF NOT EXISTS idx_training_code
                ON training_records(employee_code);
        """)
        conn.commit()
        logger.info("Schema initialized")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: Import training records
    # ═══════════════════════════════════════════════════════════

    def build_from_training(self, training_df: pd.DataFrame) -> ImportReport:
        """Import training records from excel.

        For each employee_code:
        - Create a person record with name/phone/unit/dept.
        - Create training records for each unique course (by 来源信息).
        """
        report = ImportReport(total_training_rows=len(training_df))
        conn = self.connect()
        self.init_schema()

        # Resolve columns
        col_code = _resolve_column(training_df, FIELD_ALIASES["集团员工编码"])
        if not col_code:
            report.errors.append("未找到集团员工编码列")
            return report

        col_course = _resolve_column(training_df, FIELD_ALIASES["来源信息"])
        col_name = _resolve_column(training_df, FIELD_ALIASES["姓名"])
        col_phone = _resolve_column(training_df, FIELD_ALIASES["商网手机号"])
        col_unit = _resolve_column(training_df, FIELD_ALIASES["单位名称"])
        col_dept = _resolve_column(training_df, FIELD_ALIASES["部门名称"])

        # Resolve training detail columns
        field_cols = {}
        for fname in TRAINING_FIELDS:
            resolved = _resolve_column(training_df, FIELD_ALIASES.get(fname, [fname]))
            if resolved:
                field_cols[fname] = resolved

        # Clean: strip whitespace, drop empty codes
        df = training_df.copy()
        df[col_code] = df[col_code].astype(str).str.strip()
        df = df[df[col_code] != ""].dropna(subset=[col_code])

        # Detect duplicate employee codes in the personnel dimension.
        # Multiple training rows per person are expected, but the same code
        # should map to a single person profile.
        code_counts = df[col_code].value_counts()
        duplicated_codes = code_counts[code_counts > 1].index.tolist()
        report.duplicate_codes = len(duplicated_codes)
        if duplicated_codes:
            sample = duplicated_codes[:10]
            report.errors.append(
                f"培训学时记录中存在 {len(duplicated_codes)} 个重复集团员工编码（按人员维度）"
                f"，示例：{', '.join(sample)}"
            )

        # Normalize date columns to YYYY-MM-DD and report invalid dates
        date_cols = {
            "开始学习时间": field_cols.get("开始学习时间"),
            "完成学习时间": field_cols.get("完成学习时间"),
        }
        for label, col in date_cols.items():
            if col:
                original = df[col].copy()
                df[col] = _normalize_date_series(df[col])
                invalid_mask = (
                    original.notna()
                    & (original.astype(str).str.strip() != "")
                    & df[col].str.match(r"\d{4}-\d{2}-\d{2}", na=False).eq(False)
                )
                if invalid_mask.any():
                    report.errors.append(
                        f"{label} 列存在 {int(invalid_mask.sum())} 行无法解析的日期"
                    )

        # Build person records
        persons_inserted = set()
        person_rows = []
        for _, row in df.iterrows():
            code = row[col_code]
            if code in persons_inserted:
                continue
            persons_inserted.add(code)
            person_rows.append((
                code,
                str(row.get(col_name, "")) if col_name and pd.notna(row.get(col_name, "")) else "",
                str(row.get(col_phone, "")) if col_phone and pd.notna(row.get(col_phone, "")) else "",
                str(row.get(col_unit, "")) if col_unit and pd.notna(row.get(col_unit, "")) else "",
                str(row.get(col_dept, "")) if col_dept and pd.notna(row.get(col_dept, "")) else "",
            ))

        conn.executemany(
            "INSERT OR REPLACE INTO persons (employee_code, name, phone, unit, department) "
            "VALUES (?, ?, ?, ?, ?)",
            person_rows,
        )
        report.total_persons_created = len(person_rows)

        # Build training records
        course_col_name = "来源信息"
        train_rows = []
        for _, row in df.iterrows():
            code = row[col_code]
            course = str(row.get(col_course, "")) if col_course and pd.notna(row.get(col_course, "")) else ""
            train_rows.append((
                code,
                course,
                float(row.get(field_cols.get("学时", ""), 0)) if "学时" in field_cols and pd.notna(row.get(field_cols["学时"], None)) else 0,
                str(row.get(field_cols.get("学习类型", ""), "")) if "学习类型" in field_cols else "",
                str(row.get(field_cols.get("培训类型", ""), "")) if "培训类型" in field_cols else "",
                str(row.get(field_cols.get("培训方式", ""), "")) if "培训方式" in field_cols else "",
                str(row.get(field_cols.get("主办单位", ""), "")) if "主办单位" in field_cols else "",
                str(row.get(field_cols.get("培训机构", ""), "")) if "培训机构" in field_cols else "",
                str(row.get(field_cols.get("开始学习时间", ""), "")) if "开始学习时间" in field_cols else "",
                str(row.get(field_cols.get("完成学习时间", ""), "")) if "完成学习时间" in field_cols else "",
            ))

        conn.executemany(
            "INSERT INTO training_records "
            "(employee_code, course_name, hours, study_type, training_type, "
            "training_method, organizer, institution, start_date, end_date) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            train_rows,
        )

        conn.commit()
        logger.info(
            "Build complete: %d persons, %d training records, %d duplicates",
            report.total_persons_created,
            len(train_rows),
            report.duplicate_codes,
        )
        return report

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Supplement cadre info
    # ═══════════════════════════════════════════════════════════

    def supplement_cadre_info(self, person_df: pd.DataFrame) -> ImportReport:
        """Supplement cadre_flag from personnel excel.

        Match by employee_code, update persons table.
        Report unmatched codes (both directions).
        """
        report = ImportReport()
        conn = self.connect()

        col_code = _resolve_column(person_df, FIELD_ALIASES["集团员工编码"])
        if not col_code:
            report.errors.append("干部信息表中未找到集团员工编码列")
            return report

        col_cadre = _resolve_column(person_df, FIELD_ALIASES["干部标识"])

        df = person_df.copy()
        df[col_code] = df[col_code].astype(str).str.strip()
        df = df[df[col_code] != ""].dropna(subset=[col_code])

        # Detect duplicate employee codes in cadre info table
        code_counts = df[col_code].value_counts()
        duplicated_codes = code_counts[code_counts > 1].index.tolist()
        if duplicated_codes:
            sample = duplicated_codes[:10]
            report.duplicate_codes = len(duplicated_codes)
            report.errors.append(
                f"干部人员信息表中存在 {len(duplicated_codes)} 个重复集团员工编码"
                f"，将使用第一条记录补充干部标识，示例：{', '.join(sample)}"
            )
            # Deduplicate: keep the first occurrence for cadre_flag supplementation
            df = df.drop_duplicates(subset=[col_code], keep="first")

        # Get existing codes from DB
        existing = set(
            row[0] for row in conn.execute("SELECT employee_code FROM persons").fetchall()
        )

        matched = 0
        excel_codes = set()

        for _, row in df.iterrows():
            code = row[col_code]
            excel_codes.add(code)

            if code not in existing:
                report.missing_training_count += 1
                report.errors.append(f"无法找到人员 {code} 的学时记录")
                continue

            cadre_val = str(row.get(col_cadre, "")) if col_cadre and pd.notna(row.get(col_cadre, "")) else ""
            conn.execute(
                "UPDATE persons SET cadre_flag = ? WHERE employee_code = ?",
                (cadre_val, code),
            )
            matched += 1

        # Find codes in DB but not in excel
        missing_in_excel = existing - excel_codes
        report.missing_cadre_count = len(missing_in_excel)
        if missing_in_excel:
            report.errors.append(
                f"{len(missing_in_excel)} 名人员在数据库中缺少干部标识"
            )

        conn.commit()
        logger.info(
            "Cadre supplement: %d matched, %d missing training, %d missing cadre",
            matched, report.missing_training_count, report.missing_cadre_count,
        )
        return report

    # ═══════════════════════════════════════════════════════════
    # Query helpers
    # ═══════════════════════════════════════════════════════════

    def query_to_df(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        """Run a SQL query and return results as DataFrame."""
        conn = self.connect()
        try:
            return pd.read_sql_query(sql, conn, params=params)
        except Exception as exc:
            logger.exception("SQL query failed: %s", sql)
            raise

    def get_stats(self) -> dict:
        """Get database summary statistics."""
        conn = self.connect()
        persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        records = conn.execute("SELECT COUNT(*) FROM training_records").fetchone()[0]
        with_cadre = conn.execute(
            "SELECT COUNT(*) FROM persons WHERE cadre_flag IS NOT NULL AND cadre_flag != ''"
        ).fetchone()[0]
        return {
            "人员数": persons,
            "培训记录数": records,
            "有干部标识人数": with_cadre,
        }
