# -*- coding: utf-8 -*-
"""app_hr — AI Excel 数据分析平台 (v2.0)

架构: Streamlit(UI) → modules/ → LLM(SQL gen) → SQLite → Results
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st

from config.settings import LLM_PROVIDERS, DB_PATH
from modules.excel_loader import ExcelLoader
from modules.data_profile import build_profile
from modules.llm_client import LLMClient
from modules.prompt_builder import PromptBuilder
from modules.sqlite_manager import TrainingDatabase
from utils.logger import get_logger

logger = get_logger(__name__)

# ---- Page config ----
st.set_page_config(
    page_title="AI Excel 数据分析平台",
    page_icon="📊",
    layout="wide",
)

# ---- Session state init ----
DEFAULTS = {
    "db_built": False,
    "db_stats": None,
    "db_reports": None,
    "profile": None,
    "chat_history": [],
    "llm_configured": False,
    "db_schema_text": "",
}
for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ═══════════════════════════════════════════════════════════
# Helpers (defined before use in UI)
# ═══════════════════════════════════════════════════════════

def _extract_sql(text: str) -> str:
    """Extract SQL from LLM response (handles markdown code blocks)."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Remove common non-SQL prefixes
    lines = text.split("\n")
    clean = []
    for line in lines:
        s = line.strip()
        if s and not s.startswith("#") and not s.startswith("--"):
            clean.append(s)
        elif s.startswith("--"):
            clean.append(s)
    return "\n".join(clean).strip().rstrip(";")


def _add_error_to_history(error_msg: str, code: str = "") -> None:
    """Add an error message to chat history."""
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": f"❌ {error_msg}",
        "code": code,
    })


def _build_db_schema_text(db_stats: dict) -> str:
    """Build a compact schema description for the LLM."""
    return f"""Tables:
- persons: employee_code TEXT PRIMARY KEY, name TEXT, phone TEXT, unit TEXT, department TEXT, cadre_flag TEXT
- training_records: id INTEGER PK, employee_code TEXT FK->persons, course_name TEXT, hours REAL, study_type TEXT, training_type TEXT, training_method TEXT, organizer TEXT, institution TEXT, start_date TEXT, end_date TEXT

Current stats: {db_stats}"""


# ═══════════════════════════════════════════════════════════
# Sidebar — LLM Configuration
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.header("⚙️ LLM 配置")

    provider = st.selectbox(
        "模型提供商",
        options=list(LLM_PROVIDERS.keys()),
        format_func=lambda x: {
            "deepseek": "DeepSeek",
            "openai": "OpenAI",
            "glm": "GLM (Cloudflare)",
            "ollama": "Ollama (本地)",
            "custom": "自定义",
        }.get(x, x),
        index=0,
    )

    if provider == "custom":
        custom_base = st.text_input("Base URL", placeholder="https://api.example.com/v1")
        custom_model = st.text_input("Model Name", placeholder="model-name")
        custom_key = st.text_input("API Key", type="password", placeholder="sk-...")
    else:
        cfg = LLM_PROVIDERS[provider]
        custom_base = cfg.base_url
        custom_model = cfg.model

        if provider == "ollama":
            custom_key = "ollama"
        else:
            custom_key = st.text_input(
                "API Key",
                type="password",
                placeholder="sk-..." if provider != "glm" else "CF Token...",
            )

    if st.button("✅ 确认配置", use_container_width=True):
        if not custom_key:
            st.error("请输入 API Key")
        else:
            st.session_state.llm_configured = True
            st.session_state.llm_provider = provider
            st.session_state.llm_base = custom_base
            st.session_state.llm_model = custom_model
            st.session_state.llm_key = custom_key
            st.success(f"已配置: {provider} / {custom_model}")

# ═══════════════════════════════════════════════════════════
# Main Page
# ═══════════════════════════════════════════════════════════

st.title("📊 AI Excel 数据分析平台")

tab1, tab2 = st.tabs(["📁 数据导入", "💬 AI 分析"])

# ═══════════════════════════════════════════════════════════
# Tab 1: Data Import & Database Build
# ═══════════════════════════════════════════════════════════

with tab1:
    st.markdown("### 上传 Excel 并构建数据库")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**📋 培训学时记录**")
        training_file = st.file_uploader(
            "上传培训学时记录", type=["xlsx", "csv"], key="training_file",
        )
    with col2:
        st.markdown("**👤 干部人员信息**")
        person_file = st.file_uploader(
            "上传干部人员信息", type=["xlsx", "csv"], key="person_file",
        )

    if not training_file:
        st.info("👆 请上传培训学时记录")
    else:
        if st.button("🚀 构建数据库", type="primary", use_container_width=True):
            with st.spinner("正在构建数据库..."):
                try:
                    df_training = ExcelLoader.load(training_file)

                    db = TrainingDatabase()
                    db.reset()
                    report_phase1 = db.build_from_training(df_training)
                    st.session_state.db_reports = {"phase1": report_phase1.to_dict()}

                    if person_file:
                        df_person = ExcelLoader.load(person_file)
                        report_phase2 = db.supplement_cadre_info(df_person)
                        st.session_state.db_reports["phase2"] = report_phase2.to_dict()

                    stats = db.get_stats()
                    st.session_state.db_stats = stats
                    st.session_state.db_built = True
                    st.session_state.db_schema_text = _build_db_schema_text(stats)

                    # Build a lightweight profile (not full DataFrame)
                    st.session_state.profile = build_profile(df_training)

                    db.close()

                    st.success("✅ 数据库构建完成！")
                    st.balloons()

                except Exception as exc:
                    st.error(f"构建失败: {exc}")
                    logger.exception("Database build failed")

    if st.session_state.db_built:
        st.divider()
        st.markdown("### 📊 数据库构建报告")

        stats = st.session_state.db_stats
        if stats:
            cols = st.columns(len(stats))
            for i, (label, val) in enumerate(stats.items()):
                cols[i].metric(label, val)

        if st.session_state.db_reports:
            reports = st.session_state.db_reports
            for phase, report in reports.items():
                with st.expander(f"📋 {phase} 详情", expanded=False):
                    errors = report.pop("错误", [])
                    st.json(report)
                    if errors:
                        st.warning("⚠️ 错误/警告")
                        for err in errors:
                            st.caption(f"- {err}")

# ═══════════════════════════════════════════════════════════
# Tab 2: AI Analysis (SQL-based)
# ═══════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 💬 自然语言数据分析")

    if not st.session_state.db_built:
        st.info("👈 请先在「数据导入」标签页上传 Excel 并构建数据库")
    elif not st.session_state.llm_configured:
        st.info("👈 请在左侧边栏配置 LLM API")
    else:
        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                if msg["role"] == "assistant" and "code" in msg:
                    with st.expander("🔧 查看生成的 SQL", expanded=False):
                        st.code(msg["code"], language="sql")
                st.markdown(msg["content"])

        if question := st.chat_input("输入你的分析问题，例如：统计各部门的培训总学时"):
            st.session_state.chat_history.append({"role": "user", "content": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                status = st.status("AI 分析中...", expanded=True)

                try:
                    # 1. Build LLM client
                    client = LLMClient(
                        api_key=st.session_state.llm_key,
                        provider=st.session_state.llm_provider,
                        model=st.session_state.llm_model,
                        base_url=st.session_state.llm_base,
                    )

                    # 2. Build SQL generation prompt
                    pb = PromptBuilder()
                    system_prompt, user_prompt = pb.build_sql_prompt(
                        schema_text=st.session_state.db_schema_text,
                        question=question,
                    )
                    status.write("📤 已发送分析请求...")

                    # 3. Get SQL from LLM
                    raw_response = client.chat(
                        user_message=user_prompt,
                        system_message=system_prompt,
                    )
                    status.write("📥 已收到 AI 响应...")

                    # 4. Extract SQL
                    sql = _extract_sql(raw_response)
                    status.write(f"🔍 SQL 提取完成: {len(sql)} 字符")

                    # 5. Execute SQL against SQLite
                    db = TrainingDatabase()
                    result = db.query_to_df(sql)
                    db.close()
                    status.write(f"⚡ SQL 执行完成: {len(result)} 行 × {len(result.columns)} 列")

                    # 6. Build explanation
                    if isinstance(result, pd.DataFrame) and not result.empty:
                        summary_text = (
                            f"结果: {len(result)} 行 × {len(result.columns)} 列, "
                            f"列: {', '.join(result.columns[:8])}"
                        )
                    elif isinstance(result, pd.DataFrame):
                        summary_text = "查询结果为空"
                    else:
                        summary_text = str(result)[:500]

                    expl_system, expl_prompt = pb.build_explanation_prompt(
                        question=question,
                        summary=summary_text,
                    )
                    explanation = client.chat(
                        user_message=expl_prompt,
                        system_message=expl_system,
                    )

                    status.update(label="✅ 分析完成", state="complete")

                    # 7. Display
                    st.markdown(explanation.strip())

                    if isinstance(result, pd.DataFrame) and not result.empty:
                        st.dataframe(
                            result,
                            use_container_width=True,
                            height=min(500, 35 * len(result) + 38),
                        )
                        buf = io.BytesIO()
                        result.to_excel(buf, index=False, engine="openpyxl")
                        st.download_button(
                            "📥 下载结果 Excel",
                            buf.getvalue(),
                            "analysis_result.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_{len(st.session_state.chat_history)}",
                        )
                    elif isinstance(result, pd.DataFrame):
                        st.info("查询结果为空")
                    else:
                        st.metric("结果", str(result))

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": explanation.strip(),
                        "code": sql,
                    })

                except Exception as exc:
                    status.update(label="❌ 分析失败", state="error")
                    st.error(f"分析失败: {exc}")
                    logger.exception("Analysis pipeline failed")
                    _add_error_to_history(str(exc), sql if "sql" in dir() else "")


# ═══════════════════════════════════════════════════════════
# Self-check
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("app_hr v2.0 - AI Excel Data Analysis Platform")
    print("Architecture: Streamlit(UI) → modules/ → LLM(SQL gen) → SQLite → Results")
    print("Modules: excel_loader, data_profile, llm_client, prompt_builder, sqlite_manager")
    print("OK")
