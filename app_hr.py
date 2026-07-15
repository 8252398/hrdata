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
    "agent_gen": None,       # active generator from SQLAgent.run_iter()
    "agent_event": None,     # last event yielded by the agent
    "awaiting_answer": False,  # UI is showing an AI question
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


def _drive_agent_step(answer: str | None = None) -> dict | None:
    """Advance the SQLAgent generator by one step.

    If answer is provided, sends it to the generator (resumes from ASK).
    Otherwise, calls next() to start or continue.
    Stores the yielded event in st.session_state.agent_event.
    """
    gen = st.session_state.agent_gen
    if gen is None:
        return None
    try:
        if answer is not None:
            event = gen.send(answer)
        else:
            event = next(gen)
    except StopIteration:
        st.session_state.agent_gen = None
        st.session_state.awaiting_answer = False
        return None
    st.session_state.agent_event = event
    if event["type"] == "ask":
        st.session_state.awaiting_answer = True
    elif event["type"] == "result":
        st.session_state.agent_gen = None
        st.session_state.awaiting_answer = False
    else:
        st.session_state.awaiting_answer = False
    return event


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

        # ER diagram
        with st.expander("🗂️ 数据库 ER 图", expanded=False):
            try:
                from modules.er_diagram import build_er_figure
                fig = build_er_figure()
                st.pyplot(fig)
            except Exception as exc:
                st.warning(f"ER 图生成失败: {exc}")

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
# Tab 2: AI Analysis (SQL Agent)
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
                if msg["role"] == "assistant" and "history" in msg:
                    with st.expander(f"📋 Agent 探索过程（{len(msg['history'])} 轮）", expanded=False):
                        for h in msg["history"]:
                            st.caption(f"第{h['turn']}轮: {h['rows']}行")
                            st.code(h["sql"], language="sql")
                st.markdown(msg["content"])

        # ── Human-in-the-loop agent UI ──
        # If we are waiting for an answer to an AI question, show the form
        if st.session_state.awaiting_answer:
            event = st.session_state.agent_event
            with st.chat_message("assistant"):
                st.markdown(f"**❓ AI 提问：** {event['text']}")
                with st.form(key="agent_ask_form", clear_on_submit=True):
                    human_answer = st.text_input(
                        "请输入你的回答：", key="human_answer_input"
                    )
                    submitted = st.form_submit_button("💬 提交回答，继续分析")
                if submitted and human_answer:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❓ {event['text']}",
                    })
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": human_answer,
                    })
                    # Resume generator with the human answer
                    _drive_agent_step(answer=human_answer)
                    st.rerun()

        # Normal chat input (only active when NOT awaiting answer)
        if not st.session_state.awaiting_answer:
            if question := st.chat_input(
                "输入你的分析问题，例如：统计各部门的培训总学时"
            ):
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )

                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    status = st.status("AI Agent 分析中...", expanded=True)

                    try:
                        # 1. Build LLM client
                        client = LLMClient(
                            api_key=st.session_state.llm_key,
                            provider=st.session_state.llm_provider,
                            model=st.session_state.llm_model,
                            base_url=st.session_state.llm_base,
                        )

                        # 2. Initialise generator-driven agent
                        from modules.sql_agent import SQLAgent
                        agent = SQLAgent()
                        db = TrainingDatabase()

                        st.session_state.agent_gen = agent.run_iter(
                            question=question,
                            llm_client=client,
                            db=db,
                            status_writer=status,
                        )

                        # 3. Drive generator until ask or result
                        while True:
                            event = _drive_agent_step()
                            if event is None:
                                break
                            if event["type"] == "status":
                                status.write(event["msg"])
                            elif event["type"] == "sql":
                                status.write(
                                    f"⚡ 执行 SQL ({'探索' if event['exploratory'] else '最终'}): "
                                    f"{event['rows']} 行"
                                )
                            elif event["type"] == "ask":
                                # Generator paused — store question and
                                # trigger a rerun so the ask form appears
                                status.update(
                                    label="❓ AI 需要更多信息",
                                    state="running",
                                )
                                st.rerun()
                            elif event["type"] == "result":
                                break

                        # 4. Handle result (or fall-through if ask)
                        result_event = st.session_state.agent_event
                        if result_event and result_event["type"] == "result":
                            final_sql = result_event["sql"]
                            final_df = result_event["result"]
                            explanation = result_event["explanation"]
                            turns = result_event["turns"]
                            history = result_event["history"]

                            status.update(
                                label=f"✅ Agent 完成（{turns} 轮探索）",
                                state="complete",
                            )

                            st.caption(f"共探索 {turns} 轮，生成最终 SQL:")
                            st.code(final_sql, language="sql")
                            st.markdown(explanation.strip())

                            if (
                                isinstance(final_df, pd.DataFrame)
                                and not final_df.empty
                            ):
                                st.dataframe(
                                    final_df,
                                    use_container_width=True,
                                    height=min(500, 35 * len(final_df) + 38),
                                )
                                buf = io.BytesIO()
                                final_df.to_excel(
                                    buf, index=False, engine="openpyxl"
                                )
                                st.download_button(
                                    "📥 下载结果 Excel",
                                    buf.getvalue(),
                                    "analysis_result.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"dl_{len(st.session_state.chat_history)}",
                                )
                            elif isinstance(final_df, pd.DataFrame):
                                st.info("查询结果为空")
                            else:
                                st.metric("结果", str(final_df))

                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": explanation.strip(),
                                "code": final_sql,
                                "history": history,
                            })
                        else:
                            # Ask form will render on next rerun
                            pass

                        db.close()

                    except Exception as exc:
                        status.update(label="❌ Agent 分析失败", state="error")
                        st.error(f"分析失败: {exc}")
                        logger.exception("Agent analysis failed")
                        _add_error_to_history(str(exc))


# ═══════════════════════════════════════════════════════════
# Self-check
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("app_hr v2.0 - AI Excel Data Analysis Platform")
    print("Architecture: Streamlit(UI) → modules/ → LLM(SQL gen) → SQLite → Results")
    print("Modules: excel_loader, data_profile, llm_client, prompt_builder, sqlite_manager")
    print("OK")
