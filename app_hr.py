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

from config.settings import LLM_PROVIDERS, DB_PATH, MAX_AGENT_TURNS_DEFAULT, MAX_AGENT_TURNS_EXTEND
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
    # Agent state
    "agent_gen": None,       # active generator from SQLAgent.run_iter()
    "agent_event": None,     # last event yielded by the agent
    "agent_session": None,   # AgentSessionState (serializable)
    # Pause flags
    "awaiting_answer": False,   # UI is showing an AI question (ASK)
    "awaiting_review": False,   # UI is showing review card
    "pending_review": None,
    "awaiting_exhausted": False,  # UI is showing "add 8 more" popup
    "pending_exhausted": None,
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
        st.session_state.awaiting_review = False
        st.session_state.awaiting_exhausted = False
        return None
    st.session_state.agent_event = event
    # Reset all pause flags, then set the one matching current event
    st.session_state.awaiting_answer = False
    st.session_state.awaiting_review = False
    st.session_state.awaiting_exhausted = False
    if event["type"] == "ask":
        st.session_state.awaiting_answer = True
    elif event["type"] == "review":
        st.session_state.awaiting_review = True
        st.session_state.pending_review = event
    elif event["type"] == "exhausted":
        st.session_state.awaiting_exhausted = True
        st.session_state.pending_exhausted = event
    elif event["type"] == "result":
        st.session_state.agent_gen = None
    return event


def _send_review(feedback_dict: dict) -> None:
    """Resume generator with human review feedback."""
    ev = st.session_state.pending_review
    st.session_state.chat_history.append({
        "role": "user",
        "content": f"【第{ev['turn']}轮审核】{feedback_dict['action']}: {feedback_dict.get('feedback','')}",
    })
    event = _drive_agent_step(answer=feedback_dict)
    while event and event["type"] not in ("review", "ask", "exhausted", "result"):
        event = _drive_agent_step()
    st.rerun()


def _send_ask_answer(answer: str) -> None:
    """Resume generator with ASK answer."""
    ev = st.session_state.agent_event
    st.session_state.chat_history.append({
        "role": "assistant", "content": f"❓ {ev['text']}",
    })
    st.session_state.chat_history.append({
        "role": "user", "content": answer,
    })
    event = _drive_agent_step(answer=answer)
    while event and event["type"] not in ("review", "ask", "exhausted", "result"):
        event = _drive_agent_step()
    st.rerun()


def _build_db_schema_text(db_stats: dict) -> str:
    """Build a compact schema description for the LLM."""
    return f"""Tables:
- persons: employee_code TEXT PRIMARY KEY, name TEXT, phone TEXT, unit TEXT, department TEXT, cadre_flag TEXT
- training_records: id INTEGER PK, employee_code TEXT FK->persons, course_name TEXT, hours REAL, study_type TEXT, training_type TEXT, training_method TEXT, organizer TEXT, institution TEXT, start_date TEXT, end_date TEXT

Current stats: {db_stats}"""


# ═══════════════════════════════════════════════════════════
# Sidebar — Session List + LLM Configuration
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.divider()
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
                if msg["role"] == "assistant":
                    if "reasoning" in msg:
                        st.markdown("🧠 **AI 推理**")
                        st.markdown(msg["reasoning"])
                    if "code" in msg:
                        with st.expander("🔧 查看生成的 SQL", expanded=False):
                            st.code(msg["code"], language="sql")
                    if "history" in msg:
                        with st.expander(f"📋 Agent 探索过程（{len(msg['history'])} 轮）", expanded=False):
                            for h in msg["history"]:
                                st.caption(f"第{h['turn']}轮: {h['rows']}行")
                                st.code(h["sql"], language="sql")
                st.markdown(msg["content"])

        # ── Human-in-the-loop agent UI ──
        # 1. ASK form
        if st.session_state.awaiting_answer:
            event = st.session_state.agent_event
            with st.warning("⚠️ AI 需要你的帮助"):
                st.markdown(f"❓ **AI 提问：** {event['text']}")
                with st.form(key="agent_ask_form", clear_on_submit=True):
                    human_answer = st.text_input(
                        "请输入你的回答：", key="human_answer_input"
                    )
                    submitted = st.form_submit_button("💬 提交回答，继续分析")
                if submitted and human_answer:
                    _send_ask_answer(human_answer)

        # 2. REVIEW card
        if st.session_state.awaiting_review:
            ev = st.session_state.pending_review
            with st.chat_message("assistant"):
                st.caption(
                    f"第 {ev['turn']} 轮 · {'最终' if ev['is_final'] else '探索'} · "
                    f"剩余 {ev['remaining_turns']} 轮"
                )
                # 推理在前，默认展开
                if ev.get("reasoning"):
                    st.markdown("🧠 **AI 推理**")
                    st.markdown(ev["reasoning"])
                # SQL 在后
                st.code(ev["sql"], language="sql")
                st.markdown(f"**结果：** {ev['rows']} 行 × {ev['cols']} 列")
                st.dataframe(ev["result"].head(10), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                approved = c1.button("👍 批准并继续", key="review_approve")
                reject_clicked = c2.button("👎 拒绝并重写", key="review_reject")
                append_clicked = c3.button("💬 追加指令", key="review_append")

                if approved:
                    _send_review({"action": "approve", "feedback": "", "turn": ev["turn"]})

                if reject_clicked:
                    st.session_state["review_reject_open"] = True
                if append_clicked:
                    st.session_state["review_append_open"] = True

                if st.session_state.get("review_reject_open"):
                    fb = st.text_input("拒绝原因 / 修改要求", key="reject_fb")
                    if st.button("提交拒绝", key="submit_reject"):
                        _send_review({"action": "reject", "feedback": fb, "turn": ev["turn"]})

                if st.session_state.get("review_append_open"):
                    fb = st.text_input("追加指令", key="append_fb")
                    if st.button("提交追加", key="submit_append"):
                        _send_review({"action": "append", "feedback": fb, "turn": ev["turn"]})

        # 3. EXHAUSTION popup (NEW — Decision 18)
        if st.session_state.awaiting_exhausted:
            ev = st.session_state.pending_exhausted
            st.warning(
                f"已达最大轮次（第 {ev['turn']} 轮）。是否增加 {MAX_AGENT_TURNS_EXTEND} 轮？"
            )
            c1, c2 = st.columns(2)
            if c1.button("➕ 增加轮次", key="exhaust_extend"):
                _send_review({"action": "extend", "feedback": "", "turn": ev["turn"]})
            if c2.button("⏹️ 结束分析", key="exhaust_give_up"):
                _send_review({"action": "give_up", "feedback": "", "turn": ev["turn"]})

        # 4. RESULT rendering (when result arrives)
        result_event = st.session_state.agent_event
        if result_event and result_event["type"] == "result":
            final_sql = result_event["sql"]
            final_reasoning = result_event.get("reasoning", "")
            final_df = result_event["result"]
            explanation = result_event["explanation"]
            turns = result_event["turns"]
            history = result_event["history"]

            st.caption(f"共探索 {turns} 轮")
            if final_reasoning:
                st.markdown("🧠 **最终推理**")
                st.markdown(final_reasoning)
            st.markdown("🔧 **最终 SQL**")
            st.code(final_sql, language="sql")
            st.markdown("📊 **结果分析**")
            st.markdown(explanation.strip())

            if isinstance(final_df, pd.DataFrame) and not final_df.empty:
                st.dataframe(
                    final_df,
                    use_container_width=True,
                    height=min(500, 35 * len(final_df) + 38),
                )
                buf = io.BytesIO()
                final_df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "📥 下载结果 Excel",
                    buf.getvalue(),
                    "analysis_result.xlsx",
                    mime="application/vnd.openxmlformats.officedocument.spreadsheetml.sheet",
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
                "reasoning": final_reasoning,
                "history": history,
            })
            st.session_state.agent_event = None

        # 5. Normal chat input (only when NOT awaiting anything)
        not_awaiting = not (
            st.session_state.awaiting_answer
            or st.session_state.awaiting_review
            or st.session_state.awaiting_exhausted
        )
        if not_awaiting:
            if question := st.chat_input(
                "输入你的分析问题，例如：统计各部门的培训总学时"
            ):
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                with st.chat_message("user"):
                    st.markdown(question)

                try:
                    # Build LLM client
                    client = LLMClient(
                        api_key=st.session_state.llm_key,
                        provider=st.session_state.llm_provider,
                        model=st.session_state.llm_model,
                        base_url=st.session_state.llm_base,
                    )

                    # Initialise generator-driven agent
                    from modules.sql_agent import SQLAgent
                    agent = SQLAgent()
                    db = TrainingDatabase()

                    st.session_state.agent_gen = agent.run_iter(
                        question=question,
                        llm_client=client,
                        db=db,
                    )
                    st.session_state.agent_session = agent

                    # Drive generator until first pause (review/ask/exhausted/result)
                    event = _drive_agent_step()
                    while event and event["type"] not in ("review", "ask", "exhausted", "result"):
                        event = _drive_agent_step()

                    if event and event["type"] in ("review", "ask", "exhausted", "result"):
                        st.rerun()

                    db.close()

                except Exception as exc:
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
