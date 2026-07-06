import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import matplotlib
matplotlib.use("Agg")
from openai import OpenAI
import io
import contextlib

import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "NotoSansCJK-Regular.ttc")
if os.path.exists(FONT_PATH):
    CN_FONT = FontProperties(fname=FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams["font.family"] = CN_FONT.get_name()
else:
    CN_FONT = None
matplotlib.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="HR培训数据分析", layout="wide")

# 初始化分析结果缓存
for key in ["report_df", "total_hours", "recent_40_90", "recent_90_plus", "df_person", "df_hours"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("📊 HR培训数据分析")

# ============ 固定规则 ============
KEY_FIELD = "集团员工编码"
VALID_METHODS = ["党委(党组)理论学习中心组学习", "脱产培训(3天以上)", "集中宣讲/专题讲座"]
METHOD_FIELD = "培训方式"
HOURS_FIELD = "学时"
DATE_FIELD = "完成学习时间"


with st.sidebar:
    st.header("⚙️ 字段映射")
    st.caption("默认已按实际表头配置，无需修改")
    col_id = st.text_input("员工编码", value=KEY_FIELD)
    col_method = st.text_input("培训方式", value=METHOD_FIELD)
    col_hours = st.text_input("学时", value=HOURS_FIELD)
    col_date = st.text_input("日期", value=DATE_FIELD)
    st.divider()
    sheet_name = st.text_input("表2 Sheet名（留空自动）", value="学时记录")

# ============ 上传数据 ============
st.markdown("### 表1：人员基础信息")
file1 = st.file_uploader("上传人员基础信息", type=["csv","xlsx"], key="f1")
st.markdown("### 表2：培训学时记录")
file2 = st.file_uploader("上传培训学时记录", type=["csv","xlsx"], key="f2")

if file1 and file2:
    # --- 加载 ---
    df_person = pd.read_csv(file1) if file1.name.endswith(".csv") else pd.read_excel(file1)
    if file2.name.endswith(".csv"):
        df_hours = pd.read_csv(file2)
    else:
        xl = pd.ExcelFile(file2)
        sheets = xl.sheet_names
        target = sheet_name if sheet_name and sheet_name in sheets else max(sheets, key=lambda s: pd.read_excel(file2, sheet_name=s).shape[0])
        df_hours = pd.read_excel(file2, sheet_name=target)
        sheet_name = target

    st.session_state.df_person = df_person
    st.session_state.df_hours = df_hours

    # --- 校验关键列 ---
    missing = []
    for c, src in [(col_id, "两表"), (col_method, "表2"), (col_hours, "表2"), (col_date, "表2")]:
        if src == "两表":
            if c not in df_person.columns: missing.append(f"表1缺「{c}」")
            if c not in df_hours.columns: missing.append(f"表2缺「{c}」")
        elif c not in df_hours.columns:
            missing.append(f"表2缺「{c}」")
    if missing:
        st.error("### ⚠️ 缺少关键字段")
        for m in missing: st.write(f"- {m}")
        st.info(f"表2现有列：{list(df_hours.columns)}")
        st.stop()

    # --- 预览 ---
    st.subheader("数据预览")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**表1：人员基础信息**")
        st.dataframe(df_person.head(), use_container_width=True)
        st.caption(f"{len(df_person)} 人")
    with c2:
        st.markdown("**表2：培训学时记录**")
        st.dataframe(df_hours.head(), use_container_width=True)
        st.caption(f"{len(df_hours)} 条 · Sheet「{sheet_name}」")

    if st.button("🚀 开始分析", type="primary"):
        # ===== 规则1: 严格以集团员工编码关联 =====
        merged = df_person[[col_id]].merge(df_hours, on=col_id, how="inner")
        st.info(f"**规则1**：按「{col_id}」严格关联 → 匹配 {len(merged)} 条记录")

        # ===== 规则2: 仅保留三种培训方式 =====
        filtered = merged[merged[col_method].isin(VALID_METHODS)].copy()
        st.info(f"**规则2**：仅保留 {VALID_METHODS} → {len(filtered)} 条")

        # 日期转换为 datetime
        filtered[col_date] = pd.to_datetime(filtered[col_date], errors="coerce")
        filtered = filtered.dropna(subset=[col_date])
        filtered[col_hours] = pd.to_numeric(filtered[col_hours], errors="coerce")
        filtered = filtered.dropna(subset=[col_hours])

        # ===== 规则3+4: 仅对2种特定培训方式做区间分析（不含党委中心组学习）=====
        METHOD_TWO = ["脱产培训(3天以上)", "集中宣讲/专题讲座"]
        filtered_two = filtered[filtered[col_method].isin(METHOD_TWO)]


        if filtered.empty:
            st.warning("无有效记录")
            st.stop()

        # 获取所有人员编码（包括无记录的人）
        all_ids = df_person[col_id].unique()

        # ===== 规则3: 40≤学时<90，仅两种培训方式，降序取最近 =====
        st.info("**规则3**：40≤学时<90，仅筛选脱产培训+集中宣讲，降序取最近一条")
        r40_90 = filtered_two[(filtered_two[col_hours] >= 40) & (filtered_two[col_hours] < 90)]
        if not r40_90.empty:
            r40_90_latest = r40_90.sort_values(col_date, ascending=False).groupby(col_id).first().reset_index()
            # 提取指定字段
            fields_40 = [col_id, "来源信息", "开始学习时间", "完成学习时间", "培训机构", "主办单位"]
            fields_40 = [f for f in fields_40 if f in r40_90_latest.columns]
            recent_40_90 = r40_90_latest[fields_40].copy()
            if "完成学习时间" in recent_40_90.columns:
                recent_40_90["完成学习时间"] = recent_40_90["完成学习时间"].dt.strftime("%Y-%m-%d")
            if "开始学习时间" in recent_40_90.columns:
                recent_40_90["开始学习时间"] = recent_40_90["开始学习时间"].dt.strftime("%Y-%m-%d")
        else:
            recent_40_90 = pd.DataFrame()
        st.info(f"**规则3结果**：40~90区间 → {len(recent_40_90)} 人")

        # ===== 规则4: 学时≥90，仅两种培训方式，降序取最近 =====
        st.info("**规则4**：学时≥90，仅筛选脱产培训+集中宣讲，降序取最近一条")
        r90_plus = filtered_two[filtered_two[col_hours] >= 90]
        if not r90_plus.empty:
            r90_plus_latest = r90_plus.sort_values(col_date, ascending=False).groupby(col_id).first().reset_index()
            fields_90 = [col_id, "来源信息", "开始学习时间", "完成学习时间", "培训机构", "主办单位"]
            fields_90 = [f for f in fields_90 if f in r90_plus_latest.columns]
            recent_90_plus = r90_plus_latest[fields_90].copy()
            if "完成学习时间" in recent_90_plus.columns:
                recent_90_plus["完成学习时间"] = recent_90_plus["完成学习时间"].dt.strftime("%Y-%m-%d")
            if "开始学习时间" in recent_90_plus.columns:
                recent_90_plus["开始学习时间"] = recent_90_plus["开始学习时间"].dt.strftime("%Y-%m-%d")
        else:
            recent_90_plus = pd.DataFrame()
        st.info(f"**规则4结果**：90学时以上 → {len(recent_90_plus)} 人")

        # ===== 累计学时 =====
        total_hours = (
            filtered.groupby(col_id)[col_hours].sum()
            .reset_index()
            .rename(columns={col_hours: "累计培训学时"})
        )
        total_hours["累计培训学时"] = total_hours["累计培训学时"].round(1)

        # ===== 规则5+6+7: 逐人构建整合报表 =====
        st.info("**规则5**：无符合条件记录 — 显示「无符合条件记录」")
        st.info("**规则6**：日期统一格式 YYYY-MM-DD")

        PERSON_COLS = [col_id]
        if "人员姓名" in df_person.columns:
            PERSON_COLS.append("人员姓名")
        if "干部标识" in df_person.columns:
            PERSON_COLS.append("干部标识")

        rows = []
        for pid in all_ids:
            pinfo = df_person[df_person[col_id] == pid][PERSON_COLS].iloc[0] if pid in df_person[col_id].values else {col_id: pid}
            row = {}

            # 基本信息（表1）
            row["员工编码（表1）"] = str(pid)
            row["姓名（表1）"] = str(pinfo.get("人员姓名", ""))
            row["干部标识（表1）"] = str(pinfo.get("干部标识", ""))

            # 模块一：累计学时
            th = total_hours[total_hours[col_id] == pid]
            row["累计培训学时"] = th.iloc[0]["累计培训学时"] if len(th) > 0 else 0.0

            # 模块二：40≤学时<90 最近一次
            if not recent_40_90.empty and pid in recent_40_90[col_id].values:
                rec = recent_40_90[recent_40_90[col_id] == pid].iloc[0]
                row["40~90·班次名称（表2）"] = str(rec.get("来源信息", ""))
                row["40~90·开始学习时间（表2）"] = str(rec.get("开始学习时间", ""))
                row["40~90·完成学习时间（表2）"] = str(rec.get("完成学习时间", ""))
                row["40~90·培训机构（表2）"] = str(rec.get("培训机构", ""))
                row["40~90·主办单位（表2）"] = str(rec.get("主办单位", ""))
            else:
                row["40~90"] = "无符合条件记录"

            # 模块三：学时≥90 最近一次
            if not recent_90_plus.empty and pid in recent_90_plus[col_id].values:
                rec = recent_90_plus[recent_90_plus[col_id] == pid].iloc[0]
                row["90+·班次名称（表2）"] = str(rec.get("来源信息", ""))
                row["90+·开始学习时间（表2）"] = str(rec.get("开始学习时间", ""))
                row["90+·完成学习时间（表2）"] = str(rec.get("完成学习时间", ""))
                row["90+·培训机构（表2）"] = str(rec.get("培训机构", ""))
                row["90+·主办单位（表2）"] = str(rec.get("主办单位", ""))
            else:
                row["90+"] = "无符合条件记录"

            rows.append(row)

        report_df = pd.DataFrame(rows)
        # 数值列转为数字类型，便于AI分析
        if "累计培训学时" in report_df.columns:
            report_df["累计培训学时"] = pd.to_numeric(report_df["累计培训学时"], errors="coerce")
        st.session_state.report_df = report_df
        st.session_state.total_hours = total_hours
        st.session_state.recent_40_90 = recent_40_90
        st.session_state.recent_90_plus = recent_90_plus

        # ===== 规则7: Markdown表格输出 =====
        st.info("**规则7**：表头标注字段来源（表1/表2）")
        st.caption(f"共 {len(report_df)} 人 | 筛选方式：{VALID_METHODS} | 关联键：{col_id}")

        # ---- 导出 ----
        with st.expander("📥 导出 Excel"):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine="openpyxl") as writer:
                report_df.to_excel(writer, sheet_name="整合报表", index=False)
                total_hours.to_excel(writer, sheet_name="累计学时", index=False)
                if not recent_40_90.empty:
                    recent_40_90.to_excel(writer, sheet_name="40-90学时详情", index=False)
                if not recent_90_plus.empty:
                    recent_90_plus.to_excel(writer, sheet_name="90+学时详情", index=False)
            st.download_button("下载 Excel", output.getvalue(), "HR培训分析结果.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ============ AI 深度分析（基于分析结果）============
if st.session_state.report_df is not None:
    st.divider()
    st.subheader("🤖 AI 深度分析")

    report_df = st.session_state.report_df
    total_hours = st.session_state.total_hours
    recent_40_90 = st.session_state.recent_40_90
    recent_90_plus = st.session_state.recent_90_plus
    df_person = st.session_state.df_person
    df_hours = st.session_state.df_hours
    with st.expander("⚙️ 配置 DeepSeek API Key", expanded=False):
        ds_key = st.text_input("API Key", type="password", placeholder="sk-...", key="ds_key_post")
    ds_key = st.session_state.get("ds_key_post", "")
    if ds_key:
        client = OpenAI(
            api_key=st.session_state.ds_key_post,
            base_url="https://api.deepseek.com"
        )
        ai_question = st.text_area("对分析结果提问", placeholder="例：哪些部门的人均学时最高？40~90区间人数最多的培训是什么？", key="ai_q_post")
        if st.button("AI 分析", key="ai_btn_post"):
            if ai_question:
                safe_q = ai_question.replace("{", "{{").replace("}", "}}")
                # 构建报表摘要
                report_summary = f"""分析报表(report_df)共{len(report_df)}人，
字段:
- 员工编码（表1）、姓名（表1）、干部标识（表1）
- 累计培训学时
- 40~90区间: 班次名称、开始学习时间、完成学习时间、培训机构、主办单位（表2）
- 90+区间: 班次名称、开始学习时间、完成学习时间、培训机构、主办单位（表2）

累计培训总学时: {total_hours["累计培训学时"].sum():.0f}
40~90区间有记录人数: {len(recent_40_90)}
90+区间有记录人数: {len(recent_90_plus)}

报表前20行:
{report_df.head(20).to_markdown()}"""

                ai_prompt = f"""你是HR数据分析专家。用户上传了两张表（人员基础信息、培训学时记录），
经过固定流程处理后生成了分析报表report_df。请根据report_df回答用户问题。

{report_summary}

用户问题: {safe_q}

请直接对report_df进行操作（筛选、统计、排序、分组等），
如需要也可使用total_hours、recent_40_90、recent_90_plus辅助。
只返回Python代码，不输出解释和```标记。
结果存到result变量，如需文字总结存到report变量（Markdown格式）。
如需绘图用matplotlib（fontproperties=CN_FONT，不调plt.show()）。"""
                with st.spinner("AI分析中..."):
                    resp = client.chat.completions.create(
                        model="deepseek-v4-pro",
                        messages=[{"role":"user","content":ai_prompt}]
                    )
                    code = resp.choices[0].message.content
                    code = code.strip()
                    if code.startswith("```"):
                        code = code.split("\n",1)[1] if "\n" in code else code[3:]
                    if code.endswith("```"):
                        code = code.rsplit("\n",1)[0]
                    code = code.strip()
                    lvars = {
                        "df_person":df_person,
                        "df_hours":df_hours,
                        "total_hours":total_hours,
                        "recent_40_90":recent_40_90,
                        "recent_90_plus":recent_90_plus,
                        "report_df":report_df,
                        "pd":pd,"plt":plt,"CN_FONT":CN_FONT
                    }
                    try:
                        out = io.StringIO()
                        with contextlib.redirect_stdout(out):
                            exec(code,{},lvars)
                        if "result" in lvars:
                            r = lvars["result"]
                            if isinstance(r,pd.DataFrame):
                                st.dataframe(r)
                            else:
                                st.write(r)
                        if "report" in lvars:
                            st.markdown(lvars["report"])
                        fig = plt.gcf()
                        if len(fig.axes)>0:
                            st.pyplot(fig)
                            plt.clf()
                    except Exception as e:
                        with st.expander("查看生成的代码"):
                            st.code(code, language="python")
                        st.error(f"执行出错: {e}")
