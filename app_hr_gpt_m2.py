import os
import io
from typing import Optional

import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager

from openai import OpenAI


# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="干部教育培训统计助手",
    page_icon="📊",
    layout="wide"
)

st.title("📊 干部教育培训统计助手")

st.caption(
    "自动完成干部培训统计、班次提取、累计学时统计，并生成分析报告。"
)

# ============================================================
# 项目目录
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FONT_PATH = os.path.join(
    BASE_DIR,
    "fonts",
    "NotoSansCJK-Regular.ttc"
)

CN_FONT = None

if os.path.exists(FONT_PATH):

    CN_FONT = FontProperties(fname=FONT_PATH)

    font_manager.fontManager.addfont(FONT_PATH)

    matplotlib.rcParams["font.family"] = CN_FONT.get_name()

matplotlib.rcParams["axes.unicode_minus"] = False

# ============================================================
# DeepSeek
# （仅生成分析报告）
# ============================================================

if "api_key" not in st.session_state:
    st.session_state.api_key = ""

with st.sidebar:

    st.header("DeepSeek")

    with st.form("token"):

        token = st.text_input(
            "API Key",
            type="password"
        )

        ok = st.form_submit_button("保存")

    if ok:

        st.session_state.api_key = token

        st.success("保存成功")

client = None

if st.session_state.api_key:

    client = OpenAI(

        api_key=st.session_state.api_key,

        base_url="https://api.deepseek.com"

    )

# ============================================================
# 全局配置
# ============================================================

VALID_TRAINING_TYPES = [

    "党委(党组)理论学习中心组学习",

    "脱产培训(3天以上)",

    "集中宣讲/专题讲座"

]

# ============================================================
# 字段别名
# 自动识别不同Excel
# ============================================================

FIELD_ALIAS = {

    "employee_code":[

        "集团员工编码",

        "员工编码",

        "人员编码",

        "工号",

        "员工工号"

    ],

    "name":[

        "姓名"

    ],

    "department":[

        "部门",

        "所在部门",

        "单位",

        "所属部门"

    ],

    "training_name":[

        "培训班名称",

        "班次名称",

        "培训名称",

        "培训项目"

    ],

    "training_type":[

        "培训方式",

        "培训类别"

    ],

    "hours":[

        "培训学时",

        "学时"

    ],

    "start_date":[

        "学习开始时间",

        "开始时间",

        "培训开始时间",

        "培训开始日期"

    ],

    "end_date":[

        "学习结束时间",

        "结束时间",

        "培训结束时间",

        "培训结束日期"

    ]

}

# ============================================================
# 工具函数
# ============================================================

def find_column(
    df: pd.DataFrame,
    aliases: list
) -> Optional[str]:

    """
    根据字段别名寻找真实字段
    """

    for col in aliases:

        if col in df.columns:

            return col

    return None


def load_dataframe(file):

    """
    自动读取Excel/CSV
    """

    if file is None:

        return None

    suffix = file.name.lower()

    if suffix.endswith(".xlsx"):

        df = pd.read_excel(file)

    elif suffix.endswith(".csv"):

        try:

            df = pd.read_csv(
                file,
                encoding="utf-8"
            )

        except:

            file.seek(0)

            try:

                df = pd.read_csv(
                    file,
                    encoding="gbk"
                )

            except:

                file.seek(0)

                df = pd.read_csv(file)

    else:

        return None

    df.columns = [

        str(c).strip()

        for c in df.columns

    ]

    return df


def format_date(series):

    """
    日期统一格式
    """

    s = pd.to_datetime(

        series,

        errors="coerce"

    )

    return s.dt.strftime("%Y-%m-%d")


def safe_numeric(series):

    """
    安全转数字
    """

    return pd.to_numeric(

        series,

        errors="coerce"

    ).fillna(0)

# ============================================================
# 上传数据
# ============================================================

st.subheader("① 上传数据")

col1, col2 = st.columns(2)

with col1:

    employee_file = st.file_uploader(
        "表1：干部人员基础信息",
        type=["xlsx", "csv"],
        key="employee"
    )

with col2:

    training_file = st.file_uploader(
        "表2：培训学时记录",
        type=["xlsx", "csv"],
        key="training"
    )

if employee_file is None or training_file is None:

    st.info("请上传两张数据表。")

    st.stop()

employee_df = load_dataframe(employee_file)
training_df = load_dataframe(training_file)

# ============================================================
# 数据预览
# ============================================================

with st.expander("表1预览"):

    st.dataframe(
        employee_df.head(),
        use_container_width=True
    )

with st.expander("表2预览"):

    st.dataframe(
        training_df.head(),
        use_container_width=True
    )

# ============================================================
# 字段识别
# ============================================================

employee_code_col = find_column(
    employee_df,
    FIELD_ALIAS["employee_code"]
)

employee_name_col = find_column(
    employee_df,
    FIELD_ALIAS["name"]
)

employee_department_col = find_column(
    employee_df,
    FIELD_ALIAS["department"]
)

training_code_col = find_column(
    training_df,
    FIELD_ALIAS["employee_code"]
)

training_name_col = find_column(
    training_df,
    FIELD_ALIAS["training_name"]
)

training_type_col = find_column(
    training_df,
    FIELD_ALIAS["training_type"]
)

training_hours_col = find_column(
    training_df,
    FIELD_ALIAS["hours"]
)

training_start_col = find_column(
    training_df,
    FIELD_ALIAS["start_date"]
)

training_end_col = find_column(
    training_df,
    FIELD_ALIAS["end_date"]
)

# ============================================================
# 必填字段检查
# ============================================================

missing = []

if employee_code_col is None:
    missing.append("集团员工编码(表1)")

if employee_name_col is None:
    missing.append("姓名(表1)")

if employee_department_col is None:
    missing.append("部门(表1)")

if training_code_col is None:
    missing.append("集团员工编码(表2)")

if training_name_col is None:
    missing.append("培训班名称(表2)")

if training_type_col is None:
    missing.append("培训方式(表2)")

if training_hours_col is None:
    missing.append("培训学时(表2)")

if training_start_col is None:
    missing.append("学习开始时间(表2)")

if training_end_col is None:
    missing.append("学习结束时间(表2)")

if len(missing):

    st.error("缺少以下字段：")

    for m in missing:

        st.write("•", m)

    st.stop()

# ============================================================
# 字段统一
# ============================================================

employee_df = employee_df.rename(columns={

    employee_code_col: "集团员工编码",

    employee_name_col: "姓名",

    employee_department_col: "部门"

})

training_df = training_df.rename(columns={

    training_code_col: "集团员工编码",

    training_name_col: "培训班名称",

    training_type_col: "培训方式",

    training_hours_col: "培训学时",

    training_start_col: "学习开始时间",

    training_end_col: "学习结束时间"

})

# ============================================================
# 数据清洗
# ============================================================

employee_df["集团员工编码"] = (
    employee_df["集团员工编码"]
    .astype(str)
    .str.strip()
)

training_df["集团员工编码"] = (
    training_df["集团员工编码"]
    .astype(str)
    .str.strip()
)

employee_df = employee_df[
    employee_df["集团员工编码"] != ""
]

training_df = training_df[
    training_df["集团员工编码"] != ""
]

employee_df = employee_df.drop_duplicates(
    subset=["集团员工编码"]
)

training_df = training_df.drop_duplicates()

training_df["培训学时"] = safe_numeric(
    training_df["培训学时"]
)

training_df["学习开始时间"] = pd.to_datetime(
    training_df["学习开始时间"],
    errors="coerce"
)

training_df["学习结束时间"] = pd.to_datetime(
    training_df["学习结束时间"],
    errors="coerce"
)

# ============================================================
# 培训方式过滤
# ============================================================

training_df = training_df[
    training_df["培训方式"].isin(
        VALID_TRAINING_TYPES
    )
].copy()

# ============================================================
# Inner Join
# ============================================================

merged_df = training_df.merge(

    employee_df,

    on="集团员工编码",

    how="inner"

)

st.success(
    f"有效人员：{len(employee_df)} 人；"
    f"有效培训记录：{len(training_df)} 条；"
    f"关联后记录：{len(merged_df)} 条"
)

# ============================================================
# 开始统计
# ============================================================

if not st.button(
    "开始统计",
    type="primary",
    use_container_width=True
):

    st.stop()
# ============================================================
# ② 累计培训学时
# ============================================================

hour_df = (
    merged_df
    .groupby("集团员工编码", as_index=False)["培训学时"]
    .sum()
    .rename(columns={
        "培训学时": "累计培训学时"
    })
)

# ============================================================
# 生成排序日期
# ============================================================

merged_df["排序日期"] = merged_df["学习结束时间"]

mask = merged_df["排序日期"].isna()

merged_df.loc[
    mask,
    "排序日期"
] = merged_df.loc[
    mask,
    "学习开始时间"
]

# ============================================================
# 最近一次40~90学时培训
# ============================================================

middle_df = merged_df[

    (merged_df["培训学时"] >= 40)

    &

    (merged_df["培训学时"] < 90)

    &

    (
        merged_df["培训方式"].isin([
            "脱产培训(3天以上)",
            "集中宣讲/专题讲座"
        ])
    )

].copy()

middle_df = middle_df.sort_values(

    ["集团员工编码", "排序日期"],

    ascending=[True, False]

)

middle_df = (

    middle_df

    .drop_duplicates(

        subset=["集团员工编码"],

        keep="first"

    )

)

middle_df = middle_df[[
    "集团员工编码",
    "培训班名称",
    "培训方式",
    "培训学时",
    "学习开始时间",
    "学习结束时间"
]]

middle_df = middle_df.rename(columns={

    "培训班名称":"40~90班次",

    "培训方式":"40~90培训方式",

    "培训学时":"40~90学时",

    "学习开始时间":"40~90开始时间",

    "学习结束时间":"40~90结束时间"

})

# ============================================================
# 最近一次90学时以上培训
# ============================================================

high_df = merged_df[

    merged_df["培训学时"] >= 90

].copy()

high_df = high_df[

    high_df["培训方式"].isin([

        "脱产培训(3天以上)",

        "集中宣讲/专题讲座"

    ])

]

high_df = high_df.sort_values(

    ["集团员工编码","排序日期"],

    ascending=[True,False]

)

high_df = (

    high_df

    .drop_duplicates(

        subset=["集团员工编码"],

        keep="first"

    )

)

high_df = high_df[[

    "集团员工编码",

    "培训班名称",

    "培训方式",

    "培训学时",

    "学习开始时间",

    "学习结束时间"

]]

high_df = high_df.rename(columns={

    "培训班名称":"90以上班次",

    "培训方式":"90以上培训方式",

    "培训学时":"90以上学时",

    "学习开始时间":"90以上开始时间",

    "学习结束时间":"90以上结束时间"

})

# ============================================================
# 人员基础信息
# ============================================================

base_df = employee_df[[
    "集团员工编码",
    "姓名",
    "部门"
]].copy()

# ============================================================
# 合并统计结果
# ============================================================

final_df = (

    base_df

    .merge(

        hour_df,

        how="left",

        on="集团员工编码"

    )

    .merge(

        middle_df,

        how="left",

        on="集团员工编码"

    )

    .merge(

        high_df,

        how="left",

        on="集团员工编码"

    )

)

# ============================================================
# 累计学时空值
# ============================================================

final_df["累计培训学时"] = (

    final_df["累计培训学时"]

    .fillna(0)

)

# ============================================================
# 日期统一格式
# ============================================================

date_columns = [

    "40~90开始时间",

    "40~90结束时间",

    "90以上开始时间",

    "90以上结束时间"

]

for col in date_columns:

    if col in final_df.columns:

        final_df[col] = format_date(

            final_df[col]

        )

# ============================================================
# 无符合条件记录
# ============================================================

middle_cols = [

    "40~90班次",

    "40~90培训方式",

    "40~90学时",

    "40~90开始时间",

    "40~90结束时间"

]

for col in middle_cols:

    final_df[col] = (

        final_df[col]

        .fillna("无符合条件记录")

    )

high_cols = [

    "90以上班次",

    "90以上培训方式",

    "90以上学时",

    "90以上开始时间",

    "90以上结束时间"

]

for col in high_cols:

    final_df[col] = (

        final_df[col]

        .fillna("无符合条件记录")

    )

# ============================================================
# 排序
# ============================================================

final_df = final_df.sort_values(

    "集团员工编码"

).reset_index(drop=True)

result = final_df
# ============================================================
# 工具函数
# ============================================================

def extract_latest_training(
    df: pd.DataFrame,
    min_hours: float,
    max_hours: float | None,
    prefix: str
) -> pd.DataFrame:
    """
    提取每人最近一次符合条件的培训记录
    """

    data = df.copy()

    if max_hours is None:

        data = data[
            data["培训学时"] >= min_hours
        ]

    else:

        data = data[
            (data["培训学时"] >= min_hours)
            &
            (data["培训学时"] < max_hours)
        ]

    data = data[
        data["培训方式"].isin(
            [
                "脱产培训(3天以上)",
                "集中宣讲/专题讲座"
            ]
        )
    ]

    data = data.sort_values(

        ["集团员工编码", "排序日期"],

        ascending=[True, False]

    )

    data = data.drop_duplicates(

        subset=["集团员工编码"],

        keep="first"

    )

    data = data[
        [
            "集团员工编码",
            "培训班名称",
            "培训方式",
            "培训学时",
            "学习开始时间",
            "学习结束时间"
        ]
    ]

    data = data.rename(columns={

        "培训班名称": f"{prefix}班次",

        "培训方式": f"{prefix}培训方式",

        "培训学时": f"{prefix}学时",

        "学习开始时间": f"{prefix}开始时间",

        "学习结束时间": f"{prefix}结束时间"

    })

    return data


# ============================================================
# 重新生成两个模块
# ============================================================

middle_df = extract_latest_training(
    merged_df,
    40,
    90,
    "40~90"
)

high_df = extract_latest_training(
    merged_df,
    90,
    None,
    "90以上"
)

# ============================================================
# result/display/markdown 三份数据
# ============================================================

result_df = final_df.copy()

display_df = final_df.copy()

markdown_df = final_df.copy()

# ============================================================
# 页面展示专用处理
# ============================================================

display_columns = [

    "40~90班次",
    "40~90培训方式",
    "40~90学时",
    "40~90开始时间",
    "40~90结束时间",
    "90以上班次",
    "90以上培训方式",
    "90以上学时",
    "90以上开始时间",
    "90以上结束时间"

]

for col in display_columns:

    display_df[col] = display_df[col].fillna(
        "无符合条件记录"
    )

# ============================================================
# 日期格式
# ============================================================

date_cols = [

    "40~90开始时间",
    "40~90结束时间",
    "90以上开始时间",
    "90以上结束时间"

]

for col in date_cols:

    display_df[col] = format_date(
        display_df[col]
    )

    markdown_df[col] = format_date(
        markdown_df[col]
    )

# ============================================================
# Markdown展示专用
# ============================================================

for col in display_columns:

    markdown_df[col] = markdown_df[col].fillna(
        "无符合条件记录"
    )

markdown_table = markdown_df.to_markdown(
    index=False
)

# ============================================================
# 页面
# ============================================================

st.subheader("② 三模块统计结果")

st.dataframe(

    display_df,

    use_container_width=True,

    hide_index=True

)

result = result_df

# ============================================================
# 统计摘要
# ============================================================

summary = {

    "人员总数":

        len(employee_df),

    "有效培训记录":

        len(training_df),

    "累计培训学时":

        round(

            result_df["累计培训学时"].sum(),

            2

        ),

    "平均培训学时":

        round(

            result_df["累计培训学时"].mean(),

            2

        ),

    "40~90学时人数":

        int(

            (
                result_df["40~90班次"]

                .notna()

            ).sum()

        ),

    "90学时以上人数":

        int(

            (
                result_df["90以上班次"]

                .notna()

            ).sum()

        )

}

# ============================================================
# 页面摘要
# ============================================================

st.subheader("统计摘要")

c1, c2, c3 = st.columns(3)

c1.metric(

    "干部人数",

    summary["人员总数"]

)

c2.metric(

    "有效培训记录",

    summary["有效培训记录"]

)

c3.metric(

    "累计培训学时",

    summary["累计培训学时"]

)

c1, c2, c3 = st.columns(3)

c1.metric(

    "平均培训学时",

    summary["平均培训学时"]

)

c2.metric(

    "40~90学时人数",

    summary["40~90学时人数"]

)

c3.metric(

    "90学时以上人数",

    summary["90学时以上人数"]

)

# ============================================================
# Markdown
# ============================================================

st.subheader("Markdown")

st.code(

    markdown_table,

    language="markdown"

)

# ============================================================
# Excel下载
# ============================================================

excel = io.BytesIO()

with pd.ExcelWriter(

    excel,

    engine="openpyxl"

) as writer:

    result_df.to_excel(

        writer,

        sheet_name="统计结果",

        index=False

    )

excel.seek(0)

st.download_button(

    "下载Excel",

    data=excel,

    file_name="干部培训统计.xlsx",

    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

    use_container_width=True

)

# ============================================================
# Markdown下载
# ============================================================

st.download_button(

    "下载Markdown",

    data=markdown_table,

    file_name="统计结果.md",

    mime="text/markdown",

    use_container_width=True

)

# ============================================================
# AI分析报告（仅文本分析）
# ============================================================

st.subheader("③ AI分析报告")

if client is None:

    st.info("未配置 DeepSeek API Key，跳过AI分析。")

else:

    report_df = result_df.copy()

    report_preview = report_df.head(50).to_markdown(index=False)

    report_prompt = f"""
你是一名干部教育培训分析专家。

下面是某单位干部教育培训统计结果。

请不要重新计算数据。

不要猜测不存在的信息。

仅依据下面统计结果进行分析。

========================

统计摘要：

干部总人数：

{summary["人员总数"]}

有效培训记录：

{summary["有效培训记录"]}

累计培训学时：

{summary["累计培训学时"]}

平均培训学时：

{summary["平均培训学时"]}

40~90学时人数：

{summary["40~90学时人数"]}

90学时以上人数：

{summary["90学时以上人数"]}

========================

统计表（仅展示前50行）：

{report_preview}

========================

请生成：

# 一、总体情况

总体培训情况。

# 二、培训特点

例如：

累计学时特点

40~90学时培训情况

90学时以上培训情况

# 三、存在问题

例如：

培训覆盖不足

高学时培训偏少

部分干部培训不足等。

如果统计结果不能证明，请不要编造。

# 四、建议

给出3~5条建议。

要求：

Markdown格式。

不要重复统计数字。

不要输出代码。

不要输出Markdown代码块。

"""

    try:

        with st.spinner("DeepSeek 正在生成分析报告..."):

            response = client.chat.completions.create(

                model="deepseek-chat",

                messages=[

                    {

                        "role":"user",

                        "content":report_prompt

                    }

                ]

            )

        report = response.choices[0].message.content

        st.markdown(report)

    except Exception as e:

        st.error("AI分析失败")

        st.exception(e)

# ============================================================
# 调试信息
# ============================================================

with st.expander("查看Prompt"):

    st.text(report_prompt)

with st.expander("查看统计DataFrame"):

    st.dataframe(

        result_df,

        use_container_width=True

    )

# ============================================================
# 程序结束
# ============================================================

st.success("统计完成。")