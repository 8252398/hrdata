import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import matplotlib

matplotlib.use("Agg")

from openai import OpenAI

import io
import os
import contextlib
import re

# ==========================================================
# 项目目录
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_PATH = os.path.join(
    BASE_DIR,
    "fonts",
    "NotoSansCJK-Regular.ttc"
)

if os.path.exists(FONT_PATH):
    CN_FONT = FontProperties(fname=FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams["font.family"] = CN_FONT.get_name()
else:
    CN_FONT = None

matplotlib.rcParams["axes.unicode_minus"] = False

# ==========================================================
# 页面
# ==========================================================

st.set_page_config(
    page_title="干部培训统计助手",
    layout="wide"
)

st.title("👥 干部培训统计助手")

st.caption("上传干部基础信息和培训学时记录，自动生成统计结果。")

# ==========================================================
# API Key
# ==========================================================

if "ds_token" not in st.session_state:
    st.session_state.ds_token = ""

with st.sidebar:

    st.header("⚙ DeepSeek 配置")

    with st.form(
        "token_form",
        clear_on_submit=False
    ):

        token = st.text_input(
            "DeepSeek API Key",
            type="password",
            placeholder="请输入API Key"
        )

        ok = st.form_submit_button("保存")

    if ok:

        if token:

            st.session_state.ds_token = token

            st.success("Token 已保存")

        else:

            st.warning("请输入API Key")

if not st.session_state.ds_token:

    st.info("请先在左侧输入 DeepSeek API Key。")

    st.stop()

# ==========================================================
# DeepSeek
# ==========================================================

client = OpenAI(
    api_key=st.session_state.ds_token,
    base_url="https://api.deepseek.com"
)

# ==========================================================
# 工具函数
# ==========================================================

def read_dataframe(uploaded_file):

    if uploaded_file is None:
        return None

    suffix = uploaded_file.name.lower()

    if suffix.endswith(".xlsx"):

        return pd.read_excel(uploaded_file)

    if suffix.endswith(".csv"):

        try:

            return pd.read_csv(uploaded_file, encoding="utf-8")

        except:

            uploaded_file.seek(0)

            try:

                return pd.read_csv(uploaded_file, encoding="gbk")

            except:

                uploaded_file.seek(0)

                return pd.read_csv(uploaded_file)

    return None


def normalize_columns(df):

    cols = []

    for c in df.columns:

        c = str(c).strip()

        c = c.replace("\n", "")

        c = c.replace("\r", "")

        cols.append(c)

    df.columns = cols

    return df


def preview_dataframe(name, df):

    with st.expander(
        f"{name}（{len(df)} 行 × {len(df.columns)} 列）"
    ):

        st.dataframe(df.head())


# ==========================================================
# 上传数据
# ==========================================================

st.subheader("① 上传数据")

employee_file = st.file_uploader(
    "上传【表1：干部人员基础信息】",
    type=["xlsx", "csv"],
    key="employee"
)

training_file = st.file_uploader(
    "上传【表2：培训学时记录】",
    type=["xlsx", "csv"],
    key="training"
)

employee_df = None
training_df = None

if employee_file:

    employee_df = read_dataframe(employee_file)

    employee_df = normalize_columns(employee_df)

if training_file:

    training_df = read_dataframe(training_file)

    training_df = normalize_columns(training_df)

# ==========================================================
# 数据预览
# ==========================================================

if employee_df is not None:

    preview_dataframe(
        "表1：干部基础信息",
        employee_df
    )

if training_df is not None:

    preview_dataframe(
        "表2：培训记录",
        training_df
    )

# ==========================================================
# 字段检查
# ==========================================================

EMPLOYEE_REQUIRED = [
    "集团员工编码"
]

TRAINING_REQUIRED = [
    "集团员工编码",
    "培训方式",
    "培训学时"
]

employee_ok = True
training_ok = True

if employee_df is not None:

    missing = []

    for c in EMPLOYEE_REQUIRED:

        if c not in employee_df.columns:

            missing.append(c)

    if len(missing):

        employee_ok = False

        st.error(
            "人员基础信息缺少字段："
            + "、".join(missing)
        )

if training_df is not None:

    missing = []

    for c in TRAINING_REQUIRED:

        if c not in training_df.columns:

            missing.append(c)

    if len(missing):

        training_ok = False

        st.error(
            "培训记录缺少字段："
            + "、".join(missing)
        )

if employee_df is None or training_df is None:

    st.stop()

if not employee_ok:

    st.stop()

if not training_ok:

    st.stop()
# ==========================================================
# 用户分析需求
# ==========================================================

st.subheader("② 分析需求")

default_question = """生成干部教育培训统计报表"""

question = st.text_area(
    "请输入分析需求",
    value=default_question,
    height=120
)

# ==========================================================
# 开始分析
# ==========================================================

if st.button(
    "开始分析",
    type="primary",
    use_container_width=True
):

    safe_question = (
        question
        .replace("{", "{{")
        .replace("}", "}}")
    )

    employee_info = f"""
变量名：employee_df

字段：

{list(employee_df.columns)}

前5行：

{employee_df.head().to_markdown(index=False)}
"""

    training_info = f"""
变量名：training_df

字段：

{list(training_df.columns)}

前5行：

{training_df.head().to_markdown(index=False)}
"""

    prompt = f"""
你是一名精通 Pandas 的Python数据分析专家。

当前已经存在两个DataFrame：

======================================
{employee_info}

======================================
{training_info}

======================================

用户需求：

{safe_question}

======================================

必须严格遵守以下要求：

【返回格式】

1、只返回Python代码。

2、不要输出任何解释。

3、不要输出```python。

4、不要输出```。

5、不允许定义函数。

6、不允许读取Excel。

7、不允许保存Excel。

8、不允许重新创建DataFrame。

9、直接使用：

employee_df

training_df

pd

plt

CN_FONT

=====================================================

【业务规则】

第一步：

必须以

集团员工编码

作为唯一关联字段。

严禁使用：

姓名

身份证号

部门

手机号

等任何其它字段。

=====================================================

第二步：

必须删除

employee_df

不存在的集团员工编码。

必须使用：

inner join

=====================================================

第三步：

仅保留以下三种培训方式：

党委(党组)理论学习中心组学习

脱产培训(3天以上)

集中宣讲/专题讲座

其它培训方式全部删除。

=====================================================

第四步：

培训学时必须转换成数值。

日期必须转换成datetime。

所有日期最终显示格式：

YYYY-MM-DD

=====================================================

第五步：

按集团员工编码统计：

累计培训学时

=====================================================

第六步：

提取最近一次

40≤培训学时<90

且培训方式属于：

脱产培训(3天以上)

集中宣讲/专题讲座

排序规则：

优先按

学习结束时间

降序。

若不存在结束时间，

则按

学习开始时间

降序。

保留最近一次。

输出字段：

培训班名称

培训方式

培训学时

学习开始时间

学习结束时间

=====================================================

第七步：

提取最近一次

培训学时≥90

且培训方式属于：

脱产培训(3天以上)

集中宣讲/专题讲座

排序规则完全一致。

=====================================================

第八步：

若没有符合条件的数据，

必须填写：

无符合条件记录

禁止为空。

=====================================================

第九步：

最终每个人只保留一行。

最终DataFrame建议包含：

【人员信息】

集团员工编码

姓名

单位

部门

【模块一】

累计培训学时

【模块二】

40~90班次

40~90培训方式

40~90学时

40~90开始时间

40~90结束时间

【模块三】

90以上班次

90以上培训方式

90以上学时

90以上开始时间

90以上结束时间

=====================================================

第十步：

生成Markdown表格。

保存到：

markdown_table

格式：

final_df.to_markdown(index=False)

=====================================================

最后必须生成三个变量：

result = final_df

report = Markdown分析报告

markdown_table = final_df.to_markdown(index=False)

=====================================================

绘图要求：

如果需要绘图：

禁止plt.show()

中文全部使用：

fontproperties=CN_FONT

=====================================================

禁止修改：

employee_df

training_df

只能新建变量。

代码必须可以直接运行。

"""
    # ==========================================================
    # 调用 DeepSeek
    # ==========================================================

    with st.spinner("DeepSeek 正在分析，请稍候..."):

        response = client.chat.completions.create(

            model="deepseek-v4-pro",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

    code = response.choices[0].message.content.strip()

    # ==========================================================
    # 清理 Markdown 代码块
    # ==========================================================

    if code.startswith("```"):

        lines = code.split("\n")

        lines = lines[1:]

        if len(lines) > 0 and lines[-1].strip() == "```":

            lines = lines[:-1]

        code = "\n".join(lines)

    code = code.strip()

    # ==========================================================
    # 执行代码
    # ==========================================================

    output = io.StringIO()

    local_vars = {

        "employee_df": employee_df,

        "training_df": training_df,

        "pd": pd,

        "plt": plt,

        "CN_FONT": CN_FONT

    }

    try:

        with contextlib.redirect_stdout(output):

            exec(
                code,
                {},
                local_vars
            )

        # ======================================================
        # DataFrame
        # ======================================================

        if "result" in local_vars:

            st.subheader("统计结果")

            result = local_vars["result"]

            if isinstance(result, pd.DataFrame):

                st.dataframe(
                    result,
                    use_container_width=True
                )

            else:

                st.write(result)

        # ======================================================
        # Markdown 表格
        # ======================================================

        if "markdown_table" in local_vars:

            st.subheader("Markdown报表")

            st.markdown(

                local_vars["markdown_table"]

            )

        # ======================================================
        # 分析报告
        # ======================================================

        if "report" in local_vars:

            st.subheader("分析报告")

            st.markdown(

                local_vars["report"]

            )

        # ======================================================
        # 图表
        # ======================================================

        fig = plt.gcf()

        if len(fig.axes):

            st.subheader("分析图表")

            st.pyplot(fig)

            plt.clf()

        # ======================================================
        # 控制台输出
        # ======================================================

        if output.getvalue():

            st.subheader("程序输出")

            st.text(

                output.getvalue()

            )

        # ======================================================
        # Excel 下载
        # ======================================================

        if "result" in local_vars:

            result = local_vars["result"]

            if isinstance(result, pd.DataFrame):

                excel_buffer = io.BytesIO()

                with pd.ExcelWriter(
                    excel_buffer,
                    engine="openpyxl"
                ) as writer:

                    result.to_excel(
                        writer,
                        index=False,
                        sheet_name="统计结果"
                    )

                excel_buffer.seek(0)

                st.download_button(

                    label="📥 下载统计结果 Excel",

                    data=excel_buffer,

                    file_name="干部教育培训统计结果.xlsx",

                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",

                    use_container_width=True

                )

    except Exception as e:

        st.error("AI生成的代码执行失败")

        st.exception(e)

        with st.expander("查看AI生成代码"):

            st.code(

                code,

                language="python"

            )