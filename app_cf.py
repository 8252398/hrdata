import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import matplotlib
from openai import OpenAI
import io
import contextlib
import os


# 当前项目目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# 字体文件
FONT_PATH = os.path.join(BASE_DIR, "fonts", "NotoSansCJK-Regular.ttc")

# 创建字体对象
CN_FONT = FontProperties(fname=FONT_PATH)

# 设置全局字体
# 注册字体到matplotlib字体管理器，确保rcParams全局生效
font_manager.fontManager.addfont(FONT_PATH)
matplotlib.rcParams["font.family"] = CN_FONT.get_name()

# 防止负号显示成方块
matplotlib.rcParams["axes.unicode_minus"] = False

st.set_page_config(page_title="课程数据分析助手")

st.title("📊 课程数据分析助手")

# ===============================
# API Key 输入
# ===============================

if "cf_token" not in st.session_state:
    st.session_state.cf_token = ""

with st.sidebar:
    st.header("⚙️ 配置")

    with st.form("cf_form", clear_on_submit=False):
        cf_token_input = st.text_input(
            "Cloudflare API Token",
            type="password",
            placeholder="输入你的 API Token"
        )
        submitted = st.form_submit_button("确认")
    if submitted:
        if cf_token_input:
            st.session_state.cf_token = cf_token_input
            st.success("Token 已保存")
        else:
            st.warning("请输入 Token")
if not st.session_state.cf_token:
    st.info("👈 请在左侧输入 Cloudflare API Token 后开始使用")
    st.stop()


# ===============================
# Cloudflare Workers AI 客户端
# ===============================

CF_ACCOUNT_ID = "b71aa7ad5579027a5027c0428d21e500"

client = OpenAI(
    api_key=st.session_state.cf_token,
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCOUNT_ID}/ai/v1"
)

# ===============================
# 上传数据
# ===============================

uploaded_file = st.file_uploader(
    "上传CSV或Excel",
    type=["csv","xlsx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("数据预览")

    st.dataframe(df.head())

    question = st.text_area(
        "请输入你的问题",
        placeholder="例如：统计成绩最高的三个人，并画出柱状图"
    )

    if st.button("开始分析"):

        safe_question = question.replace("{", "{{").replace("}", "}}")
        prompt = f"""
你是一名Python数据分析专家。

当前DataFrame变量名叫df。

df字段如下：

{list(df.columns)}

数据前5行：

{df.head().to_markdown()}

用户的问题：

{safe_question}


你是一名专业的 Python 数据分析专家。

当前已经存在以下变量：

- df：Pandas DataFrame，包含用户上传的数据
- pd：pandas
- plt：matplotlib.pyplot
- CN_FONT：Matplotlib 中文字体(FontProperties)

请根据用户的问题，编写可以直接运行的 Python 代码。

必须遵守以下要求：

1. 只返回 Python 代码。
2. 不要输出任何解释。
3. 不要输出 ```python 或 ```。
4. 不要定义函数。
5. 不要读取任何文件，也不要保存任何文件。
6. 不要重新创建 DataFrame，直接使用变量 df。
7. 除了结果数据，还需要生成一段文字分析报告，保存到变量 report（字符串，可使用Markdown格式）。
8. 最终分析结果数据必须保存到变量 result。
9. 如果需要绘图，请使用 matplotlib。
10. 不要调用 plt.show()。
11. 图表标题、X轴、Y轴、图例等所有中文文字必须使用：
    fontproperties=CN_FONT
12. 不要修改原始 DataFrame，如需处理，请使用新的变量。
13. 如果结果是 DataFrame、Series、数字或字符串，都放入变量 result。
14. 如果需要排序，请明确指定 ascending=True 或 False。
15. 不要使用不存在的字段，如果字段不存在，请根据已有字段进行分析。
16. 尽量编写简洁、可读、符合 pandas 最佳实践的代码。

下面是正确的示例：

用户问题：
统计各部门平均成绩，并绘制柱状图。

返回：

avg_score = df.groupby("部门")["成绩"].mean()

avg_score.plot(
    kind="bar"
)

plt.title(
    "各部门平均成绩",
    fontproperties=CN_FONT
)

plt.xlabel(
    "部门",
    fontproperties=CN_FONT
)

plt.ylabel(
    "平均成绩",
    fontproperties=CN_FONT
)

report = f"各部门平均成绩统计完成，共{{len(avg_score)}}个部门。其中最高为{{avg_score.idxmax()}}，平均分为{{avg_score.max():.1f}}。"
result = avg_score
"""

        with st.spinner("Cloudflare分析中..."):

            response = client.chat.completions.create(

                model="@cf/zai-org/glm-5.2",

                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]
            )

            code = response.choices[0].message.content

            # 清洗可能的markdown代码块标记
            code = code.strip()
            if code.startswith("```"):
                code = code.split("\n", 1)[1] if "\n" in code else code[3:]
            if code.endswith("```"):
                code = code.rsplit("\n", 1)[0] if "\n" in code else ""
            code = code.strip()

            output = io.StringIO()

            local_vars = {
                "df":df,
                "pd":pd,
                "plt":plt,
                "CN_FONT": CN_FONT
            }

            try:

                with contextlib.redirect_stdout(output):

                    exec(code,{},local_vars)

                if "result" in local_vars:

                    st.subheader("分析结果")

                    result = local_vars["result"]

                    if isinstance(result,pd.DataFrame):

                        st.dataframe(result)

                    else:

                        st.write(result)


                if "report" in local_vars:
                    st.subheader("分析报告")
                    st.markdown(local_vars["report"])
                fig = plt.gcf()

                if len(fig.axes)>0:

                    st.pyplot(fig)

                    plt.clf()

                if output.getvalue():

                    st.text(output.getvalue())

            except Exception as e:

                st.error(e)
