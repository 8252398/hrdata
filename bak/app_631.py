# -*- coding: utf-8 -*-
"""
app_631.py — Excel数据分析助手（本地 Qwen3.6 + 迭代优化）

功能：
1. 上传一张 Excel 表格
2. 用户输入分析需求（自然语言）
3. 本地 Qwen3.6 大模型根据需求编写 Python 代码处理数据
4. 页面展示处理结果并支持下载
5. 支持迭代优化：用户根据结果修改需求，大模型重新生成代码并更新结果
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib import font_manager
import matplotlib
import requests
import io
import contextlib
import os
import base64
import traceback
import textwrap

# ── 页面配置 ──────────────────────────────────────────────
st.set_page_config(page_title="Excel 数据分析助手 (Qwen3.6)", layout="wide")

# ── 字体配置 ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_PATH = os.path.join(BASE_DIR, "fonts", "NotoSansCJK-Regular.ttc")

CN_FONT = None
if os.path.exists(FONT_PATH):
    CN_FONT = FontProperties(fname=FONT_PATH)
    font_manager.fontManager.addfont(FONT_PATH)
    matplotlib.rcParams["font.family"] = CN_FONT.get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

# ── Ollama 配置 ──────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:6b"  # qwen3.6 本地模型

# ── Session State 初始化 ─────────────────────────────────
_defaults = {
    "df": None,
    "df_filename": None,
    "df_columns": None,
    "df_head_md": None,
    "requirement": "",
    "generated_code": "",
    "result_obj": None,
    "report_text": "",
    "exec_error": "",
    "analysis_done": False,
    "attempt_history": [],  # [{requirement, code, error, timestamp}]
}
for key, default in _defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── 工具函数 ──────────────────────────────────────────────

def build_prompt(df_columns, df_head_md, requirement, previous_attempts=None):
    """构建发送给大模型的 prompt，包含数据上下文和历史尝试信息。"""
    safe_req = requirement.replace("{", "{{").replace("}", "}}")

    # 历史尝试上下文
    history_section = ""
    if previous_attempts:
        history_section = "\n\n## 历史尝试记录（请参考并改进）\n"
        for i, att in enumerate(previous_attempts[-3:], 1):  # 最多带3次历史
            history_section += f"""
### 第 {i} 次尝试
**需求**：{att["requirement"]}
**生成的代码**：
```python
{att["code"][:1500]}
```
**执行结果**：{"成功" if not att.get("error") else f"错误：{att['error'][:300]}"}
"""

    prompt = f"""你是一名专业的 Python 数据分析专家。

## 数据信息
- DataFrame 变量名：`df`
- 字段列表：{df_columns}
- 数据前5行预览：
{df_head_md}

## 用户分析需求
{safe_req}
{history_section}

## 可用变量
- `df`：包含用户上传数据的 Pandas DataFrame
- `pd`：pandas 模块
- `plt`：matplotlib.pyplot
- `CN_FONT`：中文字体 FontProperties 对象（可能为 None）

## 代码编写要求

1. 只返回 Python 代码，不要任何解释。
2. 不要输出 ```python 或 ``` 标记。
3. 不要定义函数，不要读取或保存文件。
4. 不要重新创建 DataFrame，直接使用变量 `df`。
5. 最终分析结果数据必须保存到变量 `result`。
6. 同时生成一段文字分析报告，保存到变量 `report`（字符串，支持 Markdown）。
7. 如果需要绘图使用 matplotlib，不要调用 `plt.show()`。
8. 图表中所有中文必须使用 `fontproperties=CN_FONT`（如果 CN_FONT 不为 None）。
9. 不要修改原始 df，如需处理请使用新变量。
10. 如果结果是 DataFrame、Series、数字或字符串，都放入 `result`。
11. 排序时明确指定 `ascending=True` 或 `False`。
12. 不要使用不存在的字段。

## 正确示例

用户需求：统计各部门平均成绩，并绘制柱状图。

返回代码：
```
avg_score = df.groupby("部门")["成绩"].mean()
avg_score.plot(kind="bar")
plt.title("各部门平均成绩", fontproperties=CN_FONT)
plt.xlabel("部门", fontproperties=CN_FONT)
plt.ylabel("平均成绩", fontproperties=CN_FONT)
report = f"各部门平均成绩统计完成，共{{len(avg_score)}}个部门。最高为{{avg_score.idxmax()}}，平均值{{avg_score.max():.1f}}。"
result = avg_score
```

现在请根据用户需求编写 Python 代码："""

    return prompt


def clean_code(raw_code):
    """清洗大模型返回的代码文本。"""
    code = raw_code.strip()
    # 去除 markdown 代码块标记
    if code.startswith("```"):
        lines = code.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    return code.strip()


def execute_code(code, df):
    """在沙箱中执行大模型生成的代码，返回 (local_vars, error_msg)。"""
    output = io.StringIO()
    local_vars = {
        "df": df.copy(),  # 使用副本保护原始数据
        "pd": pd,
        "plt": plt,
        "CN_FONT": CN_FONT,
    }
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {}, local_vars)
        return local_vars, "", output.getvalue()
    except Exception as e:
        return local_vars, f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}", output.getvalue()


def get_download_data(result_obj, df):
    """根据结果类型生成下载数据。返回 (bytes, filename, mime_type) 或 None。"""
    if result_obj is None:
        return None

    if isinstance(result_obj, pd.DataFrame):
        buf = io.BytesIO()
        result_obj.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue(), "分析结果.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

    if isinstance(result_obj, pd.Series):
        buf = io.BytesIO()
        result_obj.to_csv(buf, encoding="utf-8-sig")
        return buf.getvalue(), "分析结果.csv", "text/csv"

    # 数字/字符串等
    text = str(result_obj)
    return text.encode("utf-8-sig"), "分析结果.txt", "text/plain"


# ── UI 标题 ──────────────────────────────────────────────
st.title("📊 Excel 数据分析助手")
st.caption(f"本地模型：`{OLLAMA_MODEL}`  |  Ollama 地址：`{OLLAMA_URL}`")

# ── 侧边栏：模型配置 ──────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 模型配置")
    model_input = st.text_input("Ollama 模型名称", value=OLLAMA_MODEL, key="model_name")
    url_input = st.text_input("Ollama URL", value=OLLAMA_URL, key="ollama_url")
    st.divider()
    st.caption("支持迭代优化：每次分析后可在下方修改需求重新分析，大模型会参考历史尝试改进代码。")

    if st.button("🔄 重置全部状态"):
        for key in _defaults:
            st.session_state[key] = _defaults[key]
        plt.clf()
        st.rerun()

    # 检查 Ollama 连接
    if st.button("🔍 测试 Ollama 连接"):
        try:
            resp = requests.get(f"{url_input}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                st.success(f"连接成功！可用模型：{', '.join(models[:10])}")
            else:
                st.warning(f"返回状态码：{resp.status_code}")
        except Exception as e:
            st.error(f"连接失败：{e}")

# ── Step 1: 上传文件 ─────────────────────────────────────
st.markdown("### 第一步：上传 Excel 文件")
uploaded_file = st.file_uploader(
    "支持 .xlsx / .xls / .csv 格式",
    type=["xlsx", "xls", "csv"],
    key="file_uploader",
)

if uploaded_file is not None:
    # 检查是否是新文件
    if st.session_state.df_filename != uploaded_file.name:
        # 新文件，重置分析状态但保留文件数据
        for key in ["requirement", "generated_code", "result_obj", "report_text",
                     "exec_error", "analysis_done", "attempt_history"]:
            st.session_state[key] = _defaults[key]

    st.session_state.df_filename = uploaded_file.name

    # 读取文件
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.session_state.df = df
        st.session_state.df_columns = list(df.columns)
        st.session_state.df_head_md = df.head(10).to_markdown(index=False)

    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    # 数据预览
    with st.expander(f"📄 数据预览 — `{uploaded_file.name}`（{len(df)} 行 × {len(df.columns)} 列）", expanded=True):
        st.dataframe(df.head(20), use_container_width=True)
        st.caption(f"共 {len(df)} 行，{len(df.columns)} 列")

    st.divider()

    # ── Step 2: 输入分析需求 ─────────────────────────────
    st.markdown("### 第二步：输入分析需求")
    col_left, col_right = st.columns([3, 1])

    with col_left:
        requirement = st.text_area(
            "用自然语言描述你的分析需求",
            value=st.session_state.requirement,
            placeholder="例如：\n- 统计各部门人数并绘制饼图\n- 按销售额降序排列，显示前10名\n- 计算各产品类别的平均利润，找出利润最高的类别\n- 按月份汇总销售额并绘制趋势折线图",
            height=120,
            key="requirement_input",
        )
        st.session_state.requirement = requirement

    with col_right:
        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🚀 开始分析", type="primary", use_container_width=True, disabled=(not requirement.strip()))

    # ── 执行分析 ─────────────────────────────────────────
    if analyze_btn and requirement.strip():
        with st.spinner(f"🤖 {st.session_state.model_name} 正在分析中……"):
            # 构建 prompt（包含历史尝试）
            prompt = build_prompt(
                st.session_state.df_columns,
                st.session_state.df_head_md,
                requirement,
                st.session_state.attempt_history if st.session_state.analysis_done else None,
            )

            # 调用 Ollama
            try:
                resp = requests.post(
                    f"{st.session_state.ollama_url}/api/generate",
                    json={
                        "model": st.session_state.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"num_predict": 4096},
                    },
                    timeout=300,
                )
                raw_code = resp.json()["response"]
            except Exception as e:
                st.error(f"Ollama 调用失败：{e}")
                st.stop()

            # 清洗代码
            code = clean_code(raw_code)
            st.session_state.generated_code = code

            # 清除之前的图表
            plt.clf()

            # 执行代码
            local_vars, error, stdout_text = execute_code(code, st.session_state.df)

            # 记录本次尝试
            attempt_record = {
                "requirement": requirement,
                "code": code,
                "error": error,
            }
            st.session_state.attempt_history.append(attempt_record)

            # 提取结果
            if not error:
                st.session_state.result_obj = local_vars.get("result")
                st.session_state.report_text = local_vars.get("report", "")
                st.session_state.exec_error = ""
                st.session_state.analysis_done = True

                # 如果有 stdout 输出且没有 report，用 stdout 作为补充
                if stdout_text.strip() and not st.session_state.report_text:
                    st.session_state.report_text = stdout_text.strip()
            else:
                st.session_state.exec_error = error
                st.session_state.result_obj = None
                st.session_state.report_text = ""
                st.session_state.analysis_done = True

    # ── Step 3: 展示结果 ─────────────────────────────────
    if st.session_state.analysis_done:
        st.divider()
        st.markdown("### 第三步：分析结果")

        # 生成的代码展示
        with st.expander("📝 查看大模型生成的代码", expanded=False):
            st.code(st.session_state.generated_code, language="python", line_numbers=True)

        # 错误提示
        if st.session_state.exec_error:
            st.error("⚠️ 代码执行出错")
            with st.expander("错误详情", expanded=True):
                st.code(st.session_state.exec_error)

        # 结果展示
        result_obj = st.session_state.result_obj
        report = st.session_state.report_text

        if result_obj is not None:
            tab1, tab2, tab3 = st.tabs(["📊 数据结果", "📈 图表", "📋 分析报告"])

            with tab1:
                if isinstance(result_obj, pd.DataFrame):
                    st.dataframe(result_obj, use_container_width=True)
                    st.caption(f"结果：{len(result_obj)} 行 × {len(result_obj.columns)} 列")
                elif isinstance(result_obj, pd.Series):
                    st.dataframe(result_obj.reset_index(), use_container_width=True)
                else:
                    st.write(result_obj)

            with tab2:
                fig = plt.gcf()
                if len(fig.axes) > 0:
                    st.pyplot(fig)
                else:
                    st.info("本次分析未生成图表。如需图表，请在需求中说明。")

            with tab3:
                if report:
                    st.markdown(report)
                else:
                    st.info("本次分析未生成文字报告。")

            # ── 下载按钮 ─────────────────────────────────
            st.divider()
            st.markdown("#### 💾 下载结果")
            dl_data = get_download_data(result_obj, st.session_state.df)

            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                if dl_data:
                    st.download_button(
                        label=f"📥 下载 {dl_data[1]}",
                        data=dl_data[0],
                        file_name=dl_data[1],
                        mime=dl_data[2],
                        use_container_width=True,
                        key="dl_result",
                    )

            with col_dl2:
                # 额外提供完整报告下载
                full_report = f"# 数据分析报告\n\n## 需求\n{st.session_state.requirement}\n\n## 分析结果\n{str(result_obj)}\n\n## 文字报告\n{report if report else '无'}"
                st.download_button(
                    label="📥 下载完整报告 (Markdown)",
                    data=full_report.encode("utf-8-sig"),
                    file_name="分析报告.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_report",
                )

        # ── 历史尝试记录 ─────────────────────────────────
        if len(st.session_state.attempt_history) > 1:
            with st.expander(f"📜 历史尝试记录（共 {len(st.session_state.attempt_history)} 次）", expanded=False):
                for i, att in enumerate(st.session_state.attempt_history):
                    status = "❌ 失败" if att["error"] else "✅ 成功"
                    st.markdown(f"**第 {i+1} 次** {status} — 需求：{att['requirement'][:80]}")
                    if att["error"]:
                        st.caption(att["error"][:200])

        # ── Step 4: 迭代优化区域 ─────────────────────────
        st.divider()
        st.markdown("### 🔄 迭代优化")
        st.info(
            "如果结果不理想，请修改上方「第二步」中的分析需求，然后再次点击「开始分析」。\n\n"
            "大模型会参考之前的尝试记录，自动改进代码。你也可以在下方快速编辑需求："
        )

        new_req = st.text_area(
            "快速修改需求（修改后点击下方按钮）",
            value=st.session_state.requirement,
            height=100,
            key="refine_input",
            placeholder="修改你的分析需求，例如添加更多细节、调整排序方式、增加图表类型……",
        )

        col_r1, col_r2 = st.columns([1, 3])
        with col_r1:
            if st.button("🔄 重新分析（改进）", type="secondary", use_container_width=True):
                st.session_state.requirement = new_req
                st.rerun()
        with col_r2:
            st.caption("修改需求后点击按钮，系统将自动填入到上方输入框并重新分析。")

else:
    # 无文件时的引导
    st.info("👆 请先上传一个 Excel 或 CSV 文件开始分析。")
    st.markdown("""
    ### 使用说明
    1. **上传文件**：支持 `.xlsx`、`.xls`、`.csv` 格式
    2. **输入需求**：用自然语言描述你想做的数据分析
    3. **查看结果**：数据、图表、报告分标签展示
    4. **下载导出**：支持 Excel/CSV/TXT 格式下载
    5. **迭代优化**：结果不满意？修改需求后重新分析，大模型会参考历史改进

    ### 前置条件
    - 本地需运行 [Ollama](https://ollama.com) 并拉取 `qwen3:6b` 模型
    - 运行命令：`ollama pull qwen3:6b`
    """)
