# ARCHITECTURE.md

# AI Excel 数据分析平台（app_hr）架构设计

Version: 1.0

## 一、项目目标

构建一个完全本地运行、可私有部署的 AI 数据分析平台。

核心能力：

-   上传 Excel（后续支持 CSV、Parquet、SQLite 等）
-   在本地通过SQLite数据库保存中间数据
-   使用自然语言进行数据分析
-   LLM 负责理解需求并生成 Pandas 分析代码
-   Python 本地执行分析代码
-   Streamlit 展示结果
-   AI 对分析结果进行解释
-   支持 OpenAI Compatible API（OpenAI、Qwen、GLM、DeepSeek、vLLM 等）

------------------------------------------------------------------------

## 二、核心设计原则

系统遵循三条基本原则：

1.  LLM 负责思考，不负责计算。
2.  Python 负责计算，不负责理解需求。
3.  Streamlit 负责展示，不负责业务逻辑。

任何新增功能不得违反上述原则。

------------------------------------------------------------------------

## 三、总体架构

    用户
     │
     ▼
    Streamlit UI
     │
     ▼
    Excel Loader
     │
     ▼
    SQLite
     │
     ├── DATA Profile（Schema、统计信息、样例）
     │
     ▼
    Prompt Builder
     │
     ▼
    LLM(OpenAI Compatible API)
     │
     ▼
    Python Code
     │
     ▼
    Safe Executor
     │
     ├── SQL/DataFrame
     ├── Plotly Figure
     ├── Matplotlib Figure
     ├── Summary
     │
     ▼
    Streamlit 展示

------------------------------------------------------------------------

## 四、职责划分

### LLM

负责：

-   理解自然语言
-   理解 DataFrame Schema
-   生成 Python 分析代码
-   对分析结果进行解释

禁止：

-   自行计算统计值
-   保存完整数据
-   网络访问
-   文件读写
-   系统调用

### Python

负责：

-   Excel读取
-   数据清洗
-   SQLite操作
-   Pandas分析
-   图表生成
-   导出Excel/PDF
-   所有数学计算

------------------------------------------------------------------------

## 五、数据流

1.  用户上传 Excel。
2.  使用 pandas 读取 DataFrame。
3.  构建 SQLite数据库。
4.  Prompt Builder 将 Profile + 用户问题发送给 LLM。
5.  LLM 返回 Python 代码。
6.  Safe Executor 在受限环境执行代码操作SQLite数据库进行数据分析。
7.  将结果交给 Streamlit 展示。
8.  将结果摘要发送给 LLM 生成解释。

------------------------------------------------------------------------

## 六、Prompt 设计原则

Prompt 中仅包含：

-   Schema
-   字段信息
-   少量样例
-   用户问题

禁止发送整个 DataFrame。

------------------------------------------------------------------------

## 七、安全执行

执行环境必须限制：

允许：

-   pandas
-   numpy
-   plotly
-   matplotlib
-   math
-   datetime

禁止：

-   os
-   subprocess
-   socket
-   requests
-   shutil
-   pathlib（写文件）
-   eval
-   exec（二次执行）

建议通过 AST 白名单进行校验。

------------------------------------------------------------------------

## 八、模块划分

    app_hr.py

    modules/
        excel_loader.py
        data_profile.py
        prompt_builder.py
        llm_client.py
        safe_executor.py
        chart_generator.py
        markdown_formatter.py
        export_manager.py

    utils/
    config/
    templates/
    fonts/

------------------------------------------------------------------------

## 九、UI 原则

-   UI 与业务逻辑完全分离。
-   页面不展示 AI 生成代码（调试模式除外）。
-   图表优先使用 Plotly。
-   所有错误均给出可读提示。

------------------------------------------------------------------------

## 十、未来规划

-   多文件分析
-   多 Sheet 联动
-   DuckDB
-   SQLite
-   RAG
-   MCP 工具
-   SQL Agent
-   知识库
-   插件机制

------------------------------------------------------------------------

## 十一、开发原则

1.  DataFrame 是唯一数据源。
2.  Prompt 尽量短。
3.  Python 负责所有计算。
4.  LLM 不直接处理全量数据。
5.  所有生成代码必须可审计。
6.  所有执行必须安全。
7.  所有模型统一使用 OpenAI Compatible API。
8.  保持模块低耦合、高内聚。
