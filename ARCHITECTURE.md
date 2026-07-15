# ARCHITECTURE.md

# AI Excel 数据分析平台（app_hr）架构设计

Version: 1.0

## 一、项目目标

构建一个完全本地运行、可私有部署的 AI 数据分析平台。

核心能力：

-   上传 Excel（后续支持 CSV、Parquet、SQLite 等）
-   在本地通过SQLite数据库保存中间数据
-   使用自然语言进行数据分析
-   LLM 负责理解需求并生成  SQL 分析代码
-   Python 本地执行分析代码
-   Streamlit 展示结果
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
-   理解数据库Schema
-   生成 Python 分析代码
-   对分析结果进行解释

禁止：

-   保存完整数据
-   网络访问
-   文件读写
-   系统调用

### Python

负责：

-   Excel读取
-   数据清洗
-   SQLite操作
-   图表生成
-   导出Excel/PDF
-   所有数学计算

------------------------------------------------------------------------

## 五、数据流

1.  用户上传 Excel。
2.  构建 SQLite数据库。
3.  Prompt Builder 将 Profile + 用户问题发送给 LLM。
4.  LLM 探索数据库，根据自身理解生成 Python 代码。
5.  Safe Executor 在受限环境执行代码操作SQLite数据库进行数据分析。
6.  将结果交给 Streamlit 展示。

例如，用户输入：
统计集团班子成员近三年的培训情况

不要直接要求：
请生成 SQL。

而是给模型一个工具：
run_sql(sql: str) -> DataFrame

并告诉它：
你可以多次调用 run_sql()，先了解数据库，再完成最终查询。

AI第一步：查看数据库结构
PRAGMA table_info(employee);
得到：
employee_id
name
department
position
...

AI第二步：看看职位有哪些
SELECT DISTINCT position
FROM employee
ORDER BY position;
返回：
董事长
总经理
副总经理
部长
副部长
工程师
高级工程师
……
AI此时已经知道企业到底有哪些职位，而不是凭空猜。

AI第三步：思考
AI在内部推理：
"班子成员"一般应包括：
董事长
总经理
副总经理
数据库里没有"党委书记"这个职位，因此不应加入。

AI第四步：生成最终的检索SQL
SELECT
    e.name,
    SUM(t.hours)
FROM employee e
JOIN training t
ON e.employee_id=t.employee_id
WHERE e.position IN (
'董事长',
'总经理',
'副总经理'
)
AND t.date>='2023-01-01'
GROUP BY e.employee_id;
执行并反馈用户真正的检索结果
------------------------------------------------------------------------

## 六、Prompt 设计原则

Prompt 中仅包含：

-   Schema
-   字段信息
-   SQL样例
-   用户问题

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
-   所有错误均给出可读提示。

------------------------------------------------------------------------

## 十、开发原则

1.  SQLite 是唯一数据源。
2.  Prompt 尽量短。
3.  Python 负责所有计算。
4.  LLM 不直接处理全量数据。
5.  所有生成代码必须可审计。
6.  所有执行必须安全。
7.  所有模型统一使用 OpenAI Compatible API。
8.  保持模块低耦合、高内聚。
