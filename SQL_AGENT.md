# HRData SQL AGENT：SQLite + LLM SQL Agent

## 目标

将当前的LLM自然语言分析重构为"数据库自发现（Schema Discovery）"：

AI收到自然语言检索要求 -\> AI探索数据库 -\> AI查看探索结果 -\> AI根据探索结果推理最适合检索要求的SQL代码生成要求 -\> AI生成SQL代码在数据库中进行检索 -\> AI反馈检索结果
图表/报告


------------------------------------------------------------------------

## 探索数据库

使用类似如下的SQL代码对数据库进行探索查询，使AI在生成真正的SQL检索代码时并不是"瞎猜"，而是在看过数据库实际内容之后再推理。

SELECT DISTINCT 职位；
SELECT DISTINCT 部门;
SELECT DISTINCT 课程名称;

------------------------------------------------------------------------

## Agent设计

不要直接 Text2SQL。

采用 Agent 模式：

用户 
↓ 
LLM 
↓ 
生成探索数据库SQL查看数据库Schema 
↓ 
SQLite执行 
↓ 
结果返回LLM
↓ 
LLM继续分析，推理业务概念
↓ 
生成最终检索SQL
↓ 
最终分析结果

允许多轮。

------------------------------------------------------------------------

## 探索性SQL

Prompt要求：

当无法确定：

-   字段
-   职位
-   部门
-   培训类别
-   课程名称

不得猜测。

优先探索数据库。

例如：

SELECT DISTINCT position FROM employee;

SELECT DISTINCT department FROM employee;

SELECT DISTINCT training_type FROM training;

PRAGMA table_info(employee);

PRAGMA table_info(training);

允许执行多次探索SQL。

------------------------------------------------------------------------

## Python职责

Python不要包含业务判断。

Python只负责：

1.  接收LLM输出SQL
2.  执行只读SQL
3.  返回结果给LLM
4.  持续循环直到LLM声明最终SQL

不要在Python中维护：

-   班子成员定义
-   中层干部定义
-   数字化课程定义
-   各类业务映射

业务理解全部交给LLM。

------------------------------------------------------------------------

## 安全限制

仅允许：

SELECT WITH PRAGMA table_info

禁止：

INSERT UPDATE DELETE DROP ALTER ATTACH

所有SQL必须只读。

------------------------------------------------------------------------

## Prompt原则

指导模型遵循：

"先观察数据库，再推理，再查询。"

遇到模糊概念（例如班子成员、技术人员、管理人员、年轻干部等）：

不要直接猜测。

应优先查看数据库真实数据，再依据数据库内容进行推理。

允许理解存在一定偏差，最终结果由人工审核。

因此模型应尽可能扩大召回率，而不是追求绝对精确。

例如可以这样设计系统 Prompt：

你是一名 SQLite 数据分析助手。

你的目标不是立即生成 SQL。

如果遇到以下情况：

- 不确定字段名称
- 不确定职位有哪些
- 不确定部门有哪些
- 不确定课程类别有哪些

请先执行探索性 SQL。

例如：

SELECT DISTINCT position FROM employee;

SELECT DISTINCT department FROM employee;

SELECT DISTINCT course_type FROM training;

确认数据库中的真实值以后，再生成最终 SQL。

可以多次查询数据库。

最终只输出最终 SQL。

------------------------------------------------------------------------

## 推荐实现

增加 DatabaseAgent：

-   execute_sql(sql)
-   validate_sql(sql)
-   schema()
-   run_agent()

其中 run_agent() 负责：

LLM -\> SQL -\> SQLite -\> Result -\> LLM

直到完成。

整个业务规则尽量保留在 Prompt，而不是 Python 代码中。

---------------------------------------------------------------------------
## 具体示例

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

---------------------------------------------------------------------------
## 重构要求

- 不改变 Streamlit UI
- 不改变导入 Excel 的方式
- 保留现有结果展示
- 保留 AI 总结功能
- 使用 SQLite 数据层
- 新增 SQL Agent
- 删除原有 DataFrame 自然语言分析代码