# 📊 课程数据分析助手

基于 Streamlit + DeepSeek AI 的智能数据分析工具，用自然语言描述需求，自动生成并执行 Python 代码进行数据分析和可视化。

## 功能

- 上传 CSV / Excel 数据文件
- 用中文自然语言提问（如"统计各部门平均成绩并画柱状图"）
- AI 自动生成 pandas + matplotlib 代码并执行
- 展示分析结果表格和图表

## 快速开始

```bash
# 1. 克隆项目
git clone <your-repo-url>
cd hrdata

# 2. 创建虚拟环境并安装依赖
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.sample .env
# 编辑 .env 填入你的 DeepSeek API Key

# 4. 运行
streamlit run app.py
```

## 技术栈

- Streamlit — Web 界面
- Pandas — 数据处理
- Matplotlib — 数据可视化
- DeepSeek API — AI 代码生成
- Noto Sans CJK — 中文字体
