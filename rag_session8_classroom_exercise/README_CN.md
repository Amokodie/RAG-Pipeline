# Session 8 课堂练习完整文件包说明

本文件包对应 **Session 8 — Retrieval-Augmented Generation (RAG)**。

## 设计原则
本练习不是让学生“把资料贴给大模型然后抄答案”，而是要求学生完成：
- 识别文档类型与权威性
- 检查检索结果是否真的支撑答案
- 发现 RAG 的失败模式
- 提出轻量级改进

## 文件结构

### 1. dataset/
- `corpus/`：10 个小型文档语料
- `corpus_catalog.csv`：语料清单、状态与权威级别
- `eval_queries.csv`：评价问题集

### 2. student/
- `Student_Assignment_Brief_EN.md`：学生任务书（英文）
- `Student_Worksheet_EN.md`：学生填写模板（英文）
- `NotebookLM_ClaudeCode_Workflow_EN.md`：工具使用指引（英文，非提示词包）

### 3. starter/
- `rag_lab_starter.ipynb`：Jupyter notebook 起步文件
- `rag_lab_starter.py`：Python 起步脚本
- `requirements.txt`：最小依赖
- `README_STARTER.md`：运行说明

### 4. teacher/
- `Teacher_Guide_CN.md`：教师讲解与组织建议
- `Quick_Grading_Rubric_CN.md`：快速评分量表
- `Reference_Answers_CN.md`：参考答案
- `gold_answers_teacher_only.csv`：教师专用标准答案表

## 推荐课堂流程（30 分钟）
- 5 分钟：NotebookLM 梳理语料结构
- 10 分钟：完成 4 个证据问答
- 10 分钟：运行 starter，检查 top-3 检索结果
- 5 分钟：写反思与改进建议

## 推荐提交物
- 学生 worksheet
- top-3 检索截图或复制结果
- 120–180 词 reflection

## 建议教师重点抽查
- 是否能识别 D02 为当前维护权威
- 是否能拒绝使用 D03 作为当前依据
- 是否能识别 D08 是误导性文档
- 是否能解释 query 术语与 source 术语不一致的问题
