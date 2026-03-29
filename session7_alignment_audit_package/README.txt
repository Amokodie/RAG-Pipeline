Session 7 课堂练习文件包说明
================================

文件包主题
-----------
Mini Alignment Audit（30分钟课堂练习）

设计定位
-----------
本练习对应 Session 7: Post-Training & Alignment。
核心目标不是让学生把 PPT 或材料直接喂给 NotebookLM / Claude Code 出现成答案，
而是让学生先理解 alignment 相关概念，再判断模型输出是否真正符合课程内容。

建议使用方式
-----------
1. 教师先发给学生：
   - data/session7_alignment_audit_dataset.csv
   - data/session7_student_scoring_template.csv
   - notebook/session7_alignment_audit_starter_notebook.ipynb
   - docs/session7_student_assignment_brief.docx

2. 教师自己保留：
   - data/session7_teacher_answer_key.csv
   - data/session7_teacher_quick_grade_sheet.csv
   - docs/session7_teacher_guide_and_rubric.docx

3. 推荐课堂时间分配：
   - 0–5 min：学生先自己写出 helpful / safe / honest / fair 的定义
   - 5–10 min：用 NotebookLM 查清概念边界
   - 10–22 min：完成案例评分、理由和修正答案
   - 22–27 min：用 Claude Code 汇总自己的结果
   - 27–30 min：形成一段部署建议

练习产出
-----------
学生最终应提交：
- 完整评分表
- 每个案例一句判断理由
- 对弱回答的修正版本
- 简短的部署建议

快速判分建议
-----------
优先看三项：
1. 理由是否贴合课程概念
2. 是否能区分 harmful compliance / over-refusal / false certainty / bias
3. 修正答案是否比原答案更 aligned

