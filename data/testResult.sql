WITH MEMBERS AS (
  SELECT employee_code, name, unit, department, cadre_flag
FROM persons
WHERE cadre_flag IN ('集团领导', '技术人才|型号两总|干部|集团领导')
), mid_training AS (
SELECT t.employee_code, t.course_name, t.hours, t.training_type, t.organizer, t.institution,
t.start_date, t.end_date
FROM training_records t
WHERE t.organizer IN ('中组部','中组部干教局','中组部干部教育局','中组部班次')
AND CAST(julianday(t.end_date) - julianday(t.start_date) AS INTEGER) <= 15
),
member_mid AS (
SELECT m.employee_code, m.name, m.unit, m.department, m.cadre_flag,
mt.course_name, mt.hours, mt.training_type, mt.organizer, mt.institution,
mt.start_date, mt.end_date
FROM members m
JOIN mid_training mt ON m.employee_code = mt.employee_code
),
course_cnt_3y AS (
SELECT employee_code, course_name
FROM training_records
WHERE end_date >= date('now','-3 years')
GROUP BY employee_code, course_name
HAVING COUNT(*) = 1
),
total_hours_4y AS (
SELECT employee_code
FROM training_records
WHERE end_date >= date('now','-4 years')
GROUP BY employee_code
HAVING SUM(hours) < 440
)
SELECT mm.employee_code, mm.name, mm.unit, mm.department, mm.cadre_flag,
mm.course_name, mm.hours, mm.training_type, mm.organizer, mm.institution,
mm.start_date, mm.end_date
FROM member_mid mm
JOIN course_cnt_3y c ON mm.employee_code = c.employee_code AND mm.course_name = c.course_name
JOIN total_hours_4y h ON mm.employee_code = h.employee_code
ORDER BY mm.employee_code, mm.start_date

# 17002031	程福波	航空工业集团总部	党组成员	集团领导	弘扬企业家精神，加快建设世界一流企业专题研讨班（第2期）	40	nan	中组部	中国大连高级经理学院	2025-04-22	2025-04-26
# 17907312	郭盛杰	航空工业集团总部	党组成员	集团领导	优化产业结构调整，大力推进新型工业化专题研讨班	112	nan	中组部	北京大学	2025-07-14	2025-07-27