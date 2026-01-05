/* 
（二） 某大学的运动会比赛项目管理数据库包括如下三张表：
Student（xh,xm,xb,nl,szx）
各属性分别表示学生的（学号、姓名、性别、年龄、所在系）。
Sports（xmh,xmm,dw）
各属性分别表示（运动项目的编号、名称、项目的计分单位）。
SS（xh,xmh,cj）
各属性分别表示（学号、运动项目的编号、成绩）。 根据上述情况， 完成如下操作：
 */

-- 1. 使用 SQL 语言,创建 student 表，并定义 xh 为主键。
create table student(
    xh char(10) primary key,
    xm char(10),
    xb char(3),
    nl smallint,
    szx varchar(20),
)
-- 2. 在Student表 xh 属性列上建立名称为xh 的聚簇索引。
create cluster index xh where xh = xh;
-- 3. 使用SQL 语言 从表Student 中删除学生“张三”的记录。
delete from student where xm = '张三';
-- 4. 使用SQL 语言为 SS 表添加一条记录： 学号为“xh001”的学生
-- 参与了编号为“xm001”的运动项目， 但还没成绩。  
insert into SS values('xh001','xm001',null);
-- 5. 使用 SQL 语言，将 Student 表学号为“xh001”的学生的姓名 改为“李明”。
update student set xm = '李明' where xh = 'xh001';
-- 6. 查询“计算机”系的学生参加了哪些运动项目， 只把运动项目 名称列出，去除重复记录。
select distinct sp.xmm as 运动项目 
from Student as s,SS as ss,Sports as sp 
where s.szx = '计算机' and s.xh = ss.xh and ss.xmh = sp.xmh; 
-- 7. 查询各个系的学生的“跳高”项目比赛的平均成绩 (不要求输 出比赛项目的计分单位) 。
select szx,avg(cj) as 跳高平均成绩 from group by szx;
-- 8. 统计各个系的总成绩情况， 并根据总成绩按降序排序。
select szx,sum(cj) as 总成绩
-- 9. 建立“计算机”系所有男学生的信息视图JSJ_M_Student。
select 
-- 10.    回收用户“李明”对 Sports 表的查询权限。
revoke select from Sports to '李明';
