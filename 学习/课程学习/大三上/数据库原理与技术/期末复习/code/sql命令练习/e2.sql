/* 
（二） 某大学的运动会比赛项目管理数据库包括如下三张表：
Student（xh,xm,xb,nl,szx）
各属性分别表示学生的（学号、姓名、性别、年龄、所在系）。
Sports（xmh,xmm,dw）
各属性分别表示（运动项目的编号、名称、项目的计分单位）。
SS（xh,xmh,cj）
各属性分别表示（学号、运动项目的编号、成绩）。 根据上述情况， 完成如下操作：
 */

/*
说明（MySQL）：
1) InnoDB 下主键就是“聚簇”组织方式；MySQL 没有 SQL Server 那种 CREATE CLUSTERED INDEX 语法。
2) MySQL 的 REVOKE 需要写成 `REVOKE ... ON db.table FROM 'user'@'host'`（且要与之前 GRANT 的对象匹配）。
3) 若 Sports/SS 表已在别处创建，可删除本文件中对应建表语句，仅保留题目 1-10 的操作。
*/

-- 1. 使用 SQL 语言,创建 student 表，并定义 xh 为主键。
create table Student (
    xh  char(10) primary key,
    xm  varchar(20) not null,
    xb  varchar(10) null,
    nl  smallint null,
    szx varchar(20) null
);

-- （可选）创建 Sports、SS（与题目给出的关系模式一致）
create table Sports (
    xmh char(10) primary key,
    xmm varchar(50) not null,
    dw  varchar(20)
) engine=InnoDB;

create table SS (
    xh  char(10) not null,
    xmh char(10) not null,
    cj  decimal(10, 2) null,
    primary key (xh, xmh),
    constraint FK_SS_Student foreign key (xh) references Student(xh),
    constraint FK_SS_Sports  foreign key (xmh) references Sports(xmh)
) engine=InnoDB;
-- 2. 在Student表 xh 属性列上建立名称为xh 的聚簇索引。
-- MySQL(InnoDB)：主键（xh）天然就是聚簇组织方式，因此题目第 2 条在 MySQL 中可视为已满足。
-- 如果老师要求“必须给出一条建聚簇索引的 SQL 语句”，MySQL 下等价表达通常写成“把 xh 设为主键”：
-- （仅当你在第 1 题建表时没有写 PRIMARY KEY 时才需要执行）
-- alter table Student add primary key (xh);
-- 如果老师要求“必须写建索引语句”，通常写普通二级索引即可（但对主键列来说是冗余的）：
-- create index idx_student_xh on Student(xh);
-- 3. 使用SQL 语言 从表Student 中删除学生“张三”的记录。
delete from Student where xm = '张三';
-- 4. 使用SQL 语言为 SS 表添加一条记录： 学号为“xh001”的学生
-- 参与了编号为“xm001”的运动项目， 但还没成绩。  
insert into SS(xh, xmh, cj) values ('xh001', 'xm001', null);
-- 5. 使用 SQL 语言，将 Student 表学号为“xh001”的学生的姓名 改为“李明”。
update Student set xm = '李明' where xh = 'xh001';
-- 6. 查询“计算机”系的学生参加了哪些运动项目， 只把运动项目 名称列出，去除重复记录。
select distinct sp.xmm as 运动项目
from Student s
join SS ss on s.xh = ss.xh
join Sports sp on ss.xmh = sp.xmh
where s.szx = '计算机';
-- 7. 查询各个系的学生的“跳高”项目比赛的平均成绩 (不要求输 出比赛项目的计分单位) 。
select s.szx, avg(ss.cj) as 跳高平均成绩
from Student s
join SS ss on s.xh = ss.xh
join Sports sp on ss.xmh = sp.xmh
where sp.xmm = '跳高'
group by s.szx;
-- 8. 统计各个系的总成绩情况， 并根据总成绩按降序排序。
select s.szx, sum(ss.cj) as 总成绩
from Student s
left join SS ss on s.xh = ss.xh
group by s.szx
order by 总成绩 desc;
-- 9. 建立“计算机”系所有男学生的信息视图JSJ_M_Student。
create view JSJ_M_Student
as
select *
from Student
where szx = '计算机' and xb = '男';
-- 10.    回收用户“李明”对 Sports 表的查询权限。
-- MySQL：注意这里的 db_name 需要替换成你实际使用的数据库名，并且要与 GRANT 的对象保持一致。
revoke select on db_name.Sports from '李明'@'%';
