/* （三） 有一个“学生选课”数据库， 数据库中包括三个表， 其关
系模式分别为：
Student(xh,xm,xb,nl,szx)
Course(kch,kcm,xxk,xf)
SG(xh,kch,cj)
其中： Student 是学生表， xh 表示学号、 xm 表示姓名、 sb 表示 性别、 nl表示年龄、 szx 表示所在系。 xh 为主码。
Course 是课程表， kch 表示课程号、 kcm 表示课程名、 xxk 表示 先修课号、 xf 表示学分。 kch 为主码。
SG 是选课表， xh 表示学号， 参照学生表的学号 xh、kch 表示课 程号，参照课程表的课程号kch、cj表示成绩。 (xh,kch)为主码。
请用SQL 语言实现下列功能： */
-- 1.建立选课表 SG， 要求实现主键约束和外键约束， 且成绩不能为空。
create table SG(
    xh varchar(10) not null,
    kch varchar(10) not null,
    cj smallint not null,
    constraint PK_SG primary key (xh, kch),
    constraint FK_SG_Student foreign key (xh) references Student(xh),
    constraint FK_SG_Course  foreign key (kch) references Course(kch)
) engine=InnoDB;
 
-- 2.查询选修课程的成绩小于 60 分的人数。
select count(*) as 成绩小于60分人数 
from SG
where cj < 60;

-- 3.查询选修了 C3 号课程的学生的学号及其成绩，查询结果按分数的
-- 降序排列。要求查询结果的标题显示为汉字。
select xh as 学号,cj as 成绩 
from SG 
where kch = 'C3'
order by cj desc
;
 
-- 4.查询选修了“数据库应用”课程且成绩在 90 分以上的学生的姓名 和所在系。
select s.xm, s.szx
from Student s
join SG sg on s.xh = sg.xh
join Course c on sg.kch = c.kch
where sg.cj > 90
    and c.kcm = '数据库应用';

-- 6.将计算机系全体学生的成绩置零。
update SG set cj = 0 
where xh in (
    select xh from Student
    where szx = '计算机'
);

-- 7.创建一个“学生成绩”视图， 包括选修了课程的学生的学号、姓名、选修课程的课程号、课程名以及成绩。
create view `学生成绩` as
select s.xh, s.xm, c.kch, c.kcm, sg.cj
from Student s
join SG sg on s.xh = sg.xh
join Course c on sg.kch = c.kch;

-- 8.为Student表建立一个按学号升序排列的唯一索引 Stusno_IDX。
create unique index Stusno_IDX on Student(xh);


-- 9.求各课程的选修人数及平均成绩。
select c.kcm as 课程名, count(*) as 选修人数, avg(sg.cj) as 平均成绩
from SG sg
join Course c on sg.kch = c.kch
group by c.kch, c.kcm;


-- 10.查询选修了课程编号为’14001’和’14002’课程的学生的学号和姓名。
select s.xh, s.xm
from Student s
join (
    select xh
    from SG
    where kch in ('14001', '14002')
    group by xh
    having count(distinct kch) = 2
) t on s.xh = t.xh;

-- 另一种写法：查询同时选修了 '14001' 和 '14002' 的学生学号与姓名
select s.xh, s.xm
from Student s
join SG sg on s.xh = sg.xh
where sg.kch in ('14001', '14002')
group by s.xh, s.xm
having count(distinct sg.kch) = 2;
