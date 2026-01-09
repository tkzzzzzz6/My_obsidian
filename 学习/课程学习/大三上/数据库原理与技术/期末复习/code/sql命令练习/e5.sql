/* （五） 已知研究生管理数据库 YJSGL 包含graduate （研究生信息）
数据表和teacher （导师信息）数据表，表结构如表 1 和表 2 所示： 
表 1 graduate （研究生信息表结构）
字段名	字段类型	字段宽度	说明
bh	char	4	研究生编号(主码)
xm	char	8	姓名
xb	char	2	性别
mz	char	20	民族
ly	char	20	来源地区
cj	int		入学成绩
dsbh	char	4	导师编号(外码)

表 2 teacher （导师信息表结构）
字段名	字段类型	字段宽度	说明
dsbh	char	4	导师编号(主码)
dsxm	char	8	姓名
zc	char	10	职称
dh	char	11	联系电话

请用SQL 语句完成以下操作： */
-- 1 ．查询每个研究生的编号、姓名、性别、民族、入学成绩、 来源地区和所选导师编号。
select bh, xm, xb, mz, cj, ly, dsbh
from graduate;

-- 2 ．查询学号为 1001 的学生的姓名和入学成绩。
select xm,cj from graduate where bh = '1001';

-- 3．查询所有姓“王”的学生的编号和来源地区。
select bh, ly
from graduate
where xm like '王%';

-- 4．查询所有入学成绩在 350 和 400 分之间的学生的编号、姓名和 所选导师的姓名及其职称。
select g.bh, g.xm, t.dsxm, t.zc
from graduate g
join teacher t on g.dsbh = t.dsbh
where g.cj between 350 and 400;

-- 6．查询入学成绩低于平均入学成绩的研究生的编号、姓名、民族。
select bh,xm,mz from graduate 
where cj < (
    select avg(cj) from graduate
);
 
-- 7．查询不同来源地区的研究生人数。
select ly,count(*) as 人数
from graduate
group by ly;

-- 9．将少数民族的研究生的入学成绩加 10 分。
update graduate set cj = cj + 10
where mz <> '汉族';

-- 10．创建一个名为 rxcj （入学成绩） 的视图， 
-- 要求使用该视图能够查询入学成绩超过平均入学成绩的研究生的编号、姓名、入学成绩和所选导师的姓名及联系电话。
create view rxcj as
select g.bh, g.xm, g.cj, t.dsxm, t.dh
from graduate g
join teacher t on g.dsbh = t.dsbh
where g.cj > (
    select avg(cj) from graduate
);

