/* 
（一） 已知员工考勤数据库 YGKQ 包含 JBQK（职工基本情况） 数据表和 QQLX （缺勤信 息）数据表，表结构如表 1 和表 2 所示：
表 1    JBQK （职工基本情况表结构）

字段名	字段类型	字段宽度	说明
zgh	CHAR	4	职工号，主码
xm	CHAR	8	姓名
sj	DATETIME		缺勤时间
ts	INT		缺勤天数
lx	CHAR	4	缺勤类型， 外码

表 2    QQLX （缺勤类型信息表结构）

字段名	字段类型	字段宽度	说明
lx	CHAR	4	缺勤类型， 主码
mc	CHAR	8	缺勤名称
ms	VARCHAR	60	缺勤描述
 */

-- 1. 查询每个职工的职工号、姓名、缺勤时间、缺勤天数和缺勤类型信息。
select j.zgh, j.xm, j.sj, j.ts, j.lx
from JBQK as j;
-- 2．查询职工号为 001 的职工的姓名和缺勤天数。
select xm, ts from JBQK where zgh = '001';

-- 3．查询所有姓“张”的职工的职工号、缺勤天数。

select zgh,ts from JBQK where xm like '张%';
-- 4．找出所有缺勤天数在 2～3 天的职工号和缺勤名称。
select j.zgh, q.mc
from JBQK as j
left join QQLX as q on j.lx = q.lx
where j.ts between 2 and 3;

-- 5．查询缺勤名称为“病假”的职工的职工号和姓名。
select j.zgh, j.xm
from JBQK as j
left join QQLX as q on j.lx = q.lx
where q.mc = '病假';

-- 6．查询缺勤天数超过平均缺勤天数的职工的职工号和姓名。
select zgh,xm from JBQK where ts > (
    select avg(ts) from JBQK
);
-- 7．求各缺勤类别的人数。
select lx as 缺勤类别,count(*) as 人数 from JBQK group by lx;
-- 8．查询在职工基本情况表中没有出现过的缺勤类型及缺勤名称。
select lx,mc from QQLX where lx not in(select distinct lx from JBQK);
-- 9．使用 SQL 语句将“旷工”人员的缺勤天数增加一天。
update JBQK set ts = ts + 1 where lx = (select lx from QQLX where mc = '旷工');
-- 10．使用 SQL 语句创建一个名为zgqq （职工缺勤） 的视图,
-- 要求 能够使用该视图查询缺勤 2 天以上的职工的职工号、姓名、缺勤 天数和缺勤名称。
create view zgqq as
select j.zgh, j.xm, j.ts, q.mc
from JBQK as j
left join QQLX as q on j.lx = q.lx;

-- 视图用法示例：查询缺勤 2 天以上
select zgh, xm, ts, mc from zgqq where ts > 2;