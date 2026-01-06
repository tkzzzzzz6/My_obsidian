/* （六） 已知汽车销售数据库 QCXS 包含 QCGS （汽车公司）数据 表、 QCXX （汽车信息） 数据表和 JYJL（交易记录） 数据表， 表结构如
表 1、表 2 和表 3 所示：
表 1    QCGS （汽车公司表结构）
字段名	字段类型	字段宽度	说明
bh	CHAR	4	公司编号,主码
mc	CHAR	10	公司名称,唯一
szd	CHAR	20	所在地
表 2    QCXX （汽车信息表结构）
字段名	字段类型	字段宽度	说明
qcbh	CHAR	4	汽车编号,主码
cxmc	CHAR	10	车型名称
bh	CHAR	4	公司编号,外码
dj	INT		单价
bxq	INT		保修期,缺省值 12
表 3    JYJL （交易记录表结构）
字段名	字段类型	字段宽度	说明
jybh	CHAR	4	交易编号,主码
qcbh	CHAR	4	汽车编号,外码
xl	INT		销量
xsrq	DATE		销售日期

请用SQL 语句完成以下操作： */
-- 1．查询通用公司单价大于 20 万的汽车信息并按照单价降序排列，
-- 需要含有以下 4 列：公司名称，车型名，单价和保修期。
select q1.mc,q2.cxmc,q2.dj,q2.bxq 
from QCGS q1
join QCXX q2
on q1.bh = q2.bh
where q1.mc = '通用公司' and q2.dj > 200000 order by q2.dj desc;
 
-- 2 ．查询每个汽车公司卖出每款汽车的总销量，需给出汽车公司编号，汽车编号和销量。
select q1.bh, q2.qcbh, sum(j.xl) as 总销量
from QCGS q1
join QCXX q2 on q1.bh = q2.bh
join JYJL j on q2.qcbh = j.qcbh
group by q1.bh, q2.qcbh;

-- 3 ． 查询所有产地为“天津”的汽车公司编号和姓名。
select bh,mc from QCGS where szd = '天津';

-- 4 ．查询生产汽车种类大于 2 的汽车公司编号。
select bh from QCXX
group by bh having Count(qcbh) > 2;

-- 5 ．查询所有进行过交易的汽车编号。
select distinct qcbh 
from JYJL;

-- 6．查询单价在 10 万-20 万之间的汽车名称和单价。
select cxmc, dj
from QCXX 
where dj between 100000 and 200000;

-- 7．查询销量高于平均销量的汽车名称
-- 目标：找出“每辆车的总销量”高于“各车总销量的平均值”的车型名称
-- 1) 先把 QCXX(车型信息) 和 JYJL(交易记录) 按汽车编号关联起来
select q.cxmc
-- 从车型信息表取车型名称
from QCXX q
-- 关联交易记录表，得到该车的每笔销量记录
join JYJL j on q.qcbh = j.qcbh
-- 2) 按“汽车编号/车型名”分组，把同一辆车的多笔交易聚合到一起
group by q.qcbh, q.cxmc
-- 3) 用 HAVING 在分组后做条件过滤：比较“该车总销量”与“平均总销量”
having sum(j.xl) > (
	-- 4) 子查询：计算“各车总销量”的平均值
	select avg(t.total_xl)
	-- t 是一张临时结果表：每行是一辆车的总销量 total_xl
	from (
		-- 先按汽车编号分组，把每辆车的销量求和，得到每辆车的总销量
		select sum(xl) as total_xl
		from JYJL
		group by qcbh
	) t
);

-- 9 ．使用SQL 语句将汽车单价增加 10%。
update QCXX set dj = dj * 1.1;

-- 10．使用 SQL 语句创建一个名为 zxxl （最新销量）的视图，
-- 要求:能够使用该视图查询 2015 年销售量的车型名称、单价、销量和销售日期。
create view zxxl as
select q.cxmc,q.dj,j.xl,j.xsrq
from QCXX q
join JYJL j on q.qcbh = j.qcbh
where year(j.xsrq) = 2015;
