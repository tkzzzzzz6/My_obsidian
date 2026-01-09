/* （七） 设“职工_社团”数据库有 3 个基本表：
职工： zg (zgh，xm，nl，xb，gz)；
社会团体： shtt(bh，nc，fzr，dd)；
参加： cj(zgh，bh，rq)。
其中：
（ 1） 职工表 zg 的主码为 zgh(职工号)。
xm 为姓名， nl 为年龄， xb 为性别， gz 为工资。
（2）社会团体表 shtt 的主码为bh（编号）， 
fzr （负责人） 为外码， 参照职工表 zg 的 zgh。nc 为名称， fzr 为负责人， dd 为活动地点
（3）参加表 cj 的职工号 zgh 和编号 bh 为主码； 
zgh 为外码， 参照职工表 zg 的 zgh； 编号 bh 为外码， 参照社会团体表 shtt 的bh。 rq 为参加日期。
使用SQL 语句完成下列操作： */

-- 1.定义参加表， 在语句中要求定义表中的主码和外码约束； 
-- (说明：表中属性的类型根据实际情况定义。)
create table zg(
    zgh varchar(10) not null,
    xm varchar(10),
    nl smallint,
    xb varchar(5),
    gz int,
    constraint PK_zg primary key (zgh)
) engine=InnoDB;

create table shtt(
    bh varchar(10) not null,
    nc varchar(20) not null,
    fzr varchar(10),
    dd varchar(20),
    constraint PK_shtt primary key (bh),
    constraint FK_shtt_zg foreign key (fzr) references zg(zgh)
) engine=InnoDB;

create table cj(
    zgh varchar(10) not null,
    bh varchar(10) not null,
    rq date,
    constraint PK_cj primary key (zgh, bh),
    constraint FK_cj_zg foreign key (zgh) references zg(zgh),
    constraint FK_cj_shtt foreign key (bh) references shtt(bh)
) engine=InnoDB;

-- 2.查询每个社会团体的参加人数；
select s.bh, s.nc as 社会团体, count(*) as 参加人数
from shtt as s
join cj as c on s.bh = c.bh
group by s.bh, s.nc;

-- 3.检索所有比“王华”年龄大的职工的姓名、年龄和性别；
select xm,nl,xb from zg 
where nl > (
    select nl from zg where xm = '王华'
);

-- 4.查找参加了歌唱队或篮球队的职工号和姓名；
select distinct z.zgh,z.xm
from zg z
join cj c on z.zgh = c.zgh
join shtt s on c.bh = s.bh
where s.nc in ('歌唱队','篮球队');

-- 5.查找没有参加任何社会团体的职工信息；
select z.*
from zg z
where not exists (
    select 1
    from cj c
    where c.zgh = z.zgh
);

-- 6.将所有参加编号为“10001”的社会团体的职工的工资增加 10%；
update zg z
set z.gz = z.gz * 1.1
where exists (
    select 1
    from cj c
    where c.zgh = z.zgh and c.bh = '10001'
);

-- 7.查询年龄最大的职工的职工号和姓名；
select zgh,xm from zg order by nl desc limit 1;

-- 8.查询各社会团体的编号以及其负责人的姓名；
select s.bh, z.xm as 负责人姓名
from shtt s
join zg z on s.fzr = z.zgh;

-- 9.删除职工号为’402’的职工参加所有社会团体的记录；
delete from cj
where zgh = '402';

-- 10.以职工姓名为参数建立一个带参数的存储过程，用于查询其所参加的社会团体的编号和名称， 
-- 并调用此存储过程查询“王明”所 参加的社会团体的编号和名称。
delimiter //
create procedure query_shtt_bh_nc(in p_xm varchar(10))
begin
    select s.bh, s.nc
    from shtt s
    join cj c on s.bh = c.bh
    join zg z on c.zgh = z.zgh
    where z.xm = p_xm;
end //
delimiter ;

call query_shtt_bh_nc('王明');

delimiter //
create procedure query_shtt_bh_nc(in p_xm varchar(10))
begin
    select shtt s
    from cj c on s.bh = c.bh
    join zg z on c.zgh = z.zgh
    where z.xm = p_xm;
end //
delimiter;

call qurry_shtt_bh_nc('王明');