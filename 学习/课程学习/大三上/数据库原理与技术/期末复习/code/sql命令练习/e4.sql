/* （四） 设有如下所示的三个关系模式：商店 Shop （bh ，mc ，cs），商品 Product （sph，
spm，jg），商店所售商品 SP（bh，sph，sl），其中带下划线的字段为主键。
各属性含义如下： 
bh (商店编号)、mc (商店名)、cs (所在城市)、sph (商品编号)、spm(商品名称)、jg (价格)、sl (商品数量)。
ShopSP
bh	mc	Cs
101	百货商店	长沙
204	长安商场	北京
256	西单商场	北京
345	铁道商店	长沙
620	太平洋百货	上海

Product
sph	spm	jg
1	钢笔	21
2	羽毛球	5
3	复读机	300
4	书包	76
试用SQL 语言完成下列操作： */
/* （1）用 Create 语句创建商店表
bh	sph	sl
101	1	105
101	2	42
101	3	25
101	4	104
204	3	61
256	1	241
256	2	91
345	1	141
345	2	18
345	4	74
620	4	125
Shop， 要求
创建主键， 商店名不允许为空， 各属性的数据类型根据表中所给数据 选定。 */
-- 说明：题干给出的 (bh,sph,sl) 实际对应“商店所售商品”关系 SP。
create table Shop(
    bh varchar(20) not null,
    sph varchar(20) not null,
    sl int not null,
    constraint PK_Shop primary key (bh, sph),
    constraint FK_Shop_ShopSP foreign key (bh) references ShopSP(bh),
    constraint FK_Shop_Product foreign key (sph) references Product(sph)
) engine=InnoDB;

-- （2） 检索所有商店的商店名和所在城市。
select distinct mc,Cs from ShopSP;

-- （3） 检索价格低于 50 元的所有商品的商品名和价格。
select spm,jg from Product where jg < 50;

-- （4）检索位于“北京”的商店的商店编号， 商店名，结果按照 商店编号降序排列。
select bh,mc from ShopSP where Cs = '北京' order by bh desc;

-- （5） 检索供应“书包”的商店名称。
select distinct ss.mc
from ShopSP ss
join Shop sp on ss.bh = sp.bh
join Product p on sp.sph = p.sph
where p.spm = '书包';

-- （7） 将商品“复读机”的价格修改为 350。
update Product set jg = 350 where spm ='复读机';

-- （8） 将“百货商店”的商店名修改为“百货商场”。
update ShopSP set mc = '百货商场' where mc = '百货商店';

-- （9） 创建视图：“铁道商店”所售商品的商品编号， 商品名和数 量。
create view `铁道商店` as
select sp.bh, sp.sph, p.spm, sp.sl
from Shop sp
join ShopSP ss on sp.bh = ss.bh
join Product p on sp.sph = p.sph
where ss.mc = '铁道商店';

-- （10）将查询和更新 SP 表的权限赋给用户 U1。
grant select,update on table SP to 'U1'@'%';