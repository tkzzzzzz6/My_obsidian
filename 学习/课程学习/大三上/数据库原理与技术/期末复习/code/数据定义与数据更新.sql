-- 1. 数据定义

-- 创建名为 test 的模式（schema）
create schema test authoritative tan;

-- 创建表 tab1
create table tab1(
    col1 smallint,           -- 小整数，通常占 2 字节
    col2 int,                -- 整数，通常占 4 字节
    col3 char(20),           -- 固定长度字符串，20 个字符，不足补空格
    col4 varchar(30),        -- 可变长度字符串，最多 30 个字符
    col5 decimal(10,2),      -- 精确十进制数，总共 10 位，小数点后 2 位
    col6 numeric(10,3)       -- 精确十进制数，总共 10 位，小数点后 3 位
);

-- 模式的删除
drop schema test cascade;

-- 定义基本表
create table student(
    sno char(9) primary key,
    sname char(20) unique,
    ssex char(2),
    sage smallint,
    sdept char(20)
);

create table course(
    cno char(5) primary key,
    cname char(20),
    cpno char(4),
    ccredit smallint,
    foreign key(ccpno) references course(cno)
);

create table S(
    Sno char(9) primary key,
    Sname char(20),
    Status char(2),
    City char(20)
);

-- 修改基本表
alter table student
add colume 入学时间 date;

alter table student
alter column sage int;

alter table course
alter column cname varchar(30) unique;

alter table course
add unique(cname);

-- 删除基本表
drop table student cascade;

-- 2. 数据更新

