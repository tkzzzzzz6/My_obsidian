show databases;

create database college;

use college;

show tables;

drop database college;

create table mytable
(
    col1 char(10),
    col2 int,
    col3 varchar(20),
    col4 float
);

use college;

create table student
(
    primary key (id) int not null unique,
    name varchar(20),
    age int,
    sex char(2),
    dept varchar(20)
    constraint chk_age check(age >= 18 and age <= 60),
    cno char(5) references class(cn0) 
);

insert into student values('1001', 'Alice', 20, 'F', 'CS', 'C01');

update student set age = 17 where id = '1001';

delete from student where id = '1001';

delete from student;

select name,age from student;

select distinct dept from student;

select cno as class_number, name as student_name from student;

-- query with expression
select sname as 姓名,2017 - age as 出生日期 from student;

-- and,or,not,between and,in,like,is null
select sno,cno,grade from sc where grade between 60 and 100;

select sno,cno,grade from sc where grade > 85 and (cno = 'kc001' or cno = 'kc002');

select tno,tname,prof from teacher where salary between 1000 and 1500;

select sno,cno,grade from sc where cno in ('c1','c2');

select cno from class where cname like '%数%';

select cno from course where cpno is null;

-- static functions
-- count,sum,avg,max,min
select avg(age) as 平均年龄 from teacher where dept = 'cs' and sex = 'F';

select count(*) as 人数 from teacher where prof = 'professor';

select count(*) from student where dept = 'cs';

select sno,count(*) as 所选课数 from sc group by sno;

select dept,count(*) as 人数 from student group by dept;

select sno,count(*) as 选课门数 from sc group by sno having count(*) >= 4;

select sno,sname,age from student order by age desc;

select teacher.tno,tn,cno from teacher,tc
where (teacher.tno = tc.tno) and tn = 'Smith';

select student.sname,sage,sc.cno from student,sc,course
where (student.sno = sc.sno and sc.cno = course.cno) and sage > 20 and sex = 'M';

select t1.tname,t1.sex,t1.age,t2.age from teacher as t1,teacher as t2
where t2.name = 'Johnson' and t1.age > t2.age;

select * from table1 cross join table2;

select student.sname,student.age,sc.cno from student inner join sc
on student.sno = sc.sno inner join course on sc.cno = course.cno
where student.age > 20 and sex = 'M';

select sc.min(grade) as min_grade from student inner join sc 
on student.sno = sc.sno
group by student.sno
order by student.sno desc;

select student.sname,sc.cno from student left outer join sc 
on student.sno = sc.sno;

select course.cname,sc.grade from course right outer join sc
on cousre.cno = sc.cno;

select teacher.tname,tc.cno from teacher 
where teacher.prof = (
    select prof from teacher where tname = 'smith'
);

select sno,sname,sage from student
where age > (
    select avg(age) from student 
    where dept = 'cs'
);

select tname from teacher
where sno = any(
    select tno from tc where cno = 'c001'
);

select tname from teacher
where salary in(
    select tno from tc where cno = 'c001'
);

select tname,tslary from teacher
where salary > all(
    select salary from teacher
);

select tname from teacher
where exists(
    select tname from tc where teacher.tno = tc.tno and cno = 'c001'
);

select sname from student
where no exists(
    select * from course
    where not exists(
        select * from sc
        where course.cno = sc.cno and sc.sno = student.sno
    )
);



