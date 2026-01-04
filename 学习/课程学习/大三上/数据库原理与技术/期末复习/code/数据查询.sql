-- 单表查询
select sname,sno,sdept from student;

select * from student;

select sname as 姓名, sno as 学号 from student;

select distinct sdept from student;

select distinct sno from sc where grade < 60;

select sname,sdept,sage from student
where sage not between 20 and 30;

select sname,ssex from student
where sdept in ('cs','ma','is')

select sname,sno from student 
where sname like '_阳%';

select cno,ccredit from course
where cname like 'DB_design' escape '';

select sno,cno from student left join sc
on student.sno = sc.sno
where sc.grade is null;

select sname,ssex from student
where sdept = 'cs' or sdept = 'is' or sdept = 'ma';

select * from student order by sdept asc,sage desc;

select sum(credit) as 总学分数 from sc left join course
on sc.cno = course.cno where sc.sno = '1001';

select sno from sc group by sno having count(*) >3;

select 学号,姓名,专业 from 学生 where 学号 not in(
    select 学号 from 学习 where 成绩 < 90
);

select 学号,姓名,专业 from 学生 where 学号 exists(
    select 学号 from 学习 where 成绩 > 90
) and 奖学金 = 0;

select 学号,姓名,专业 from 学生 as S
where(s.奖学金 = 0 or s.奖学金 is null) 
and exists(
    select 学号 from 学习 as x
    where x.学号 = s.学号 and x.成绩 > 90
);

select 学号,姓名,专业 from 学生 left join 学习
on 学生.学号 = 学习.学号
where 学习.成绩 > 90 and 学生.奖学金 = 0;

-- 连接查询
select from student.*,sc.* from student inner join sc
on student.sno = sc.sno;

select c2.name,c2.cno,c1.name as 间接先修课 from course as c1 join course as c2
on c1.cno = c2.cpno;  

select student.sno,sname,cname,grade
from student join sc on student.sno = sc.sno
join course on sc.cno = course.cno;
 
select student.sno,sname,cname,grade
from student,sc,course
where student.sno = sc.sno and sc.cno = course.cno;

-- 嵌套查询

select sname from student
where sdept = (
    select sdept from student where sanme = 'tanke'
);

select sno,sname,cno from sc as x where grade >= (
    select avg(grade) from sc as y where x.sno = y.sno
);

select x.sname,y.sage from student as x
where x.sdept != 'cs' and x.sage< any(
    select sage from student as y where sdept = 'cs'
);

select sname from student as x
where s.sno exists(
    select sno from sc as y where s.sno = y.sno
    and cno = '001';
)

select e.eno from Employee as e
where exists(
    select * from Construct as c
    where c.pno = '4' and c.eno = e.eno
) and exists(
    select * from Construct as c
    where c.pno = '5' and c.eno = e.eno
);

select sno,sum(cgs) from Construct
group by sno
having sum(cgs) > 500;

-- 集合查询
select * from student where sdept = 'cs'
union
select * from student where sage < 20;