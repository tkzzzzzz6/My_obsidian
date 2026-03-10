CREATE TABLE student(
    'student_id' INT,
    'name' VARCHAR(20),
    'major' VARCHAR(20),
    PRIMARY KEY('student_id')
)

INSERT INTO 'student' VALUES(1,'小白','计算机科学与技术');

SELECT * FROM 'student';

INSERT INTO 'student'('student_id','name','major') VALUES(2,'小黑','数学');
INSERT INTO 'student'('student_id','name','major') VALUES(3,'小红','NULL');
INSERT INTO 'student'('student_id','name') VALUES(4,'小蓝');
