-- constrain

SET SQL_SAFE_UPDATES = 0;

CREATE TABLE student(
    'student_id' INT AUTO_INCREMENT,
    'name' VARCHAR(20) NOT NULL,
    'major' VARCHAR(20) DEFAULT 'NULL',
    PRIMARY KEY('student_id') 
)

INSERT INTO 'student' VALUES(1,'小白','计算机科学与技术');

SELECT * FROM 'student'; --search for all data

INSERT INTO 'student'('student_id','name','major') VALUES(2,'小黑','数学');
INSERT INTO 'student'('student_id','name','major') VALUES(3,'小红','NULL');
INSERT INTO 'student'('student_id','name') VALUES(4,'小蓝');

UPDATE student
SET 'major' = '大数据科学与技术'
WHERE 'major' = '数据'

