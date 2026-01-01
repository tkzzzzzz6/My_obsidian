CREATE TABLE student(
    'student_id' INT AUTO_INCREMENT,
    'name' VARCHAR(20) NOT NULL,
    'major' VARCHAR(20) DEFAULT 'NULL',
    PRIMARY KEY('student_id') 
    'score' INT
)

INSERT INTO 'student' VALUES(1,'小白','计算机科学与技术');

SELECT * FROM 'student'; --search for all data

INSERT INTO 'student'('student_id','name','major') VALUES(2,'小黑','数学');
INSERT INTO 'student'('student_id','name','major') VALUES(3,'小红','NULL');
INSERT INTO 'student'('student_id','name') VALUES(4,'小蓝');

SELECT 'name','major' FROM student ORDER BY 'score','student_id' ASC;
-- ASC low to high *DEFUALT*
-- DESC high to low

SELECT * FROM student ORDER BY 'score' DESC LIMIT 3;
SELECT * FROM student WHERE'score' <> 80;
SELECT * FROM student WHERE 'major' IN ('数学','计算机科学与技术','NULL');
SELECT * FROM student WHERE 'major' = 'NULL' OR 'major' = '计算机科学与技术';
