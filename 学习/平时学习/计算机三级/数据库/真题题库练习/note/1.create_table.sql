CREATE DATABASE 'sql_tutorial';
SHOW 'sql_tutorial';
USE 'sql_tutorial';

-- INT
-- DECIMAL(m,n)
-- VARCHAR(m)
-- BOLB
-- DATE
-- TIMESTAMP

CREATE TABLE student(
    'student_id' INT PRIMARY KEY,
    'name' VARCHAR(20),
    'major' VARCHAR(20),
)

DESCRIBE 'student';
DROP TABLE  'student';

ALTER TABLE student ADD gpa DECIMAL(4,2);
ALTER TABLE student DROP COLUMN gpa;