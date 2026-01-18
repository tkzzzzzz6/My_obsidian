-- aggregate function

SELECT COUNT(*) FROM Employee;

SELECT SUM(emp_id) FROM Employee WHERE birth_date > '1970-01-01' AND sex = 'F';

SELECT AVG(salary) FROM Employee;

SELECT SUM(salary) FROM Employee;

SELECT name FROM Employee WHERE salary = MAX(salary);

SELECT name FROM Employee WHERE salary = MIN(salary);