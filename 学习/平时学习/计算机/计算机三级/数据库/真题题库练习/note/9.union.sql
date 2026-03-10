-- union
SELECT name FROM Employee 
UNION
SELECT client_name FROM client
UNION 
SELECT branch_name FROM branch
;

SLEECT emp_id AS total_id,name as total_name FROM Employee
UNION
SELECT client_id,client_name FROM client
;