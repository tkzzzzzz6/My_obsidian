SELECT name FROM employee WHERE emp_id = (SELECT manager_id FROM branch WHERE branch_name = 'develop');

SELECT name FROM employee WHERE emp_id = IN(SELECT emp_id FROM Works_with WHERE total_sales >= 50000);