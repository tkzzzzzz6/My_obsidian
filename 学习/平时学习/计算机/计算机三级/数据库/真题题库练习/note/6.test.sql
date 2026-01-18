CREATE TABLE Employee(
    emp_id int PRIMARY KEY,
    name varchar(20),
    birth_date date,
    sex char(1),
    salary int,
    FOREIGN KEY(branch_id) REFERENCES Branch(branch_id),
    FOREIGN KEY(sup_id) REFERENCES Employee(emp_id)
)

CREATE TABLE Branch(
    branch_id int PRIMARY KEY,
    branch_name varchar(20),
    FOREIGN KEY(manager_id) REFERENCES Employee(emp_id) ON DELETE SET NULL
)

CREATE TABLE Client(
    client_id int PRIMARY KEY,
    client_name varchar(20),
    phone varchar(11),
    FOREIGN KEY(branch_id) REFERENCES Branch(branch_id) ON DELETE SET NULL
)

CREATE TABLE Works_with(
    emp_id int,
    client_id int,
    total_sales int,
    PRIMARY KEY(emp_id, client_id),
    FOREIGN KEY(emp_id) REFERENCES Employee(emp_id) ON DELETE CASCADE,
    FOREIGN KEY(client_id) REFERENCES Client(client_id) ON DELETE CASCADE
)

SELECT * FROM Employee;

select * FROM Client;

select * FROM Employee ORDER BY salary ASC;

SELECT * FROM Employee ORDER BY salary DESC LIMIT 3;
-- 错误语法：SELECT * TOP(3) FROM Employee ORDER BY salary DESC;
-- 正确语法（SQL Server）：
SELECT TOP(3) * FROM Employee ORDER BY salary DESC;
-- 正确语法（MySQL）：
-- SELECT * FROM Employee ORDER BY salary DESC LIMIT 3;

SELECT DISTINCT name FROM Employee; 