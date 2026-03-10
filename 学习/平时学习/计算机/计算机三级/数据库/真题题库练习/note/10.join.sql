INSERT INTO branch VALUES(1,'谭棵',NULL);

SELECT * FROM empolyee join branch on empolyee.branch_id = branch.branch_id;
SELECT * FROM empolyee left join branch on empolyee.branch_id = branch.branch_id;
SELECT * FROM empolyee right join branch on empolyee.branch_id = branch.branch_id;