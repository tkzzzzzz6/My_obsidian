CREATE FUNCTION f_BorrowBook(@jszh char(20))

RETURNS @ResultTable TABLE(
    书籍编号 char(20),
    书籍名称 varchar(100),
    定价 decimal(10,2),
    借书日期 datetime
)
