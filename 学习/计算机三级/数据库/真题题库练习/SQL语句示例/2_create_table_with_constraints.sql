-- 创建表R4，包含复合主键和外键约束
CREATE TABLE R4 (
    a int,                    -- 定义列a，数据类型为整数
    e int,                    -- 定义列e，数据类型为整数  
    g int,                    -- 定义列g，数据类型为整数
    PRIMARY KEY (a, e),       -- 定义复合主键，由列a和e组成
    FOREIGN KEY (a) REFERENCES R1(a),  -- 定义外键约束：列a引用表R1的列a
    FOREIGN KEY (e) REFERENCES R3(e)   -- 定义外键约束：列e引用表R3的列e
);
