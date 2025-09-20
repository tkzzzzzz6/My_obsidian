-- 取“商品数量”排在前 3 名的商品类别，并且把第 3 名的并列也一并返回
SELECT TOP (3) WITH TIES
       商品类别,              -- 要展示的分组字段：商品类别
       COUNT(*) AS 商品数量    -- 每个类别下的商品条目数
FROM   商品表                  -- 数据来源表
GROUP BY 商品类别              -- 先按商品类别分组做聚合统计
ORDER BY COUNT(*) DESC;        -- 按“商品数量”从大到小排序；WITH TIES 依据该排序保留并列
