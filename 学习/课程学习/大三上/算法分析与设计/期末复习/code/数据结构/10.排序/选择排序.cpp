// 选择排序
/* 
for i <- 1 to  n -1 do
    for j <- i + 1 do
        // 如果 a[i] 大于 a[j],交换两者位置
        if a[i] > a[j] then
            交换 a[i] 和 a[j]
        end
    end
end 
*/
void select_sort(int *a, int n)
{
    int i, j, t;
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            if (a[i] > a[j])
            {
                t = a[i];
                a[i] = a[j];
                a[j] = t;
            }
        }
    }
}
