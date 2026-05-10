// 顺序查找
int sequenceSearch(int a[], int value, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (a[i] == value)
        {
            return i;
        }
    }
    return -1;
}