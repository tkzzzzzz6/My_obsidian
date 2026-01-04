// 插入排序,升序

/* 
for j <- 2 to n do
    key = a[j]
    i = j - 1
    while i > 0 and a[i] > key do
        a[i+1] = a[i]
        i = i - 1
    end
    a[i+1] = key
end
*/

void insertionSort(int arr[],int n)
{
    for(int i = 1;i < n;++i)
    {
        int key = arr[i];
        int j = i - 1;
        while(j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

void insertionSort(int r[],int n)
{
    int i,j;
    int temp;
    for(i = 1;i <n;i++)
    {
        temp = r[i];
        j = i - 1;
        while(j >= 0 && temp < r[j]) // 降序temp变大于
        {
            r[j + 1] = r[j];
            j--;
        }
        r[j + 1] = temp;
    }
}