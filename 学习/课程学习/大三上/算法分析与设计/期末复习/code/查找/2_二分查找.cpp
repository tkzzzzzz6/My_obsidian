// 二分查找,升序数组

// 递归方式实现
int binarySearch(int arr[], int value, int low, int high)
{
    int mid = (low + high) / 2;
    if (arr[mid] == value)
        return mid;
    if (arr[mid] > value)
        return binarySearch(arr, value, low, mid - 1);
    else
        return binarySearch(arr, value, mid + 1, high);
}

// 非递归方式实现
int binarySearch(int arr[], int value, int n)
{
    int low = 0, high = n - 1, mid;
    while (low <= high)
    {
    }
}
