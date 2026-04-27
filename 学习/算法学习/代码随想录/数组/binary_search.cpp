#include <vector>
using namespace std;

// 左闭右闭
int binary_search_closed_closed(const vector<int> &nums, int target)
{
    int left = 0, right = static_cast<int>(nums.size()) - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2; // 等效 (left + right)/2,但可防止溢出
        if (nums[mid] > target)
            right = mid - 1;
        else if (nums[mid] < target)
            left = mid + 1;
        else
            return mid;
    }
    return -1;
}

// 左闭右开
int binary_search_closed_open(const vector<int> &nums, int target)
{
    int left = 0, right = static_cast<int>(nums.size()); // diff1
    while (left < right)                                 // diff2
    {
        int mid = left + (right - left) / 2;
        if (nums[mid] > target)
            right = mid; // diff3
        else if (nums[mid] < target)
            left = mid + 1;
        else
            return mid;
    }
    return -1;
}