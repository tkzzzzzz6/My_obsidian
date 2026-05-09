class Solution
{
public:
    vector<int> sortedSquares(vector<int> &nums)
    {
        vector<int> res(nums.size()); // 需要预先给出 res 的大小,不然 res[k--]会报错
        int k = nums.size() - 1, i = 0, j = k;
        while (i <= j)
        {
            int sq_i = nums[i] * nums[i];
            int sq_j = nums[j] * nums[j];
            if (sq_i > sq_j)
            {
                res[k--] = sq_i;
                ++i;
            }
            else
            { // sq_i == sq_j 时,选 i 和选 j 位置都是可以的
                res[k--] = sq_j;
                --j;
            }
        }
        return res;
    }
};