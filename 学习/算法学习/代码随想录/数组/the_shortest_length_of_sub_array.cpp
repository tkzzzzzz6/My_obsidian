class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int i = 0,res = INT_MAX,sum = 0;
        for(int j = 0;j < nums.size();++j){
            sum += nums[j];
            while(sum >= target){
                res = min(res,j - i + 1);
                sum -= nums[i];
                ++i;
            }
        }
        return res == INT_MAX ? 0 : res ;
    }
};