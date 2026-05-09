void earse_num1(vector<int> &arr,int e){
    int slow = fast = 0;
    for(int fast = 0;fast < arr.size();++fast){
        if(arr[fast] != e){
            arr[slow] = arr[fast]; //等效于 arr[slow++] = arr[fast];
            ++slow;
        }
    }
}

void earse_num2(vector<int> &arr, int e){
    for(int i = 0; i < arr.size(); ++i){
        if(arr[i] == e){
            for(int j = i+1; j < arr.size(); ++j){
                arr[j-1] = arr[j];
            }
            arr.pop_back();  // 删除最后一个元素
            --i;  // 重新检查当前位置
        }
    }
}