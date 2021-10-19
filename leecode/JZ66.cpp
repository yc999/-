#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include<algorithm>
using namespace std;
class Solution {
public:
        int his[100][100]={0};
    
    int movingCount(int threshold, int rows, int cols) {
        if(threshold<0 || rows<0 || cols<0){
            return 0;
        }
        int count = dfs(0,0, threshold, rows,  cols);
        return count;
    }
    int dfs(int row, int col, int threshold,int rows, int cols){
        if(row>=0 && col>=0 && row<rows && col<cols ){
            if(overthreshold(row,col,threshold) || his[row][col]==1){
                    return 0;
             }
        
        his[row][col]=1;
        return 1+dfs(row+1,col,threshold,rows,  cols)+
                dfs(row,col+1,threshold,rows,  cols)+
                dfs(row-1,col,threshold, rows,  cols)+
                dfs(row,col-1,threshold, rows,  cols);
         
         }
        return 0;
    }
    int overthreshold(int i,int j,int th){
        int ret=0;
        while(i){
            ret += i % 10;
            i /= 10;
        }
        while(j){
            ret += j % 10;
            j /= 10;
        }
        if(ret>th){
            return 1;
        }
        return 0;
    }
};

int main(){
    Solution a;
    a.movingCount(5,10,10);
    return 0;
}