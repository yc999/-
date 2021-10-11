#include <iostream>
#include <vector>
using namespace std;
int num[7];
int value[7] = {1,2,5,10,20,50,100};
void getcost(int index,long long int cost, vector<int> tmpnum){
    if(index>7 || cost <0){
        return;
    }
    if(cost==0){
        int len = tmpnum.size();
        int i=0;
        for(i=len;i<=7;i++){
            tmpnum.push_back(0);
        }
        for(i=0;i<6;i++){
            cout<< tmpnum[i]<<' ';
        }
            cout<< tmpnum[6]<<endl;
        return;
    }
        for(int j=0;j<=num[index];j++){
            tmpnum.push_back(j);
             getcost(index+1, cost-j*value[index],  tmpnum);
            tmpnum.pop_back();
        }
    return;
}

int main() {
    for(int i=0;i<7;i++){
        cin>> num[i];
    }
    long long int cost;
    cin>>cost;
    
    for(int i=0;i<=num[0];i++){
        vector<int> tmpnum;
        tmpnum.push_back(i);
       getcost(1, cost-i,  tmpnum);
        tmpnum.pop_back();
    }
    return 0;
}