// 本题为考试多行输入输出规范示例，无需提交，不计分。
#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include<algorithm>
using namespace std;
//= {"A" : 1, '2':2,'3':3,'4':4, '5':5, '6':6, '7':7, '8':8,'9':9 , 'T':10,'J':11,'Q':12,'K':13}
map<char, int> shunxu ;


map<char, int> shunxu1 ;
bool cmp(string a, string b){
    if(a[0]!=b[0]){
        return shunxu[a[0]] < shunxu[b[0]];
    }else{
        return shunxu1[a[1]] < shunxu1[b[1]];
        
    }
    return true;
    
}
// As  Kd Jd Js Jc Tc 9c
// 7s  Kd Jd 8s Jc Tc 9c

int main(){
    shunxu['A']=1;
    shunxu['2']=2;
    shunxu['3']=3;
    shunxu['4']=4;
    shunxu['5']=5;
    shunxu['6']=6;
    shunxu['7']=7;
    shunxu['8']=8;
    shunxu['9']=9;
    shunxu['T']=10;
    shunxu['J']=11;
    shunxu['Q']=12;
    shunxu['K']=13;
    shunxu1['s']=4;
    shunxu1['d']=3;
    shunxu1['c']=2;
    shunxu1['h']=1;
    
    //freopen("1.in","r",stdin);
    vector<string> input;
    string tmp;
    int n,ans = 0;
    for(int i = 0; i < 7; i++){
        cin >> tmp;
        input.push_back(tmp);
    }
    sort(input.begin(),input.end(),cmp);
    for(int i = 0; i < 7; i++)
        cout<< input[i]<< " ";
    
    cout<<endl;

    for(int i = 0; i < 7; i++){
        int tmp =shunxu[input[i][0]];
        vector<string> ret;
        ret.push_back(input[i]);
        int count = 0;

        for(int j = i+1;j<7;j++){
            if(tmp == shunxu[input[j][0]]-1){
                tmp = shunxu[input[j][0]];
                count++;
                ret.push_back(input[j]);
            }
            if(count == 4){
                for(int k=0;k<5;k++){
                    cout<< input[k]<<" ";
                }
                return 0 ;
            }
        }
    }
    cout<<"false";
    return 0;
}