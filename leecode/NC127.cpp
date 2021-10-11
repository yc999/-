#include <iostream>
#include <cstdio>
#include <map>
#include <vector>
#include<algorithm>
using namespace std;

class Solution {
public:
    /**
     * longest common substring
     * @param str1 string字符串 the string
     * @param str2 string字符串 the string
     * @return string字符串
     */
    string LCS(string str1, string str2) {
        // write code here
        int len = 0;
        int maxlen=-1;
        int len1=str1.size();
        int len2 = str2.size();
        int index;
        int maxindex=-1;
        for(int i=0;i<len1;i++){
            for(int j=0;j<len2;j++){
                len = 0;
                for(int k=j;k<len2;k++){
                    if(str1[i+len]==str2[k]){
                        len++;
                        if(len>maxlen){
                            maxlen = len;
                            maxindex = i;
                            cout<<maxlen<<endl;
                        }
                    }else{
                       break;
                    }
                }
            }
        }
        if(maxindex==-1){
            return "-1";
        }
        string result = str1.substr(maxindex,maxlen);
        cout<<result<<endl;
        return result;
    }
};

int main(){
    Solution a;
    a.LCS("1AB2345CD","12345EF");
    return 0;
}