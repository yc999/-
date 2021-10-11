#include <iostream>
#include <string>
#include <map>
using namespace std;

class CUrlParse {
public:
    map<string, string> m_param;
    int parse(const string &url){
        int lenth = url.size();
        int i;
        for( i=0;i<lenth;i++){
            if(url[i]=='?'){
                break;
            }
        }
        int j = i+1;
        while (j<lenth){
            string pa;
            for(int k=j;k<lenth;k++){
                if(url[k]=='='){
                    pa = url.substr(j,k-j);
                    j=k+1;
                    break;
                }
            }
            string pb;
            for(int k=j;k<lenth;k++){
                if(url[k]=='&'){
                    pb = url.substr(j,k-j);
                    j=k+1;
                    break;
                }
                if(k==lenth-1){
                    pb = url.substr(j,lenth-j);
                    j=k+1;
                    break;
                }
            }
            m_param[pa]=pb;
        }
        

       return 0;
    }
    
    string getParam(const string& param){
        return m_param[param];
    }
    
};


int main(int argc, char *argv[]) {
    // if(argc <=1){
    //     return 0;
    // }
    // string url = argv[1];
    CUrlParse obj;
    string a = ".com?a=b&time=x";
    obj.parse(a);
    
    CUrlParse copy_obj(obj);
    cout << copy_obj.getParam("time") << endl;
    return 0;
}