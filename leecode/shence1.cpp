#include <iostream>
using namespace std;
int main() {
    int a,b;
    int x,y,m;
    cin >> x >> y >> m;
    if( x>=m || y>= m){
        cout<< 0<<endl;
        
        return 0;
    }else if( x<=0 && y<=0){
        cout<< -1<<endl;
        return 0;
    }
    int count;
    
    while(x<m && y<m){// 注意，如果输入是多个测试用例，请通过while循环处理多个测试用例
       if(x>y){
            x = x;
            y = x+y;
        }else{
            x = x+y;
            y = y;
        }
        count++;
    }
        cout << count << endl;
    return 0;
}