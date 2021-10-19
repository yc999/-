#include <iostream>
#include <string>
#include <map>
#include <queue>

using namespace std;
int main(){
    int input[1002];
    int i=0;
    cin>>input[i++];
    char tmp;
    cin>>tmp;
    while (tmp!='\n')
    {
       cin >> input[i++];
    }
    cout<< i;
    
}