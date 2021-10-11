
#include <iostream>
#include <string>
#include <map>
#include <queue>

using namespace std;
int findMinPath(vector<int>& start, vector<int>& end) {
        // write code here
        int value[11][11] =  { 
                               0,0,0,0,0,0,0,0,0,0,0,
                               0,0,1,1,0,0,0,0,0,0,0,
                               0,0,1,0,0,0,0,1,0,0,0,
                               0,1,1,0,0,1,1,0,0,0,0,
                               0,0,0,0,0,1,0,0,0,0,0,
                               0,0,0,0,0,1,0,0,0,0,0,

                               0,0,0,0,0,1,0,0,0,0,0,
                               0,0,0,1,1,1,0,0,0,0,0,
                               0,0,1,0,0,0,0,1,1,0,0,
                               0,0,0,0,0,0,0,0,0,0,0,
                               0,0,0,0,0,0,0,0,0,0,0,

                              
                            };
      int x1,y1,x2,y2;
      int count=0;
        int visted[11][11]={0};
        x2 = end[0];
        y2 = end[1];
        queue<vector<int>> visit;
        start.push_back(count);
        visit.push(start);
        while(!visit.empty()){
            int flag = 1;
            x1 = visit.front()[0];
            y1 = visit.front()[1];
            if( x1==x2&&y1==y2){
            cout<< x1<<' '<<y1<<' '<<visit.front()[2]<<endl;

                return visit.front()[2];
            }
            cout<< x1<<' '<<y1<<' '<<visit.front()[2]<<endl;
            visit.pop();
            // count++;
            if(x1+1<=10){
                if(value[x1+1][y1]==0 && visted[x1+1][y1]==0){
                    visted[x1+1][y1]=1;
                    vector<int> tmp;
                    tmp.push_back(x1+1);
                    tmp.push_back(y1);
                    if(flag){
                        flag=0;
                        count++;
                    }
                    tmp.push_back(count);
                    visit.push(tmp);
                }
            }
            if(x1-1>=0){
                if(value[x1-1][y1]==0 && visted[x1-1][y1]==0){
                    visted[x1-1][y1]=1;
                    vector<int> tmp;
                    tmp.push_back(x1-1);
                    tmp.push_back(y1);
                    if(flag){
                        flag=0;
                        count++;
                    }
                    tmp.push_back(count);
                    visit.push(tmp);
                }
            }
            if(y1+1<=10){
                if(value[x1][y1+1]==0 && visted[x1][y1+1]==0){
                    visted[x1][y1+1]=1;
                    vector<int> tmp;
                    tmp.push_back(x1);
                    tmp.push_back(y1+1);
                    if(flag){
                        flag=0;
                        count++;
                    }
                    tmp.push_back(count);
                    visit.push(tmp);
                }
            }
            if(y1-1>=0){
                if(value[x1][y1-1]==0 && visted[x1][y1-1]==0){
                    visted[x1][y1-1]=1;
                    vector<int> tmp;
                    tmp.push_back(x1);
                    tmp.push_back(y1-1);
                    if(flag){
                        flag=0;
                        count++;
                    }
                    tmp.push_back(count);
                    visit.push(tmp);
                }
            }
        }
        return count;
    }

int main(){
    vector<int> a;
    vector<int> b;
    a.push_back(1);
    a.push_back(1);
    b.push_back(2);
    b.push_back(3);
   int count =  findMinPath(a,b);
   cout<< count;

}