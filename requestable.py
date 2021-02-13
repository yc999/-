import  requests
from bs4 import  BeautifulSoup
import re
import time
import eventlet

path = "D:/dnswork/sharevm/unclassify.txt"
f = open(path,"r")
urlList = f.readlines()

eventlet.monkey_patch(time=True)
time_limit = 15  #set timeout time 3s
# @timeout_decorator.timeout(30)
def requesturl(url):
    s=requests.session()
    s.keep_alive = False
    r = s.get(url,timeout=5)
    # return

good_result1=[]
bad_result = []
else_result=[]
for url in urlList:
    time.sleep(0.5)
    url = "".join(url.split())
#     output = open(path + "1.result/"+ url + ".txt","w",encoding='utf-8')
    httpurl = 'http://www.' + url
    with eventlet.Timeout(time_limit,False):
        try:
            requesturl(httpurl)
            good_result1.append(url)
        except:
            httpsurl = 'https://www.' + url
            try:
                requesturl(httpsurl)
                good_result1.append(httpsurl)
            except:
                # print("bad")
                bad_result.append(url)
            else:
                print(httpsurl)
                else_result.append(httpsurl)
        else:
            print("http "+ url)
            else_result.append(url)
        continue
    print("fail ", url)
    bad_result.append(url)

path = "D:/dnswork/sharevm/"
websit = "good_result1"
filename = path + websit + ".txt"
f = open(filename,'w')
for item in good_result1:
        f.write(item + '\n')
f.close()

websit = "bad_result1"
filename = path + websit + ".txt"
f = open(filename,'w')
for item in bad_result:
        f.write(item + '\n')
f.close()


websit = "else_result"
filename = path + websit + ".txt"
f = open(filename,'w')
for item in else_result:
        f.write(item + '\n')
f.close()