
import requests

path = "D:/dnswork/sharevm/"  # 保存路径
filename = path + "successfile.txt"
writefile = open(filename,'w', encoding='utf-8')

path = "D:/dnswork/sharevm/"  # 读取路径
filename = path + "urllist.txt"
readfile = open(filename,'r', encoding='utf-8')

headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

def query_web(domain_name):
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}
    try:
        response=requests.get(domain_name,headers = headers)
        if response.status_code == 200:
            return True
        return False
    except:
        return False

while True:
    line = readfile.readline()
    if  line:
        parts = line.split(',')
        url = parts[1]
        httpurl = "http://" + url
        finish = False
        try:
            response=requests.get(httpurl,headers = headers)
            if response.status_code == 200:
                writefile.write(line)
            finish = True
        except:
            pass
        if not finish:
            if url.split(".")[0]!="www":
                httpurl = 'http://www.' + url
                parts[1] = 'www.' + url
                line = ','.join(parts)
            else:
                httpurl = 'http://' + url.replace('www.','',1)
                parts[1] = url.replace('www.','',1)
                line = ','.join(parts)
            try:
                response=requests.get(httpurl,headers = headers)
                if response.status_code == 200:
                    writefile.write(line)
            except:
                pass
    else:
        break


writefile.close()

headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

domain_name = 'http://www.baidu.com'
domain_name = 'http://36.152.44.96'
response=requests.get(domain_name,verify=False,allow_redirects=True,headers = headers)


response=requests.get(domain_name,verify=False,allow_redirects=True,headers = headers)
