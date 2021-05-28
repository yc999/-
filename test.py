#-- coding: utf-8 --
import  requests
from bs4 import  BeautifulSoup, Comment
from hyper.contrib import HTTP20Adapter

headers = {
#  "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
# "Referer": 'www.baidu.com',
# "host": "allegro.pl"
":authority": "allegro.pl",
":method": "GET",
":path": "/",
":scheme": "https",
"accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
"accept-encoding": "gzip, deflate, br",
"accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
# "cookie": "_cmuid=81d73072-2bbe-4a13-945b-ab145dbc23ab; __gfp_64b=T8ZmiPzI85urKGng3_NG7vTdaGgi_9mR9drIR8u.PXf.T7|1620284247; gdpr_permission_given=1; datadome=Hji1qdEFzNPT99QQd0StqcyeyR72BiFyxU6R-Tidm5_2cSwrCjlbULzOt1HZmjtl2sjjTz3l49.5Mqwv8PkHiEeTsKgkn_yxeL.8UyMycR",
"dpr": "1",
# "sec-ch-ua": "Google Chrome";v="89", "Chromium";v="89", ";Not A Brand";v="99"
# sec-ch-ua-mobile: ?0
# sec-fetch-dest: document
# sec-fetch-mode: navigate
# sec-fetch-site: none
# sec-fetch-user: ?1
"upgrade-insecure-requests": "1",
"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
"viewport-width": "1920"
}

sessions=requests.session()
# sessions.mount('http://allegro.pl', HTTP20Adapter())
# response=requests.get('http://www.bigpian.cn')
# response=requests.get('https://nuxechina.com',verify=False)
# response=sessions.get('http://tech.ifeng.com',headers = headers)
# response=sessions.get('http://tech.ifeng.com')


#https://stapharma.com.cn/cn/about-us/facilities/
#https://stapharma.com/about-us/facilities/
# http://www.vslai.com/
# re_text=response.text
# print(re_text)
# print(type(re_text))

# response.encoding='utf-8'
# re_text=response.text
# print(re_text)

# print(re_content)
# print(type(re_content))


# headers={
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
# "Upgrade-Insecure-Requests": "1",
# "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"
# }
# response=requests.get('http://89ws.com',headers = headers,allow_redirects=False)



# [s.extract() for s in soup('script')]
# [s.extract() for s in soup('style')]

url ="http://" + "byby.xywy.com"
sessions=requests.session()
sessions.mount(url, HTTP20Adapter())
headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        } 
response = requests.get(url,verify=False,allow_redirects=True,headers = headers)
response.encoding = requests.utils.get_encodings_from_content(response.text)
print(response.encoding)

if response.encoding == ['gb2312']:
    response.encoding = 'GBK'
if response.encoding == ['gbk']:
    response.encoding = 'GBK'
print(response.encoding)
# print(response.text)
# sessions.close()
soup = BeautifulSoup(response.text,'html.parser')

[s.extract() for s in soup('script')]
[s.extract() for s in soup('style')]
for element in soup(text = lambda text: isinstance(text, Comment)):
            element.extract()
print(type(soup.get_text()))
# print(soup.get_text())
