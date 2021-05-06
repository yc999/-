#-- coding: utf-8 --
import  requests
from bs4 import  BeautifulSoup

headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

# response=requests.get('http://www.bigpian.cn')
response=requests.get('http://easyicon.net',headers= headers)
if response.status_code == 200:
    print("true")


print(response.status_code)


#https://stapharma.com.cn/cn/about-us/facilities/
#https://stapharma.com/about-us/facilities/
# http://www.vslai.com/
re_text=response.text
re_content=response.content
print(re_text)
print(type(re_text))
print(re_content)
print(type(re_content))
response.encoding='utf-8'
re_text=response.text
print(re_text)


