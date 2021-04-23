#-- coding: utf-8 --
import  requests
from bs4 import  BeautifulSoup

# response=requests.get('http://www.bigpian.cn')
response=requests.get('https://stapharma.com/about-us/facilities/',verify=False)
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


