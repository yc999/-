#-- coding: utf-8 --
# 爬取网页
from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"
websit = "hao.66360.cn"
filename = path + websit + ".txt"

f = open(filename,'w', encoding='utf-8')

r = requests.get('http://hao.66360.cn/')
r.encoding = "utf-8"
soup = BeautifulSoup(r.text)
# print(soup.get_text())
tags = soup.find_all(name='div', class_='main-kuzhan')
for tag in tags:
    list_items = tag.find_all(name='div', class_='kuzhan')
    for item in list_items:
        ul_items = item.find_all(name='ul', class_='list')
        for ul_item in ul_items:
            li_items = ul_item.find_all(name ='li')
            site_tag = li_items[0].get_text()
            # print(li_items[0])
            for li_item in li_items[1:]:
                item_url = li_item.find('a')["href"]
                item_name = li_item.find('a').get_text().strip('\n')
                print(site_tag+","+item_url+","+item_name)
                f.write(site_tag+","+item_url+","+item_name + '\n')