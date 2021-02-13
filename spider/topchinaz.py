from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"
websit = "top.chinaz"
filename = path + websit + ".txt"

f = open(filename,'w', encoding='utf-8')
url = 'https://top.chinaz.com/hangye/'

r = requests.get(url)
r.encoding = "utf-8"
soup = BeautifulSoup(r.text, 'html.parser')
# print(soup.get_text())
tags = soup.find_all(name='div', class_='HeadFilter clearfix')
# tags = second.find_all(name='div', class_='list')
for tag in tags:
    list_items = tag.find_all(name='a')
    for item in list_items:
        item_class = item.get_text()
        print(item_class)
        item_url = item['href']
        sub_r = requests.get(url+ item_url)
        sub_r.encoding = "utf-8"
        sub_soup = BeautifulSoup(sub_r.text, 'html.parser')
        sub_tags = sub_soup.find_all(name='div', class_='HeadFilter clearfix')
        sub_items = sub_tags[0].find_all(name='a')
        for sub_item in sub_items:
            subitem_url = sub_item['href']
            subtitle = sub_item.get_text()
            print(subtitle)
            f.write(subtitle+","+subitem_url+","+item_class + '\n')





