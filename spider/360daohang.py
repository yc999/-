from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"
websit = "hao.360.com"
filename = path + websit + ".txt"

f = open(filename,'w')

r = requests.get('https://hao.360.com/')
soup = BeautifulSoup(r.text, 'html.parser')
tags = soup.find_all(name='div', class_='sites-lib-title')
for tag in tags:
    site_tag = tag.find(name = 'a', class_ = 'sites-classify').get_text()
    item_name = tag.find(name = 'a', class_ = 'sites-lib-site-text').get_text()
    item_url = tag.find(name = 'a', class_ = 'sites-lib-site-text')["href"]

    items = tag.find_all(name='div', class_='sites-lib-site')
    f.write(site_tag+","+item_url+","+item_name+ '\n')
    print(site_tag+","+item_url+","+item_name)
    for item in items:
        item_url = item.find('a')["href"]
        item_name = item.find('a').get_text().strip('\n')
        print(site_tag+","+item_url+","+item_name)
        f.write(site_tag+","+item_url+","+item_name + '\n')