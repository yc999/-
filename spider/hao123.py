from bs4 import BeautifulSoup
import requests

r = requests.get('https://www.hao123.com/')
soup = BeautifulSoup(r.text, 'html.parser')
tags = soup.find_all(name='ul', class_='cool-row')
for tag in tags:
    site_tag = tag.find(name='li',class_='site-item first').get_text('a')
    items = tag.find_all(name='li', class_='site-item')
    for item in items:
        item_url = item.find('a')["href"]
        item_name = item.get_text('a')
        print(site_tag+","+item_url+","+item_name)
