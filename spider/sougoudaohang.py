from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"
websit = "123.sogou.com"
filename = path + websit + ".txt"

f = open(filename,'w', encoding='utf-8')

r = requests.get('https://123.sogou.com/')
soup = BeautifulSoup(r.text, 'html.parser')
# print(soup.get_text())
tags = soup.find_all(name='div', class_='m-5 m')
# tags = second.find_all(name='div', class_='list')
for tag in tags:
    list_items = tag.find_all(name='div', class_='list')
    # item_name = tag.find(name = 'a', class_ = 'sites-lib-site-text').get_text()
    # item_url = tag.find(name = 'a', class_ = 'sites-lib-site-text')["href"]
    # print(site_tag+","+item_url+","+item_name)
    for item in list_items:
        # print(item)
        ul_items = item.find_all(name='ul', class_='cf')
        # print(ul_items)
        for ul_item in ul_items:
            # print(ul_item)
            li_items1 = ul_item.find(name ='li', class_ = 'col-1')
            if li_items1 == None:
                continue
            site_tag = li_items1.get_text()
            li_items2 = ul_item.find(name ='li', class_ = 'col-2')
            li_items2 = li_items2.find(name='ul', class_='cf')
            li_items2 = li_items2.find_all(name='li')
            for li_item in li_items2:
                item_url = li_item.find('a')["href"]
                item_name = li_item.find('a').get_text().strip('\n')
                print(site_tag+","+item_url+","+item_name)
                f.write(site_tag+","+item_url+","+item_name + '\n')


# for tag in tags:
    # site_tag = tag.find(name='li',class_='site-item first').get_text('a')
    # items = tag.find_all(name='li', class_='site-item')
    # for item in items:
    #     item_url = item.find('a')["href"]
    #     item_name = item.get_text('a')
    #     print(site_tag+","+item_url+","+item_name)
