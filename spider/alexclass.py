from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"
websit = "alexaclass"
filename = path + websit + ".txt"

f = open(filename,'w', encoding='utf-8')

r = requests.get('https://alexa.chinaz.com/Category/index.html')
r.encoding = "utf-8"
soup = BeautifulSoup(r.text, 'html.parser')
# print(soup.get_text())
tagtitle = soup.find_all(name='div', class_='h5_title')
tagsubtitle = soup.find_all(name='div', class_='world_h3nav clearfix')
for index, title in enumerate(tagtitle):
    site_tag = title.get_text()
    # print(index)
    subtitlelist = tagsubtitle[index].find_all(name='a')
    for subtitle in subtitlelist:
        # print(subtitle['href'], subtitle.get_text())
        f.write(subtitle.get_text()+","+subtitle['href']+","+site_tag + '\n')

# print(tagsubtitle[1])
    
    # list_items = tag.find_all(name='div', class_='kuzhan')
    # for item in list_items:
    #     ul_items = item.find_all(name='ul', class_='list')
    #     for ul_item in ul_items:
    #         li_items = ul_item.find_all(name ='li')
    #         site_tag = li_items[0].get_text()
    #         # print(li_items[0])
    #         for li_item in li_items[1:]:
    #             item_url = li_item.find('a')["href"]
    #             item_name = li_item.find('a').get_text().strip('\n')
    #             print(site_tag+","+item_url+","+item_name)
                # f.write(site_tag+","+item_url+","+item_name + '\n')


