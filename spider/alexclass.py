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


