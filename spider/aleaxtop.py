from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"  # 保存路径
filename = path + "aleax" + ".txt"
f = open(filename,'w', encoding='utf-8')

website = 'http://stuffgate.com'
r = requests.get('http://stuffgate.com/stuff/website/top-1000-sites')
r.encoding = "utf-8"
soup = BeautifulSoup(r.text, 'html.parser')
# print(soup.get_text())
tagtitle = soup.find(name='div', class_='sg-overflow')
tagtitle = tagtitle.find(name='tbody')
tagsubtitle = tagtitle.find_all(name='tr')
for index, title in enumerate(tagsubtitle):
    site_tag = title.find(name='a')
    sit_url = site_tag.get_text()
    f.write(sit_url + '\n')
    # print(sit_url)

for times in range(0,700):
    nexttag = soup.find(name='div', class_='col-lg-8')
    nexttag = nexttag.find(name='table', class_='table')
    atags = nexttag.find_all(name='a')
    # nexttext = 'Next 1000 websites »'
    lentag = len(atags)
    atag = atags[lentag-1]['href']
    nextatag = website + atag
    print(nextatag)
    r = requests.get(nextatag)
    r.encoding = "utf-8"
    soup = BeautifulSoup(r.text, 'html.parser')
    tagtitle = soup.find(name='div', class_='sg-overflow')
    tagtitle = tagtitle.find(name='tbody')
    tagsubtitle = tagtitle.find_all(name='tr')
    for index, title in enumerate(tagsubtitle):
        site_tag = title.find(name='a')
        sit_url = site_tag.get_text()
        f.write(sit_url + '\n')
f.close()