from bs4 import BeautifulSoup
import requests

path = "D:/dnswork/sharevm/"
websit = "alexaclass"
filename = path + websit + ".txt"

weburl = "https://alexa.chinaz.com"
with open(filename, 'r', encoding='utf-8') as file_to_read:
    while True:
        line = file_to_read.readline()
        parts = line.split(",")
        print(line)
        # print(parts)
        if not line:
            break
        fwrite = open(path+"alexclassify/"+parts[0].strip('\n')+".txt",'a+', encoding='utf-8')
        s=requests.session()
        s.keep_alive = False
        r = s.get(weburl+parts[1])
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, 'html.parser')
        # print(soup.get_text())
        tagtitle = soup.find_all(name='div', class_='righttxt')
        for title in tagtitle:
            item = title.find_all(name='a')
            itemurl = item[1]['href']
            itemabstract = title.find(name='p').get_text()
            fwrite.write(parts[0]+","+itemurl+","+itemabstract + '\n')
        
        for i in range(2,21):
            urlparts = parts[1].split(".")
            s=requests.session()
            s.keep_alive = False
            print(weburl+urlparts[0]+"_"+str(i) + ".html")
            r = s.get(weburl+urlparts[0]+"_"+str(i) + ".html")
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, 'html.parser')
            tagtitle = soup.find_all(name='div', class_='righttxt')
            for title in tagtitle:
                item = title.find_all(name='a')
                try:
                    itemurl = item[1]['href']
                except:
                    print(weburl+urlparts[0]+"_"+str(i) + ".html")
                    print(item)
                    # fwrite.write(parts[0]+","+itemurl+","+itemabstract + '\n')

                else:
                    itemAbstract = title.find(name='p').get_text()
                    fwrite.write(parts[0]+","+itemurl+","+itemabstract + '\n')
        fwrite.close()
            # print("_"+str(i))
        # tagsubtitle = soup.find_all(name='div', class_='world_h3nav clearfix')
        # for index, title in enumerate(tagtitle):
        #     site_tag = title.get_text()
        #     # print(index)
        #     subtitlelist = tagsubtitle[index].find_all(name='a')
        #     for subtitle in subtitlelist:
        #         # print(subtitle['href'], subtitle.get_text())
        #         f.write(subtitle.get_text()+","+subtitle['href']+","+site_tag + '\n')

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


