from bs4 import BeautifulSoup
import requests
import re
import time
path = "D:/dnswork/sharevm/"
websit = "top.chinaz"
filename = path + websit + ".txt"

weburl = 'https://top.chinaz.com/hangye/'
# url = 'https://top.chinaz.com/hangye/'
with open(filename, 'r', encoding='utf-8') as file_to_read:
    while True:
        time.sleep(4)
        line = file_to_read.readline()
        parts = line.split(",")
        print(line)
        # print(parts)
        if not line:
            break
        fwrite = open(path+"topchinaz/"+parts[0].strip('\n')+".txt",'a+', encoding='utf-8')
        s=requests.session()
        s.keep_alive = False
        r = s.get(weburl+parts[1])
        r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, 'html.parser')
        # print(soup.get_text())
        listtitile = soup.find(name='ul', class_='listCentent')
        tagtitle = listtitile.find_all(name='li')
        for title in tagtitle:
            item = title.find(name='h3', class_ = 'rightTxtHead')
            itemabstract = item.find(name= 'a').get_text()
            itemurl = item.find(name='span').get_text()
            fwrite.write(parts[0]+","+itemurl+","+itemabstract + '\n')
        parturl = parts[1].split(".")[0]
        pattern = parturl + "_(\d+).html"
        # print(pattern)
        su = re.compile(pattern)
        numresult = su.findall(r.text)
        print(numresult)
        maxnum = 0
        for num in numresult:
            if int(num) > maxnum:
                maxnum = int(num)

        for i in range(2,maxnum+1):
            time.sleep(3)
            urlparts = parts[1].split(".")
            s=requests.session()
            s.keep_alive = False
            print(weburl+urlparts[0]+"_"+str(i) + ".html")
            r = s.get(weburl+urlparts[0]+"_"+str(i) + ".html")
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, 'html.parser')
            sub_listtitile = soup.find(name='ul', class_='listCentent')
            sub_tagtitle = sub_listtitile.find_all(name='li')
            for sub_title in sub_tagtitle:
                sub_item = sub_title.find(name='h3', class_ = 'rightTxtHead')
                sub_itemabstract = sub_item.find(name= 'a').get_text()
                sub_itemurl = sub_item.find(name='span').get_text()
                fwrite.write(parts[0]+","+sub_itemurl+","+sub_itemabstract + '\n')
fwrite.close()


