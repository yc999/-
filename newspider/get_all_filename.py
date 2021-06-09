#-- coding: utf-8 --
# 从已经爬取的网站中检索所有能访问的网站，删除不能访问的网站

import os

target_path = "/home/jiangy2/dnswork/httpwebdata/"  # 已经爬取的网站数据的文件夹
target_filelist =  os.listdir(target_path)      # 已经爬取的网站数据的各个类别的文件夹集合

saveurl = []

readpath = "/home/jiangy2/dnswork/topchinaz1/"  # 保存所有网站的文件的文件夹
allfile_list = os.listdir(readpath)   # 读取url目录

urlcount = 0
# 保存的路径
save_path = "/home/jiangy2/dnswork/httpwebdata_list/"  
#从txt文件开始遍历
for txtfilename in allfile_list:
    tmplist = []
    txtfile = open(readpath + txtfilename,"r",encoding="utf-8")
    urlList = txtfile.readlines()  # 所有网站集合
    filename = txtfilename.replace(".txt",'')
    web_filename = os.listdir(target_path+ filename + "/") # 能访问的网站集合
    for url in urlList:     # 遍历全部的文件
        try:
            tmpurl = "".join(url.split())
            tmpurl = tmpurl.split(",")[1]
        except:
            continue
        urlcount += 1
        tmpurl = tmpurl.replace('www.','',1)
        if tmpurl not in saveurl:
            if tmpurl + '.txt' in web_filename or  'www.' + tmpurl +'.txt' in web_filename:
                tmplist.append(url)
                saveurl.append(tmpurl)
    urllfile = open(save_path + txtfilename,'w',encoding='utf-8')
    urllfile.writelines(tmplist)
    urllfile.close()

print('urlcount ',urlcount)
print('len(saveurl) ',len(saveurl))
