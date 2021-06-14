#-- coding: utf-8 --
import  requests
import re
# import eventlet
import os
import sys
import io
import json
import numpy as np
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
import random
import mytool
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *
from bs4 import  BeautifulSoup, Comment
import re
import time
import os
import json
from urllib.parse import urljoin
from hyper.contrib import HTTP20Adapter


headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        } 
maxwebpage = 4
savepath = "/home/jiangy2/dnswork/pdnswebdata/"
savedfileslist = os.listdir(savepath)    #所有成功爬取的url文件名,需要对文件名处理。

# 写入文件
def writeurlfile(url, data):
    path  = savepath
    tmpurl = url.replace('http://','',1)
    tmpurl = tmpurl.replace('https://','',1)

    urllfile = open(path + tmpurl +".txt",'w',encoding='utf-8')
    urllfile.write(json.dumps(data, ensure_ascii=False))

#判断两个url是否是同一个网站
# sourceurl 原url
# targeturl 要判断的url
def samewebsite(sourceurl, targeturl):
    tmp_sourceurl = sourceurl.replace('http://','')
    tmp_sourceurl = tmp_sourceurl.replace('https://','')
    tmp_sourceurl = tmp_sourceurl.replace('www.','')
    tmp_sourceurl = tmp_sourceurl.split('/')[0].strip()

    tmp_targeturl = targeturl.replace('http://','')
    tmp_targeturl = tmp_targeturl.replace('https://','')
    tmp_targeturl = tmp_targeturl.split('/')[0].strip()
    if tmp_sourceurl in tmp_targeturl:
        print(sourceurl, targeturl)
        return True
    return False



#寻找是否存在介绍该网站的链接 如 关于我们 公司简介 等
def havekey(tag):
    if  tag.has_attr('href') or  tag.has_attr('data-href'):
        if tag.string != None and len(tag.string.strip()) < 8:
            searchObj = re.search("关于.{0,8}", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search(".{0,4}简介", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search(".{0,4}概况", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search(".{0,4}介绍", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("了解.{0,8}", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("走近.{0,8}", tag.string, flags=0)
            if searchObj:
                return True


def solvehref(href):
    tmps = href.split('"')
    tmp = tmps[1]
    tmp = tmp.replace('http://','')
    tmp = tmp.replace('https://','')
    tmps = tmp.split('/')
    tmp = tmps[0]
    return tmp


def return_all_url(url):
    allatags = []
    try:
        r = requests.get(url)
        r.encoding = r.apparent_encoding
        # r.encoding = "utf-8"
        soup = BeautifulSoup(r.text, 'html.parser')
        pattern = re.compile("href\s{0,3}=\s{0,3}\"[^\"]+\"")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solvehref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
    except Exception as e:
        print(e)
        pass
    return allatags


#请求url并且加入字典
def get_and_add(url,webdata, webcount):
    requests.packages.urllib3.disable_warnings()
    sessions=requests.session()
    sessions.keep_alive = False
    sessions.mount(url, HTTP20Adapter())
    try:
        response = requests.get(url,verify=False,allow_redirects=True,headers = headers, timeout=30)
    except Exception as e:
        print(e)
        # sessions.close()
        return False
    response.encoding = requests.utils.get_encodings_from_content(response.text)
    if response.encoding == ['gbk2312']:
        response.encoding = 'GBK'
    elif response.encoding == ['gb2312']:
        response.encoding = 'GBK'
    elif response.encoding == ['gbk']:
        response.encoding = 'GBK'
    elif response.encoding == ['GBK']:
        response.encoding = 'GBK'
    elif type(response.encoding) == list:
        if 'gb2312' in response.encoding:
            response.encoding = 'GBK'
        elif 'gbk' in response.encoding:
            response.encoding = 'GBK'
        elif 'gbk2312' in response.encoding:
            response.encoding = 'GBK'
    if response.status_code != 200:
        # sessions.close()
        return False
    # sessions.close()
    webdata[url] = response.text
    # writeurlfile(url, response.text)
    print(webcount)
    return response


def requesturl(url):
    print(url)
    webinfo={}  # 最后保存的数据
    havegetlist = [] # 已经访问过的网页
    webdata = {} # 保存网页数据
    havegetcount = 0
    #找到当前的相关介绍页面
    def findaboutwebpage(abouturl, soup, count):
        about = soup.find_all(havekey)
        # 寻找关于页面的链接
        aboutlist = []
        for tag in about:
            if tag.has_attr('href'):
                try:
                    tmpurl = urljoin(abouturl, tag['href'])
                except:
                    continue
            elif tag.has_attr('data-href'):
                try:
                    tmpurl = urljoin(abouturl, tag['data-href'])
                except:
                    continue
            if tmpurl not in havegetlist and samewebsite(abouturl, tmpurl) and count < maxwebpage:
                next_response = get_and_add(tmpurl, webdata, count)
                if next_response != False:
                    url_now_tmp = next_response.url          # 当前的url
                    # 加入已爬队列
                    havegetlist.append(abouturl)
                    if url_now_tmp != abouturl:
                        havegetlist.append(url_now_tmp)
                    aboutlist.append(url_now_tmp)
                    count += 1
                    if len(aboutlist)>2:
                        break
        return count
    response = get_and_add(url, webdata, havegetcount)
    if response == False:
        return False
    
    def initwebinfo():
        webinfo['title'] = ""
        webinfo['description'] = ""
        webinfo['keywords'] = ""
        webinfo['webtext'] = []

    initwebinfo()
    url_now = response.url          # 当前的url
    try:
        soup = BeautifulSoup(response.text, 'html.parser')
    except:
        return False
    havegetlist.append(url_now)
    havegetcount += 1
    havegetcount = findaboutwebpage(url_now,soup, havegetcount)
    # 获取网页head中元素 title keywords description 存入webinfo中
    def get_headtext():
            [s.extract() for s in soup('script')]
            [s.extract() for s in soup('style')]
            for element in soup(text = lambda text: isinstance(text, Comment)):
                element.extract()
            head = soup.head
            if webinfo['title'] == "" or webinfo['title'] == None:
                try:
                    webinfo['title'] += head.title.string.strip()
                except:
                    pass
            try:
                webinfo['description'] += head.find('meta',attrs={'name':'description'})['content']
            except:
                pass
            try:
                webinfo['description'] += head.find('meta',attrs={'name':'Description'})['content']
            except:
                pass
            try:
                webinfo['description'] += head.find('meta',attrs={'name':'DESCRIPTION'})['content']
            except:
                pass
            try:
                webinfo['keywords'] += head.find('meta',attrs={'name':'keywords'})['content']
            except:
                pass
            try:
                webinfo['keywords'] += head.find('meta',attrs={'name':'Keywords'})['content']
            except:
                pass
            try:
                webinfo['keywords'] += head.find('meta',attrs={'name':'KEYWORDS'})['content']
            except:
                pass
    def get_bodytext():
        for textstring in soup.stripped_strings:
            if len(repr(textstring))>4:
                webinfo['webtext'].append(repr(textstring))
        # webinfo['webtext'] += webtext
    def get_info():
        get_headtext()
        [s.extract() for s in soup('head')]
        get_bodytext()

    
    # 第一阶段
    #开始获取信息
    get_info()
    # 可能是欢迎页
    skip_text = ['点击','跳转','进入']
    href_text = ['index', 'main','default','info','home']
    #数据太少  找到所有的a标签 选择合适的访问
    if True:
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
        except:
            pass
        else:
            atag = soup.find_all('a')
            # 点击href符合的链接
            for tag in atag:
                if tag.has_attr('href') and tag.get_text():
                    for keyword in skip_text:   #访问可能的跳转页面
                        if keyword in tag.get_text():
                            try:
                                next_url = urljoin(url_now, tag['href'])
                            except:
                                break
                            if samewebsite(url_now, next_url) and next_url not in havegetlist and havegetcount < maxwebpage: # 需要和当前url一致
                                next_response = get_and_add(next_url, webdata, havegetcount)
                                if next_response == False:
                                    continue
                                abouturl = next_response.url          # 当前的url
                                if next_url != abouturl:
                                    havegetlist.append(next_url)
                                havegetlist.append(abouturl)
                                try:
                                    tmpsoup = BeautifulSoup(next_response.text, 'html.parser')
                                except:
                                    continue
                                havegetcount += 1
                                havegetcount = findaboutwebpage(abouturl, tmpsoup, havegetcount)
                                break
            atag = return_all_url(soup)
            for tag in atag:
                tmpurl = url_now.replace("http://","",1)
                tmpurl = tmpurl.replace("https://","",1)
                tmpurl = tmpurl.replace("www.","",1)  # 
                for keyword in href_text:
                    try:
                        next_url = urljoin(url_now, tag['href']) #寻找可能的相关链接
                    except:
                        continue
                    if tmpurl in next_url and keyword in next_url and next_url not in havegetlist and samewebsite(url_now, next_url) and havegetcount < maxwebpage:
                        next_response = get_and_add(next_url, webdata, havegetcount)
                        if next_response == False:
                            continue
                        abouturl = next_response.url          # 当前的url
                        if next_url != abouturl:
                                havegetlist.append(next_url)
                        havegetlist.append(abouturl)
                        try:
                            tmpsoup = BeautifulSoup(next_response.text, 'html.parser')
                        except:
                            continue
                        havegetcount += 1
                        havegetcount = findaboutwebpage(abouturl, tmpsoup, havegetcount)
        # print(havegetcount)
    writeurlfile(url, webdata)
    return True


dnstpye_value = {'1' : "A", '5':"CNAME", '28':"AAAA"}

# 读取dns数据
# dnsdata_path = "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2020-11/pdns_data"
dnsdata_path = "/home/jiangy2/dnswork/cdnlist/pdns_data"
dnsdata_file = open(dnsdata_path, 'r', encoding='utf-8')

saveurl = []

while True:
    line = dnsdata_file.readline()
    if  line:
        try:
            dnsdata = mytool.prasednsdata(line)
        except:
            continue
        if dnsdata['Dnstype'] not in dnstpye_value: # 只取 A AAAA CNAME记录
            continue
        # print(dnsdata)
        url = mytool.getrkey_domainname(dnsdata['rkey'])
        tmpurl = url.replace('www.','',1)
        if tmpurl not in saveurl:
            if url + ".txt" not in savedfileslist and "www." + url + ".txt" not in savedfileslist:
                httpurl =  'http://' + url
                resultdata = requesturl(httpurl)
                if resultdata == False:
                    if url.split(".")[0]!="www":
                        httpurl = 'http://www.' + url
                    else:
                        httpurl = 'http://' + url.replace('www.','',1)
                    resultdata = requesturl(httpurl)
                    if resultdata == True:
                        saveurl.append(tmpurl)
                else:
                    saveurl.append(tmpurl)
            else:
                saveurl.append(tmpurl)
    else:
        break
