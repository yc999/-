#-- coding: utf-8 --
#将爬取失败的网站重新爬取 使用httprequest
import  requests
from bs4 import  BeautifulSoup, Comment
import re
import time
# import eventlet
import os
import json
from urllib.parse import urljoin
from hyper.contrib import HTTP20Adapter


badtitles=[
    '资源不存在',    '找不到',      '页面载入出错',           '临时域名',           '阻断页',
    '建设中',       '访问出错',     '出错啦',                'IIS7',            '温馨提示',    
     '无法找到',    '未被授权查看',  '已暂停' ,             '详细错误',              '禁止访问',         
    '载入出错',         '没有找到',     '无法显示',           '无法访问',       '升级维护中',          
    '正在维护',         '配置未生效',   '访问报错',          '域名停靠',        '网站访问报错',
     '服务器错误',      '不再可用',     '错误',                   '错误提示',   '发生错误', 
        '非法阻断',     '链接超时',     '站点不存在',         '系统发生错误',   '网站欠费提醒'
     'Not Found',               'Welcome to nginx',              'Suspended Domain',
    'IIS Windows',              'Invalid URL',                  '400 Unknown Virtual Host',
     'Server Error',            '403 Forbidden',                'Temporarily Unavailable', 
      'Database Error',           'temporarily unavailable',     'Bad gateway',        
         'error Page',             'Internal Server Error',      '405',   
    'Service Unavailable',          'Access forbidden',         'System Error',
      '404 Not Found',               'null',                        'Error',   
       'Connection timed out',      'TestPage',                     'Test Page',
        'Bad Gateway',  'Bad Request',      'Time-out',                 'No configuration',     
        '403 Frobidden','ACCESS DENIED','Problem loading page']


headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        } 

# 判断标题是否正常 
# mytitle 需要判断的title 
# 正常返回 False 不正常返回 True
def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False
    
# 最多爬取的页面数量
maxwebpage = 4
# 读取网页url
readpath = "../../topchinaz1/"
# readpath = "D:/dnswork/sharevm/topchinaz/"
# readpath = "E:/webdata/"           

# 保存从chinaz所有网站的内容
savepath = "../../httpwebdata/"
# savepath = "E:/webdata/"
savedir = "" # 类别文件夹
logpath = "E:/webdata/relog.txt"

# logpath = "../../newwebdata/relog.txt"
# messageless_log_path =  "../../newwebdata/messagelog.txt"
# if not os.path.isdir(savepath):
#     os.mkdir(savepath)


# 写入文件
def writeurlfile(url, data):
    path  = savepath + savedir +"/" 
    if not os.path.isdir(path):
        os.mkdir(path)
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
    if response.encoding == ['gb2312']:
        response.encoding = 'GBK'
    if response.encoding == ['gbk']:
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
    soup = BeautifulSoup(response.text, 'html.parser')
    havegetlist.append(url_now)
    havegetcount += 1
    havegetcount = findaboutwebpage(url_now,soup, havegetcount)
    # 获取网页head中元素 title keywords description 存入webinfo中
    def get_headtext():
        # soup = BeautifulSoup(r.text, 'html.parser')  soup = BeautifulSoup(browser.page_source, 'lxml')
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
        soup = BeautifulSoup(response.text, 'html.parser')
        atag = soup.find_all('a')
        # 点击href符合的链接
        for tag in atag:
            if tag.has_attr('href') and tag.get_text():
                for keyword in skip_text:   #访问可能的跳转页面
                    if keyword in tag.get_text():
                        try:
                            next_url = urljoin(url_now, tag['href'])
                        except:
                            continue
                        if samewebsite(url_now, next_url) and next_url not in havegetlist and havegetcount < maxwebpage: # 需要和当前url一致
                            next_response = get_and_add(next_url, webdata, havegetcount)
                            if next_response == False:
                                continue
                            abouturl = next_response.url          # 当前的url
                            havegetlist.append(abouturl)
                            havegetcount += 1
                            if next_url != abouturl:
                                havegetlist.append(next_url)
                            tmpsoup = BeautifulSoup(next_response.text, 'html.parser')
                            havegetcount = findaboutwebpage(abouturl, tmpsoup, havegetcount)
                            break
                tmpurl = url_now.replace("http://","",1)
                tmpurl = tmpurl.replace("https://","",1)
                tmpurl = tmpurl.replace("www.","",1)
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
                        havegetlist.append(abouturl)
                        havegetcount += 1
                        if next_url != abouturl:
                            havegetlist.append(next_url)
                        soup = BeautifulSoup(next_response.text, 'html.parser')
                        havegetcount = findaboutwebpage(abouturl, soup, havegetcount)
    print(havegetcount)
    writeurlfile(url, webdata)
    return True

target_path1 = "/home/jiangy2/dnswork/httpwebdata/"
target_filelist =  os.listdir(target_path1)
saveurl = []
readpath = "/home/jiangy2/dnswork/topchinaz1/"
fs = os.listdir(readpath)   #读取url目录

tmpfs = []
for filename in fs:
    if filename.replace('.txt','') not in target_filelist:
            tmpfs.append(filename)

fs = tmpfs 


for filename in fs:
    filepath = readpath + filename
    print(filepath)
    f = open(filepath,"r",encoding="utf-8")
    urlList = f.readlines()                 #   所有待爬取的url
    f.close()
    savedir = filename.split(".")[0]
    dirpath = savepath + savedir        # 爬取的数据保存的文件夹路径
    isExists=os.path.exists(dirpath)    #根据url文件创建文件夹
    if not isExists:
        os.makedirs(dirpath)
    savedfileslist = os.listdir(dirpath)    #所有成功爬取的url文件名,需要对文件名处理。

    for url in urlList:
        time.sleep(15)
        try:
            url = "".join(url.split())
            url = url.split(",")[1]
        except:
            continue
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
                    

