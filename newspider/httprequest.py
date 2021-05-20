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



badtitles=['404 Not Found', '找不到',  'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7','Default','已暂停' ,'Server Error','403 Forbidden','禁止访问','载入出错','没有找到',
'无法显示','无法访问','Bad Gateway','正在维护','配置未生效','访问报错','Welcome to nginx','Suspended Domain',
'IIS Windows','Invalid URL','服务器错误','400 Unknown Virtual Host','无法找到','资源不存在',
'Temporarily Unavailable','Database Error','temporarily unavailable','Bad gateway','不再可用','error Page',
'Internal Server Error','升级维护中','Service Unavailable','站点不存在','405','Access forbidden','System Error',
'详细错误','页面载入出错','Error','错误','Connection timed out','域名停靠','网站访问报错','错误提示','临时域名',
'未被授权查看','Test Page','发生错误','非法阻断','链接超时','403 Frobidden','建设中','访问出错','出错啦','ACCESS DENIED','系统发生错误','Problem loading page']

time_limit = 40  #set timeout time 3s

# 判断标题是否正常 
# mytitle 需要判断的title 
# 正常返回 False 不正常返回 True
def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False
    
                            

# 保存从chinaz所有网站的内容
# savepath = "E:/webdata/"
# logpath = "E:/webdata/relog.txt"
savepath = "../../newwebdata/"
logpath = "../../newwebdata/relog.txt"
messageless_log_path =  "../../newwebdata/messagelog.txt"
if not os.path.isdir(savepath):
    os.mkdir(savepath)

logfile = open(logpath,'a+')
def makelog(logmessage):
    logfile.write(logmessage + '\n')

messagelogfile = open(messageless_log_path,'a+')
def messagelesslog(logmessage):
    messagelogfile.write(logmessage + '\n')


def writeurlfile(url,data):
    urllfile = open(savepath + url,'w')
    urllfile.write(data + '\n')

#判断两个url是否是同一个网站
# sourceurl 原url
# targeturl 要判断的url
def samewebsite(sourceurl, targeturl):
    tmp_sourceurl = sourceurl.replace('http://','')
    tmp_sourceurl = tmp_sourceurl.replace('https://','')
    tmp_sourceurl = tmp_sourceurl.split('/')[0].strip()
    tmp_targeturl = targeturl.replace('http://','')
    tmp_targeturl = tmp_targeturl.replace('https://','')
    tmp_targeturl = tmp_targeturl.split('/')[0].strip()
    if tmp_targeturl == tmp_sourceurl:
        return True
    return False




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


#请求url并且写入文件
def get_and_write(url):
    try:
        response = requests.get(url,verify=False,allow_redirects=True,headers = headers)
    except Exception as e:
        print(e)
        return False
    response.encoding = response.apparent_encoding
    writeurlfile(url, response.text)
    return response


# urls = return_all_url("http://sina.com")
# print(urls)
# 查询网址，爬取内容
# def requesturl(url, savefilepath):
def requesturl(url):
    print(url)
    webinfo={}  # 最后保存的数据
    aboutlist = []  # 关于页面的连接

    response = get_and_write(url)
    if response == False:
        return
    
    # re_text=response.text
    # re_content=response.content
    url_now = response.headers['Url-Hash'] # 当前的url
    url_now = response.url          # 当前的url

    def initwebinfo():
        webinfo['title'] = ""
        webinfo['description'] = ""
        webinfo['keywords'] = ""
        webinfo['webtext'] = []

    initwebinfo()
    soup = BeautifulSoup(response.text, 'html.parser')

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
    # 如果是无效网站提前返回
    if ifbadtitle(webinfo['title']):
        return webinfo

    
# 欢迎页
    skip_text = ['点击','跳转','进入']
    href_text = ['index', 'main','default','info','home']
    #数据太少  找到所有的a标签 选择合适的访问
    if len(webinfo['webtext'])<15:
        soup = BeautifulSoup(response.text, 'html.parser')
        atag = soup.find_all('a')
        # 点击href符合的链接
        for tag in atag:
            tmpbool = True
            if tag.get_text():
                for keyword in skip_text:   #访问可能的跳转页面
                    if keyword in tag.get_text():
                        next_url = urljoin(url, tag['href'])
                        if samewebsite(url_now, next_url): # 需要和当前url一致
                            next_response = get_and_write(url)
                            if next_response == False:
                                break
                            tmpbool = False
                            url_now = response.headers['Url-Hash'] # 当前的url
                            url_now = response.url          # 当前的url
                            soup = BeautifulSoup(next_response.text, 'html.parser')
            if tmpbool:
                tmpurl = url.replace("http://","",1)
                tmpurl = tmpurl.replace("www.","",1)
                for keyword in href_text:
                    if tag.has_attr('href'):
                        next_url = urljoin(url, tag['href'])
                        if tmpurl in next_url and keyword in next_url:
                            try:
                                next_response = requests.get(next_url,verify=False,allow_redirects=True,headers = headers)
                                soup = BeautifulSoup(next_response.text, 'html.parser')
                            except Exception as e:
                                print(e)
                                break
                            

# 第二阶段
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
    # soup = BeautifulSoup(browser.page_source, 'html.parser')
    about = soup.find_all(havekey)
    # 寻找关于页面的链接
    aboutlist = []
    for href in about:
        if tag.has_attr('href'):
            tmpurl = urljoin(url, tag['href'])
        elif tag.has_attr('data-href'):
            tmpurl = urljoin(url, tag['data-href'])
        try:
            next_response = requests.get(tmpurl,verify=False,allow_redirects=True,headers = headers)
        except:
            continue
        aboutlist.append(tmpurl)

        if len(aboutlist)>2:
            break
        


            

## 结束 
    return webinfo

#将数据写入文件
def writedata(savefilepath,webinfo):
    if len(webinfo['webtext'])<15:
        messagelesslog(url + " too less message")
    f = open(savefilepath, "w",encoding="utf-8")
    f.write(json.dumps(webinfo, ensure_ascii=False))
    f.close()#关闭文件


# 读取网页url
readpath = "../../topchinaz/"
# readpath = "D:/dnswork/sharevm/topchinaz/"
# readpath = "E:/webdata/"
fs = os.listdir(readpath)   #读取url目录
for filename in fs:
    filepath = readpath + filename
    print(filepath)
    f = open(filepath,"r",encoding="utf-8")
    urlList = f.readlines()                 #   所有待爬取的url
    f.close()
    dirname = filename.split(".")[0]
    dirpath = savepath + dirname        # 爬取的数据保存的文件夹路径
    isExists=os.path.exists(dirpath)    #根据url文件创建文件夹
    if not isExists:
        os.makedirs(dirpath)
    savedfileslist = os.listdir(dirpath)    #所有成功爬取的url文件名,需要对文件名处理。

    for url in urlList:
        time.sleep(1)
        try:
            url = "".join(url.split())
            url = url.split(",")[1]
            savefilepath = dirpath +"/" + url + ".txt"
            if url + ".txt" not in savedfileslist and "www." + url + ".txt" not in savedfileslist:
                try:
                    httpsurl =  'http://' + url
                    resultdata = requesturl(httpsurl)
                    if ifbadtitle(resultdata['title']):
                        raise Exception("title error")
                    writedata(savefilepath,resultdata)
                except:
                    try:
                        if url.split(".")[0]!="www":
                            httpsurl = 'http://www.' + url
                            savefilepath = dirpath +"/www." + url + ".txt"
                        else:
                            httpsurl = 'http://' + url.replace('www.','',1)
                            savefilepath = dirpath +"/" + url.replace('www.','',1) + ".txt"
                        resultdata = requesturl(httpsurl)
                        #网页是否无法访问
                        if ifbadtitle(resultdata['title']):
                            raise Exception("title error")
                        writedata(savefilepath, resultdata)
                    except Exception as e:
                        print (e)
                        if "Reached error page" not in str(e) and "title error" not in str(e):
                            makelog(e + url)
        except:
            pass

browser.quit()
