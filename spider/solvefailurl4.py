#-- coding: utf-8 --
#将爬取失败的网站重新爬取
import  requests
from bs4 import  BeautifulSoup, Comment
import re
import time
# import eventlet
import os
import json
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException

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
logpath = "../../newwebdata/relog1.txt"
messageless_log_path =  "../../newwebdata/messagelog1.txt"
if not os.path.isdir(savepath):
    os.mkdir(savepath)

logfile = open(logpath,'a+')
def makelog(logmessage):
    logfile.write(logmessage + '\n')

messagelogfile = open(messageless_log_path,'a+')
def messagelesslog(logmessage):
    messagelogfile.write(logmessage + '\n')

# option = webdriver.ChromeOptions()

option = Options()
option.add_argument('--no-sandbox')
option.add_argument('--disable-dev-shm-usage')
option.add_argument('--headless') #静默运行
option.add_argument('log-level=3')
option.add_argument('--disable-gpu')  # 禁用GPU加速,GPU加速可能会导致Chrome出现黑屏，且CPU占用率高达80%以上
browser = webdriver.Firefox(options=option)
# browser = webdriver.Chrome(options=option)
browser.implicitly_wait(time_limit)
browser.set_page_load_timeout(time_limit)

# 查询网址，爬取内容
# def requesturl(url, savefilepath):
def requesturl(url):
    print(url)
    webinfo={}  #最后保存的数据
    webtext = []    #首页内容文本
    abouttext = []  #关于页面内容文本
    aboutlist = []  # 关于页面的连接
    stopjs = """window.stop ? window.stop() : document.execCommand("Stop");"""

    def initwebinfo():
        webinfo['title'] = ""
        webinfo['description'] = ""
        webinfo['keywords'] = ""
        webinfo['webtext'] = []

    try:
        js = 'void(window.open(""));'
        browser.execute_script(js)
        time.sleep(3)
    except:
        print ("window.open("")  error")
        pass

    handles = browser.window_handles
    while len(handles)>1:
        browser.close()
        browser.switch_to.window(handles[1])
        handles = browser.window_handles
        time.sleep(2)
    try:
        browser.get(url)
        WebDriverWait(browser, time_limit, 1).until_not(EC.title_is(""))
    except   TimeoutException:
        print("TimeoutException")
        browser.execute_script(stopjs) #   超时停止js脚步
    except UnexpectedAlertPresentException:
        print("UnexpectedAlertPresentException")
        time.sleep(5)
        element = browser.switch_to.active_element
        element.click()
    # r.encoding = r.apparent_encoding

    time.sleep(3)
    initwebinfo()
    soup = BeautifulSoup(browser.page_source, 'html.parser')

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

    #信息太少可能有跳转等待 重新获取
    if len(webinfo['webtext'])<15:
        time.sleep(65)
        initwebinfo()
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        get_info()
    
    
    skip_text = ['点击','跳转','进入']
    href_text = ['index', 'main','default','info','home']
    #数据太少  找到所有的a标签 选择合适的访问
    if len(webinfo['webtext'])<15:
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        atag = soup.find_all('a')
        js1 = """var ipt = document.getElementsByTagName("a");
                for (i = 0; i<ipt.length; i ++){
                    if (ipt[i].innerText.trim() == arguments[0]){
                        ipt[i].target = "_self";
                        ipt[i].click();
                        return ;
                    }
                }"""
        # 点击href符合的链接
        js2 = """var ipt = document.getElementsByTagName("a");
                for (i = 0; i<ipt.length; i ++){
                    if (ipt[i].getAttribute('href').trim() == arguments[0]){
                        ipt[i].target = "_self";
                        ipt[i].click();
                        return ;
                    }
                }"""
        for tag in atag:
            if len(webinfo['webtext'])<15:
                tmpbool = True
                if tag.get_text():
                    for keyword in skip_text:   #访问可能的跳转页面
                        if keyword in tag.get_text():
                            tmpbool = False
                            try:
                                browser.execute_script(js1,tag.get_text().strip())
                                time.sleep(10)
                                browser.switch_to.alert.accept()
                                time.sleep(3)
                                break
                            except Exception as e:
                                print(e)
                                pass
                            soup = BeautifulSoup(browser.page_source, 'html.parser')
                            get_info()
                if tmpbool:
                    tmpurl = url.replace("http://","",1)
                    tmpurl = tmpurl.replace("www.","",1)
                    for keyword in href_text:
                        if tag.has_attr('href'):
                            if tmpurl in tag['href'] and keyword in tag['href']:
                                try:
                                    browser.execute_script(js2,tag['href'].strip())
                                    time.sleep(8)
                                    browser.switch_to.alert.accept()
                                    time.sleep(3)
                                except Exception as e:
                                    print(e)
                                    pass
                                soup = BeautifulSoup(browser.page_source, 'html.parser')
                                get_info()
                                break
    #找input 
    if len(webinfo['webtext'])<15:
        print("find input")
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        inputag = soup.find_all('input')
        print(inputag)
        js1 = """var ipt = document.getElementsByTagName("input");
                for (i = 0; i<ipt.length; i ++){
                    if (ipt[i].name == arguments[0]){
                        ipt[i].target = "_self";
                        ipt[i].click();
                        return ;
                    }
                }"""
        for tag in inputag:
            # print( str(tag))
            print(tag['name'])
            tmpbool=True
            if tmpbool:
                if tag.has_attr('name') :
                    for keyword in href_text:   #访问可能的跳转页面
                        if keyword in str(tag):
                            try:
                                print(tag)
                                browser.execute_script(js1,tag['name'].strip())
                                time.sleep(10)
                            except Exception as e:
                                print(e)
                                pass
                            soup = BeautifulSoup(browser.page_source, 'html.parser')
                            get_info()
                            tmpbool=False
                            break
            else:
                break
    #  可能有frame 寻找全部frame
    while True:
        try:
            i = 0
            while  True:
            # while  len(webinfo['webtext'])<15:
                browser.switch_to.frame(i)
                i=i+1
                soup = BeautifulSoup(browser.page_source, 'html.parser')
                get_info()
                browser.switch_to.default_content()
        except:
            print("frame error")
            browser.switch_to.default_content()
            break

# 第二阶段
    #寻找是否存在介绍该网站的链接 如 关于我们 公司简介 等
    def havekey(tag):
        if  tag.has_attr('href') or  tag.has_attr('data-href'):
            if tag.string != None and len(tag.string.strip()) < 8:
                searchObj = re.search("关于.{0,4}", tag.string, flags=0)
                if searchObj:
                    return True
                searchObj = re.search(".{0,3}简介", tag.string, flags=0)
                if searchObj:
                    return True
                searchObj = re.search(".{0,3}概况", tag.string, flags=0)
                if searchObj:
                    return True
                searchObj = re.search(".{0,3}介绍", tag.string, flags=0)
                if searchObj:
                    return True
                searchObj = re.search("了解[\u4e00-\u9fa5]{1,3}", tag.string, flags=0)
                if searchObj:
                    return True
    soup = BeautifulSoup(browser.page_source, 'html.parser')
    about = soup.find_all(havekey)
    # 寻找关于页面的链接
    for href in about:
        if href.string !=None:
            aboutlist.append(href.string.strip())
        elif href.has_attr('title'):
            aboutlist.append(href['title'].strip())
    aboutlist = list(set(aboutlist))    #去重

    if len(aboutlist)>0:
        aboutcount = min(len(aboutlist),3)
        js = """var ipt = document.getElementsByTagName("a");
            for (i = 0; i<ipt.length; i ++){
                if (ipt[i].innerText.trim() == arguments[0]){
                    ipt[i].target = "_self";
                    ipt[i].click();
                    return ;
                }
                if ( ipt[i].hasAttribute('title') ){
                    if (ipt[i].getAttribute('title').trim() == arguments[0] ){
                    ipt[i].target = "_self";
                    ipt[i].click();
                    return ;
                }
               }
            }
            var ipt1 = document.getElementsByTagName("li");
            for (i = 0; i<ipt1.length; i ++){
                if (ipt1[i].innerText.trim() == arguments[0]){
                    ipt1[i].target = "_self";
                    ipt1[i].click();
                    return;
                }
            } 
            """
        for i in range(0, aboutcount):
            browser.execute_script(js,aboutlist[i])
            time.sleep(5)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            script = [s.extract() for s in soup('script')]
            style = [s.extract() for s in soup('style')]
            for element in soup(text = lambda text: isinstance(text, Comment)):
                element.extract()
            # if webinfo['title'] == "" or webinfo['title'] == None:
            get_headtext()
            [s.extract() for s in soup('head')]
            for textstring in soup.stripped_strings:
                if len(repr(textstring))>8:
                    abouttext.append(repr(textstring))
            try:
                browser.back()
            except   TimeoutException:
                print("step2 TimeoutException")
                browser.execute_script(stopjs)
        webinfo['abouttext'] = abouttext
    else:
        webinfo['abouttext'] = []

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

readpath1 = "../../topchinaz2/"
fs1 =  os.listdir(readpath1)
fs1 = fs1[::-1]
for filename in fs1:
    
        filepath = readpath1 + filename
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
