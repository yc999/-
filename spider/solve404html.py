#将爬取失败的网站重新爬取
import  requests
from bs4 import  BeautifulSoup, Comment
import re
import time
import eventlet
import os
import json
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException

badtitle=['404', '找不到', '', 'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7']

# 保存从chinaz所有网站的内容
savepath = "E:/webdata/"
logpath = "E:/webdata/relog.txt"
logfile = open(logpath,'w')
def makelog(logmessage):
    logfile.write(logmessage + '\n')

eventlet.monkey_patch(time=True)
time_limit = 40  #set timeout time 3s
# @timeout_decorator.timeout(30)
headers = {
'user-agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3573.0 Safari/537.36',
}
 # option = webdriver.ChromeOptions()

option = Options()
option.add_argument('--no-sandbox')
# option.add_argument("--window-size=1920,1080")
option.add_argument('--disable-dev-shm-usage')
option.add_argument('--headless') #静默运行
option.add_argument('--disable-gpu')  # 禁用GPU加速,GPU加速可能会导致Chrome出现黑屏，且CPU占用率高达80%以上
browser = webdriver.Firefox(options=option)
# browser = webdriver.Chrome(options=option)
browser.implicitly_wait(time_limit)
browser.set_page_load_timeout(time_limit)

# 查询网址，爬取内容
def requesturl(url, savefilepath):
    print(url)
    webinfo={}  #最后保存的数据
    webtext = []    #首页内容文本
    abouttext = []  #关于页面内容文本
    aboutlist = []  # 关于页面的连接
    webinfo['title'] = ""
    webinfo['description'] = ""
    webinfo['keywords'] = ""
    try:
        browser.get(url)
        WebDriverWait(browser, time_limit, 0.5).until_not(EC.title_is(""))
    except   TimeoutException:
        browser.execute_script('window.stop()')
    # r = s.get(httpsurl,timeout=15,headers=headers)
    # r.encoding = r.apparent_encoding
    soup = BeautifulSoup(browser.page_source, 'html.parser')
    def get_headtext():
        # soup = BeautifulSoup(r.text, 'html.parser')
        # soup = BeautifulSoup(browser.page_source, 'lxml')
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]
        for element in soup(text = lambda text: isinstance(text, Comment)):
            element.extract()
        head = soup.head
        if webinfo['title'] == "":
            try:
                webinfo['title'] = head.title.string
            except:
                webinfo['title'] = ""
                pass
        if webinfo['description'] == "":
            try:
                webinfo['description'] = head.find('meta',attrs={'name':'description'})['content']
            except:
                webinfo['description'] = ""
                pass
        if webinfo['description'] == "":
            try:
                webinfo['description'] = head.find('meta',attrs={'name':'Description'})['content']
            except:
                webinfo['description'] = ""
                pass
        if webinfo['description'] == "":
            try:
                webinfo['description'] = head.find('meta',attrs={'name':'DESCRIPTION'})['content']
            except:
                webinfo['description'] = ""
                pass
        if webinfo['keywords'] == "":
            try:
                webinfo['keywords'] = head.find('meta',attrs={'name':'keywords'})['content']

            except:
                webinfo['keywords'] = ""
                pass
        if webinfo['keywords'] == "":
            try:
                webinfo['keywords'] = head.find('meta',attrs={'name':'Keywords'})['content']

            except:
                webinfo['keywords'] = ""
                pass
        if webinfo['keywords'] == "":
            try:
                webinfo['keywords'] = head.find('meta',attrs={'name':'KEYWORDS'})['content']
            except:
                webinfo['keywords'] = ""
                pass
    def get_bodytext():
        for textstring in soup.stripped_strings:
            if len(repr(textstring))>4:
                webtext.append(repr(textstring))
        webinfo['webtext'] = webtext


    #获取信息
    get_headtext()
    [s.extract() for s in soup('head')]
    get_bodytext()
    #信息太少等待
    if len(webtext)<15:
        time.sleep(69)
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        get_headtext()
        [s.extract() for s in soup('head')]
        get_bodytext()
    # 信息太少寻找一次frame
    if len(webtext)<15:
        try:
            browser.switch_to.frame(0)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            get_headtext()
            [s.extract() for s in soup('head')]
            get_bodytext()
        except:
            pass
    if len(webtext)<15:
        makelog(url + " too less message")
    def havekey(tag):
        # if tag.name == "a" and tag.has_attr('href'):
            if tag.string != None and len(tag.string.strip()) < 6:
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
    about = soup.find_all(havekey)
    for href in about:
        # print(href.string)
        aboutlist.append(href.string)

    if len(aboutlist)>0:
        js = """var ipt = document.getElementsByTagName("a");
        for(x in ipt){
            if (ipt[x].innerHTML == arguments[0]){
                ipt[x].target = "_self";
                ipt[x].click();
            }
        }"""
        browser.execute_script(js,aboutlist[0])
        WebDriverWait(browser, time_limit, 0.5).until_not(EC.title_is(""))
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]
        for element in soup(text = lambda text: isinstance(text, Comment)):
            element.extract()
        get_headtext()
        [s.extract() for s in soup('head')]
        for textstring in soup.stripped_strings:
            if len(repr(textstring))>8:
                abouttext.append(repr(textstring))
                # print(repr(textstring))
        webinfo['abouttext'] = abouttext
    else:
        webinfo['abouttext'] = []
 #将数据写入文件
    f = open(savefilepath, "w",encoding="utf-8")
    f.write(json.dumps(webinfo,ensure_ascii=False))
    f.close()#关闭文件


good_result1=[]
bad_result = []
else_result=[]

readpath = "D:/dnswork/sharevm/topchinaz/"
# readpath = "E:/webdata/"
fs = os.listdir(readpath)   #读取url目录
for filename in fs:
    dirname = filename.split(".")[0]
    dirpath = savepath + dirname        # 爬取的数据保存的文件夹路径
    isExists=os.path.exists(dirpath)    #根据url文件创建文件夹
    if not isExists:
        os.makedirs(dirpath)
    savedfileslist = os.listdir(dirpath)    #所有成功爬取的url,需要对文件名处理。
    for savedfilename in savedfileslist:
        time.sleep(1)
        savefilepath = dirpath +"/" + savedfilename
        print(savefilepath)
        fdata = open(savefilepath,"r",encoding="utf-8")

        filedata = json.loads(fdata.read())
        def failwebdata(jsondata):
            jsondata['title']
        if failwebdata(filedata):
            
    #     output = open(path + "1.result/"+ url + ".txt","w",encoding='utf-8')
            # try:
            #         httpsurl =  'http://' + url
            #         if url.split(".")[0]!="www":
            #             httpsurl = 'http://www.' + url
            #         requesturl(httpsurl, savefilepath)
            #         good_result1.append(url)
            # except:
            #     try:
            #         httpsurl =  'http://' + url

            #         if url.split(".")[0]!="www":
            #             httpsurl = 'http://www.' + url
            #         requesturl(httpsurl, savefilepath)
            #         good_result1.append(url)
            #         httpsurl =  'http://www.' + url
            #     except Exception as e:
            #         print (e)
            #         print("fail ", url)
            #         # makelog (str(e))
            #         makelog("fail "+ url)
            #         bad_result.append(url)
browser.quit()

path = "E:/webdata/"
websit = "regood_result1"
filename = path + websit + ".txt"
f = open(filename,'w')
for item in good_result1:
        f.write(item + '\n')
f.close()

websit = "rebad_result1"
filename = path + websit + ".txt"
f = open(filename,'w')
for item in bad_result:
        f.write(item + '\n')
f.close()


websit = "reelse_result"
filename = path + websit + ".txt"
f = open(filename,'w')
for item in else_result:
        f.write(item + '\n')
f.close()