import  requests
from bs4 import  BeautifulSoup, Comment
import random
import re
import time
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException
s=requests.session()
s.keep_alive = False
url =  "ebrun.com"
url =  "dennis.com.cn"


headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        } 
# if url.split(".")[0]!="www":
#     httpsurl = 'https://www.' + url
# else:
httpsurl =  'http://www.' + url
httpsurl = "http://www.keke289.com/"
# httpsurl = "http://www.auchan.com.cn"
httpsurl = "http://www.crv.com.cn"
httpsurl = "http://www.spdb.com.cn"
httpsurl = "http://qjmotor.com/"
httpsurl = "http://www.kejixun.com"
httpsurl = "http://ccidnet.com"
# httpsurl = "http://ghzxedu.com"
httpsurl="http://ailab.cn"
httpsurl = "http://www.tech.ifeng.com"
httpsurl="https://sspai.com/"
httpsurl = "http://www.yezi.cn"
# http://www.itxinwen.com     http://www.ctocio.com   http://www.qqjiyu.com
httpsurl = "http://qqjiyu.com"
httpsurl = "https://www.qianzhan.com/"
# http://gz.jiaju.sina.com.cn/  www.win10go.com
httpsurl = "http://www.win10go.com"
httpsurl = "http://www.oobmedia.com"
httpsurl = "https://www.xiaoheiban.cn/"
httpsurl = "http://www.voyagemedia.com.cn/"
httpsurl = "http://gz.jiaju.sina.com.cn/"
httpsurl ="http://techweb.com.cn"
httpsurl = "https://www.wfj.com.cn/"
httpsurl ="http://www.zijinji.com/"
httpsurl  ="http://www.damao.cn/"
httpsurl = "http://www.epchina.com"
httpsurl = "http://www.cnbm.com.cn"
httpsurl="http://www.ctocio.com/"
httpsurl ="http://www.chiconysquare.com"
#http://www.wxsd.com        http://www.cdph.com.cn http://www.timewaying.com http://www.atidesoft.com
#   http://www.shansteelgroup.com
#Alert Text: None
#Message: Dismissed user prompt dialog: 百度未授权使用地图API，可能是因为您提供的密钥不是有效的百度LBS开放平台密钥，或此密钥未对本应用的百
#度地图JavaScriptAPI授权。您可以访问如下网址了解如何获取有效的密钥：http://lbsyun.baidu.com/apiconsole/key#。
httpsurl = "http://www.atidesoft.com"
httpsurl ="http://www.shansteelgroup.com"
httpsurl ="http://www.qiule.cn/"  #http://www.zgncpw.com http://www.china-spirulina.net http://gaoxiaoit.com
# http://www.zhangjiang.net www.sinceress.com  http://iquanfen.com http://www.gdmm.com
httpsurl = "http://www.china-spirulina.net"
httpsurl = "http://www.zhangjiang.net"
httpsurl = "http://www.sinceress.com"
httpsurl = "http://www.dingsheng-group.com"  #域名停靠 被攻击
httpsurl = "http://www.yinggugufen.com"

# http://www.wanlvyuanlin.com/ # flash
httpsurl = "http://hengcheng-tools.com.cn/" #!!!
httpsurl="http://www.ytmachinery.net/"
httpsurl="http://ytsanchuan.com"
# www.gzcol.com baitetc.com http://www.weili.com.cn/ 连接被重置  baid8.cn zhaokuaidi.org
httpsurl =" http://baid8.cn"
httpsurl ="http://pay.gw.com.cn" #
httpsurl="http://www.ziyang.gov.cn/" #两个frame
httpsurl ="https://mbalib.com/"
# http://www.dingsheng-group.com/  http://hengcheng-tools.com.cn/  http://www.uker.net/ http://ytsanchuan.com/  https://www.yamaha.com.cn/ 
#http://www.weili.com.cn/ https://pay.gw.com.cn/pay-mall https://www.gaopaiwood.com/ vipzhuanli.com
# http://www.solarzoom.com http://www.yuan-hang.com www.360tuan.com https://mbalib.com/ http://www.itmo.com
# http://www.egou.com  http://www.360tuan.com
httpsurl="https://www.huanbaoj.com/"
httpsurl="https://ww.8220966.com/?id=1"

print(httpsurl)
# r = s.get(httpsurl,timeout=15,headers=headers)
# print(r.text)
webinfo={}  #最后保存的数据
webtext = []    #首页内容文本
abouttext = []  #关于页面内容文本
aboutlist = []  # 关于页面的连接
webinfo['title'] = ''
webinfo['description'] = ''
webinfo['keywords'] = ''
time_limit = 40  #set timeout time 3s

# option = webdriver.ChromeOptions()
option = Options()
option.add_argument('--no-sandbox')
option.add_argument('--disable-dev-shm-usage')
# option.add_argument('--headless') #静默运行
# option.add_argument('--disable-gpu')  # 禁用GPU加速,GPU加速可能会导致Chrome出现黑屏，且CPU占用率高达80%以上
# option.add_experimental_option('excludeSwitches', ['enable-logging'])
try:
    # browser = webdriver.Chrome(options=option)
    browser = webdriver.Firefox(options=option)
    browser.set_page_load_timeout(time_limit)
    browser.implicitly_wait(time_limit)
    # browser.get(httpsurl)
    browser.get(httpsurl)
    WebDriverWait(browser, time_limit, 0.5).until_not(EC.title_is(""))
    # time.sleep(10)
except   TimeoutException:
    print("time out")
    browser.execute_script('window.stop()')
except UnexpectedAlertPresentException:
    time.sleep(5)
    element = browser.switch_to.active_element
    element.click()
    # a = browser.switch_to_alert()
    # a.accept()
    # browser.delete_all_cookies()

except Exception as e:
    print("aa")
    print (e)
    quit()

# r = s.get(httpsurl,timeout=15,headers=headers)
# r.encoding = "utf-8"
# r.encoding = r.apparent_encoding
def get_headtext():
        # soup = BeautifulSoup(r.text, 'html.parser')
        # soup = BeautifulSoup(browser.page_source, 'lxml')
        [s.extract() for s in soup('script')]
        [s.extract() for s in soup('style')]
        for element in soup(text = lambda text: isinstance(text, Comment)):
            element.extract()
        head = soup.head
        if webinfo['title'] == "" or webinfo['title'] == None:
            try:
                webinfo['title'] = head.title.string.strip()
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
    # 保存网页文本
    # [s.extract() for s in soup('head')]
def get_bodytext():
    for textstring in soup.stripped_strings:
        if len(repr(textstring))>4:
            webtext.append(repr(textstring))
    webinfo['webtext'] = webtext
# 拿到所有相关信息
def get_info():
    get_headtext()
    [s.extract() for s in soup('head')]
    get_bodytext()

# print("get web page")
time.sleep(5)
soup = BeautifulSoup(browser.page_source, 'html.parser')
# print(soup)
get_info()

if len(webtext)<15:
    time.sleep(10)
    print("messeage less")
    webtext=[]
    soup = BeautifulSoup(browser.page_source, 'html.parser')
    get_info()


skip_text = ['点击','跳转','进入']
href_text = ['index', 'main','default']
#数据太少  找到所有的a标签 访问
if len(webtext)<15:
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
        if len(webtext)<15:
            if tag.get_text():
                for keyword in skip_text:   #访问可能的跳转页面
                    if keyword in tag.get_text():
                        browser.execute_script(js1,tag.get_text().strip())
                        time.sleep(10)
                        soup = BeautifulSoup(browser.page_source, 'html.parser')
                        get_info()
                        break
            else:
                for keyword in href_text:
                    if keyword in tag['href']:
                        browser.execute_script(js2,tag['href'].strip())
                        time.sleep(10)
                        soup = BeautifulSoup(browser.page_source, 'html.parser')
                        get_info()
                        break
    

# 查找frame
if len(webtext)<15:
    try:
        i = 0
        while  len(webtext)<15:
            print ("1")
            browser.switch_to.frame(i)
            print ("2")
            # browser.switch_to.frame(1)
        # browser.switch_to.frame(0)
            i=i+1
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            # print(soup)
            get_info()
            print(webinfo)
            browser.switch_to.default_content()
    except :
        pass

print('len(webtext)')
print(len(webtext))
# print(webinfo)
def havekey(tag):
    if  tag.has_attr('href') or  tag.has_attr('data-href'):
        if tag.has_attr('title'):
            if len(tag['title'].strip()) < 8 :
                searchObj = re.search("关于.{0,4}", tag['title'], flags=0)
                if searchObj:
                    return True
                searchObj = re.search("[\u4e00-\u9fa5]{1,3}简介", tag['title'], flags=0)
                if searchObj:
                    return True
                searchObj = re.search("[\u4e00-\u9fa5]{1,3}概况", tag['title'], flags=0)
                if searchObj:
                    return True
                searchObj = re.search("[\u4e00-\u9fa5]{1,3}介绍", tag['title'], flags=0)
                if searchObj:
                    return True
                searchObj = re.search("了解[\u4e00-\u9fa5]{1,3}", tag['title'], flags=0)
                if searchObj:
                    return True
        if tag.string != None and len(tag.string.strip()) < 8 :
            searchObj = re.search("关于.{0,4}", tag.string, flags=0)
            if searchObj:
                # print(searchObj.group())
                return True
            searchObj = re.search("[\u4e00-\u9fa5]{1,3}简介", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("[\u4e00-\u9fa5]{1,3}概况", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("[\u4e00-\u9fa5]{1,3}介绍", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("了解[\u4e00-\u9fa5]{1,3}", tag.string, flags=0)
            if searchObj:
                return True

soup = BeautifulSoup(browser.page_source, 'html.parser')
about = soup.find_all(havekey)

print("about")
print(about)
# 寻找关于页面的链接
for href in about:
    if href.string !=None:
        aboutlist.append(href.string.strip())
    elif href.has_attr('title'):
        aboutlist.append(href['title'].strip())
aboutlist = list(set(aboutlist))    #去重
print('aboutlist)')
print(aboutlist)

# 如果有关于页面，点击访问
if len(aboutlist)>0:
    print('aboutlist[0]')
    print(aboutlist[0])
    aboutcount = min(len(aboutlist),3)
    print('aboutcount')
    print(aboutcount)
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
        time.sleep(10)
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        script = [s.extract() for s in soup('script')]
        style = [s.extract() for s in soup('style')]
        for element in soup(text = lambda text: isinstance(text, Comment)):
            element.extract()
        head = soup.head

        get_headtext()
        [s.extract() for s in soup('head')]
        for textstring in soup.stripped_strings:
            if len(repr(textstring))>8:
                abouttext.append(repr(textstring))
                # print(repr(textstring))
        try:
            browser.back()
        except   TimeoutException:
            browser.execute_script('window.stop()')
        except UnexpectedAlertPresentException:
            print("aa")
            time.sleep(5)
            element = browser.switch_to.active_element
            element.click()
        except Exception as e:
            print (e)
            quit()
    webinfo['abouttext']  = abouttext
else:
    webinfo['abouttext'] = []

print(webinfo)
# browser.close()



