import  requests
from bs4 import  BeautifulSoup, Comment
import random
import re
import time
import traceback
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

badtitles=['404 Not Found', '找不到',  'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7','Default','已暂停' ,'Server Error','403 Forbidden','禁止访问','载入出错','没有找到',
'无法显示','无法访问','Bad Gateway','正在维护','配置未生效','访问报错','Welcome to nginx','Suspended Domain',
'IIS Windows','Invalid URL','服务器错误','400 Unknown Virtual Host','无法找到','资源不存在',
'Temporarily Unavailable','Database Error','temporarily unavailable','Bad gateway','不再可用','error Page',
'Internal Server Error','升级维护中','Service Unavailable','站点不存在','405','Access forbidden','System Error',
'详细错误','页面载入出错','Error','错误','Connection timed out','域名停靠','网站访问报错','错误提示','临时域名',
'未被授权查看','Test Page','发生错误','非法阻断','链接超时','403 Frobidden','建设中','访问出错']
headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        } 

httpsurl =  'http://www.' + url
httpsurl = "http://www.keke289.com/"
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




# r = s.get(httpsurl,timeout=15,headers=headers)
# print(r.text)
webinfo={}  #最后保存的数据
webtext = []    #首页内容文本
abouttext = []  #关于页面内容文本
aboutlist = []  # 关于页面的连接


def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False
time_limit = 10  #set timeout time 3s

option = Options()
option.add_argument('--no-sandbox')
option.add_argument('log-level=3')
# option.add_argument('--disable-dev-shm-usage')
# option.add_argument('--headless') #静默运行
option.add_argument('--disable-gpu')  # 禁用GPU加速,GPU加速可能会导致Chrome出现黑屏，且CPU占用率高达80%以上
browser = webdriver.Firefox(options=option)
# browser = webdriver.Chrome(options=option)
browser.implicitly_wait(time_limit)
browser.set_page_load_timeout(time_limit)
# 查询网址，爬取内容
def requesturl(url):
    print(url)
    browser.delete_all_cookies()
    webinfo={}  #最后保存的数据
    webtext = []    #首页内容文本
    abouttext = []  #关于页面内容文本
    aboutlist = []  # 关于页面的连接
    def initwebinfo():
        webinfo['title'] = ""
        webinfo['description'] = ""
        webinfo['keywords'] = ""
        webinfo['webtext'] = []
    stopjs = """
                window.stop ? window.stop() : document.execCommand("Stop");
                """

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
        print('加载超过5秒，强制停止加载....')
        browser.execute_script(stopjs) #   超时停止js脚步
    except UnexpectedAlertPresentException:
        time.sleep(5)
        element = browser.switch_to.active_element
        element.click()
    # except Exception as e:
    #     print (e)


    time.sleep(5)
    initwebinfo()
    soup = BeautifulSoup(browser.page_source, 'html.parser')

    # 获取网页head中元素 title keywords description 存入webinfo中
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
                    webinfo['title'] += head.title.string.strip()
                except:
                    # webinfo['title'] = ""
                    pass
        # if webinfo['description'] == "":
            try:
                webinfo['description'] += head.find('meta',attrs={'name':'description'})['content']
            except:
                # webinfo['description'] = ""
                pass
        # if webinfo['description'] == "":
            try:
                webinfo['description'] += head.find('meta',attrs={'name':'Description'})['content']
            except:
                # webinfo['description'] = ""
                pass
        # if webinfo['description'] == "":
            try:
                webinfo['description'] += head.find('meta',attrs={'name':'DESCRIPTION'})['content']
            except:
                # webinfo['description'] = ""
                pass
        # if webinfo['keywords'] == "":
            try:
                webinfo['keywords'] += head.find('meta',attrs={'name':'keywords'})['content']
            except:
                # webinfo['keywords'] = ""
                pass
        # if webinfo['keywords'] == "":
            try:
                webinfo['keywords'] += head.find('meta',attrs={'name':'Keywords'})['content']
            except:
                # webinfo['keywords'] = ""
                pass
        # if webinfo['keywords'] == "":
            try:
                webinfo['keywords'] += head.find('meta',attrs={'name':'KEYWORDS'})['content']
            except:
                # webinfo['keywords'] = ""
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
    print(" step 1")
    #开始获取信息
    get_info()
    # print(webinfo['title'])
    print(" step 1 1")
    if ifbadtitle(webinfo['title']):
        return webinfo
    #信息太少可能有跳转等待 重新获取
    if len(webinfo['webtext'])<15:
        time.sleep(10)
        initwebinfo()
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        get_info()

    
    skip_text = ['点击','跳转','进入']
    href_text = ['index', 'main','default','info']
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
                            browser.execute_script(js1,tag.get_text().strip())
                            time.sleep(10)
                            soup = BeautifulSoup(browser.page_source, 'html.parser')
                            get_info()
                            break
                if tmpbool:
                    tmpurl = url.replace("http://","",1)
                    tmpurl = tmpurl.replace("www.","",1)
                    for keyword in href_text:
                        if tag.has_attr('href'):
                            if tmpurl in tag['href'] and keyword in tag['href']:
                                try:
                                    browser.execute_script(js2,tag['href'].strip())
                                    time.sleep(5)
                                    print("aaaa")
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
    print(" step 1 2")

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
                # print(webinfo)
                browser.switch_to.default_content()
        except :
            break

    print(" step 1 3")

# 第二阶段
    # print(webinfo['title'])

    print(" step 2")
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
    print(" step 2 1")

    for href in about:
        if href.string !=None:
            aboutlist.append(href.string.strip())
        elif href.has_attr('title'):
            aboutlist.append(href['title'].strip())
    aboutlist = list(set(aboutlist))    #去重

    print(" step 2 2  aboutlist")
    print(aboutlist)
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
            time.sleep(2)
            soup = BeautifulSoup(browser.page_source, 'html.parser')
            script = [s.extract() for s in soup('script')]
            style = [s.extract() for s in soup('style')]
            for element in soup(text = lambda text: isinstance(text, Comment)):
                element.extract()
            # if webinfo['title'] == "" or webinfo['title'] == None:
            #     print(">>>")
            get_headtext()
            [s.extract() for s in soup('head')]
            for textstring in soup.stripped_strings:
                if len(repr(textstring))>8:
                    abouttext.append(repr(textstring))
                    # print(repr(textstring))
            try:
                browser.back()
            except   TimeoutException:
                browser.execute_script(stopjs)
        webinfo['abouttext'] = abouttext
    else:
        webinfo['abouttext'] = []
    # print(webinfo['title'])
    
    print(" step 2 3")

    return webinfo

 #将数据写入文件




url = "www.zhengjimt.com"
# url = "caenet.cn"
# url = "health.china.com.cn"
# url="p2p.hexun.com"
url = "360tuan.com"
url="www.stheadline1.com"
url = "topbiz360.com"
url = "hengcheng-tools.com.cn"
url = "bigpian.cn"
# browser.get(url)
# browser.get("about:preferences")#进入选项页面
# browser.find_element(By.ID, "linkTargeting").click()

try:
    print("1")
    httpsurl =  'http://' + url
    resultdata = requesturl(httpsurl)
    #网页是否无法访问
    for badtitle in badtitles:
        if badtitle in resultdata['title']:
            print(badtitle)
            raise Exception("title error")
except Exception as e:
    print (e)
    try:
        print("2")
        if url.split(".")[0]!="www":
            httpsurl = 'http://www.' + url
        else:
            httpsurl = 'http://' + url.replace('www.','',1)
        resultdata = requesturl(httpsurl)
        #网页是否无法访问
        for badtitle in badtitles:
            if badtitle in resultdata['title']:
                raise Exception("title error")
        # good_result1.append(url)
    except Exception as e:
        print (e)
        print("aaa")
        print(str(e))
        # print(traceback.format_exc()) 
        if "Reached error page" in str(e):
            print("aaa")
            print(type(e))
        # print("fail ", url)

url = "www.zhengjimt.com"
# url = "caenet.cn"
# url = "health.china.com.cn"
url="p2p.hexun.com"
# url = "360tuan.com"
url="www.huanbaoj.com"
url = "shansteelgroup.com"
url = "www.miaobolive.com"


try:
    print("1")
    httpsurl =  'http://' + url
    resultdata = requesturl(httpsurl)
    # print(resultdata)
    #网页是否无法访问
    for badtitle in badtitles:
        if badtitle in resultdata['title']:
            print('title error')
            print(badtitle)
            raise Exception
except Exception as e:
    print (e)
    try:
        print("2")
        if url.split(".")[0]!="www":
            httpsurl = 'http://www.' + url
        else:
            httpsurl = 'http://' + url.replace('www.','',1)
        resultdata = requesturl(httpsurl)
        print(resultdata)
        #网页是否无法访问
        for badtitle in badtitles:
            if badtitle in resultdata['title']:
                raise Exception
        # good_result1.append(url)
    except Exception as e:
        print (e)
        # print("fail ", url)
# browser.quit()



# url = "www.zhengjimt.com"
# # url = "caenet.cn"
# # url = "health.china.com.cn"
# # url="p2p.hexun.com"
# # url = "360tuan.com"
