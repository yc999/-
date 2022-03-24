#-- coding: utf-8 --
import time
import  requests
import queue
from bs4 import  BeautifulSoup, Comment
from hyper.contrib import HTTP20Adapter
from urllib.parse import urljoin
import re

badtitles=[
    '资源不存在',    '找不到',      '页面载入出错',           '临时域名',           '阻断页',
    '建设中',       '访问出错',     '出错啦',                'IIS7',            '温馨提示',    
     '无法找到',    '未被授权查看',  '已暂停' ,             '详细错误',              '禁止访问',         
    '载入出错',         '没有找到',     '无法显示',           '无法访问',       '升级维护中',          
    '正在维护',         '配置未生效',   '访问报错',          '域名停靠',        '网站访问报错',
     '服务器错误',      '不再可用',     '错误',                   '错误提示',   '发生错误', 
        '非法阻断',     '链接超时',     '站点不存在',         '系统发生错误',   '网站欠费提醒',
        '页面没找到','域名售卖',        '时无法进行访问', '域名未备案','正在升级维护',
        '共享IP', '域名停放','恭喜，站点创建成功','您的站点已过期'
        '网站暂时无法访问',
        'Apache Tomcat', 'Welcome to OpenResty','Welcome to CentOS',
        'Welcome to tengine',
     'Not Found',               'Welcome to nginx',              'Suspended Domain',
    'IIS Windows',              'Invalid URL',                  '400 Unknown Virtual Host',
     'Server Error',            '403 Forbidden',                'Temporarily Unavailable', 
      'Database Error',           'temporarily unavailable',     'Bad gateway',        
         'error Page',             'Internal Server Error',      '405',   
    'Service Unavailable',          'Access forbidden',         'System Error',
      '404 Not Found',               'null',    'Apache2 Ubuntu Default Page',
                          'Error',   
       'Connection timed out',      'TestPage',                     'Test Page',
        'Bad Gateway',  'Bad Request',      'Time-out',                 'No configuration',     
        '403 Frobidden','ACCESS DENIED','Problem loading page']


headers = {
    #  "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Safari/537.36",
    # "Referer": 'www.baidu.com',
    # "host": "allegro.pl"
    ":authority": "allegro.pl",
    ":method": "GET",
    ":path": "/",
    ":scheme": "https",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "accept-encoding": "gzip, deflate, br",
    "accept-language": "zh-CN,zh;q=0.9,en;q=0.8",
    # "cookie": "_cmuid=81d73072-2bbe-4a13-945b-ab145dbc23ab; __gfp_64b=T8ZmiPzI85urKGng3_NG7vTdaGgi_9mR9drIR8u.PXf.T7|1620284247; gdpr_permission_given=1; datadome=Hji1qdEFzNPT99QQd0StqcyeyR72BiFyxU6R-Tidm5_2cSwrCjlbULzOt1HZmjtl2sjjTz3l49.5Mqwv8PkHiEeTsKgkn_yxeL.8UyMycR",
    "dpr": "1",
    # "sec-ch-ua": "Google Chrome";v="89", "Chromium";v="89", ";Not A Brand";v="99"
    # sec-ch-ua-mobile: ?0
    # sec-fetch-dest: document
    # sec-fetch-mode: navigate
    # sec-fetch-site: none
    # sec-fetch-user: ?1
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
    "viewport-width": "1920"
}


badurl = ['css','ico','png','jpg','js','xml','jpeg',
           'gif' ,'woff','json','svg','less']

def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False

def havekey(tag):
    if  tag.has_attr('href') or  tag.has_attr('data-href'):
        if tag.string != None and len(tag.string.strip()) < 8:
            searchObj = re.search("关于.{0,8}", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search(".{0,8}简介", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search(".{0,8}概况", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search(".{0,8}介绍", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("了解.{0,8}", tag.string, flags=0)
            if searchObj:
                return True
            searchObj = re.search("走近.{0,8}", tag.string, flags=0)
            if searchObj:
                return True

headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0'
#   ,'Referer':'http://gzlss.hrssgz.gov.cn',

        } 

# headers={
#     'Host': 'gzlss.hrssgz.gov.cn',
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0',
#     'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
#     'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
#     'Accept-Encoding': 'gzip, deflate',
#     'Referer': 'http://gzlss.hrssgz.gov.cn/',
#     'Cookie': 'iCo9PwW9p8=MDAwM2IyYzhjYzAwMDAwMDAwMjEwDiR3CUcxNjIzODU2OTUw; JSESSIONID=RfsTsy9IEiR0JzHLlwm_JNV99dazeCYIaO4Ou9ulXTwk7wIipZpA!203181343'
#    , 'Upgrade-Insecure-Requests': '1',
#     'Cache-Control': 'max-age=0'
# }



def samewebsite(sourceurl, targeturl):
    tmp_split = targeturl.split("?")[0]
    tmp_split = tmp_split.split("#")[0]
    tmp_split = tmp_split.split(".")

    for url in badurl:
        if url in tmp_split:
            # print(url ," in ",targeturl)
            return False
    tmp_sourceurl = sourceurl.replace('http://','')
    tmp_sourceurl = tmp_sourceurl.replace('https://','')
    tmp_sourceurl = tmp_sourceurl.replace('www.','')
    tmp_sourceurl = tmp_sourceurl.split('/')[0].strip()

    tmp_targeturl = targeturl.replace('http://','')
    tmp_targeturl = tmp_targeturl.replace('https://','')
    tmp_targeturl = tmp_targeturl.split('/')[0].strip()
    # print(tmp_sourceurl, tmp_targeturl)
    if tmp_sourceurl in tmp_targeturl:
        # print(sourceurl, targeturl)
        return True
    return False

#将后面的url规范化 把多余的//改成/
def norm_url(url):
    tmpurls = url.split("//")
    prefix_url = tmpurls[0]
    tmpurl = url.replace('http://','')
    tmpurl = tmpurl.replace('https://','')
    # tmpurl = tmpurl.replace('www.','')
    tmpurl = tmpurl.replace('//','/')
    if tmpurl[-1]=='/':
        tmpurl = tmpurl[0:-1]
    return prefix_url + "//" + tmpurl

#拿到url的fqdn ，去掉/后的全部路径
# http://app.imufu.cn//homePage 变成 http://app.imufu.cn
def geturlroot(url):
    tmpurls = url.split("//")
    tmpurl = tmpurls[1]
    # if tmpurl.split(".")[0] == "www":
    #     tmpurl = tmpurl.replace("www.","",1)
    tmpurl = tmpurl.split("/")[0]
    tmpurl = tmpurls[0] + "//" + tmpurl
    return tmpurl

# 处理双引号
def solvehref(href):
    tmps = href.split('"')
    if len(tmps)<2:
        print(href)
    tmp = tmps[1]
    return tmp

# 处理单引号
def solveopenhref(href):
    tmps = href.split("'")
    if len(tmps)<2:
        print(href)
    tmp = tmps[1]
    # print(tmp)
    return tmp

# 处理
# <meta content="0.2;url=http://www.radio.cn/pc-portal/home/index.html" http-equiv="refresh"/>
#类似的
def solveurlhref(href):
    tmps = href.split("=")
    tmp = tmps[1]
    tmps = tmp.split('"')
    tmp = tmps[0]
    tmps = tmp.split(';')
    tmp = tmps[0]
    # print(tmp)
    return tmp

def return_all_url(soup):
    allatags = []
    try:
        pattern = re.compile("href\s{0,3}=\s{0,3}\"[^\"]+\"")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solvehref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
        pattern = re.compile("href\s{0,3}=\s{0,3}\'[^\']+\'")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solveopenhref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
        pattern = re.compile("open\s{0,3}\(\s{0,3}\'[^\']+\'")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solveopenhref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
        pattern = re.compile("src\s{0,3}=\s{0,3}\"[^\"]+\"")
        hrefs=re.findall(pattern,soup.prettify())
        # print(hrefs)
        for href in hrefs:
            tmpurl = solvehref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
        pattern = re.compile("rel\s{0,2}=\s{0,3}\'[^\']+\'")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solveopenhref(href)
            if tmpurl not in allatags:
                # print(tmpurl)
                allatags.append(tmpurl)
        pattern = re.compile("url\s{0,2}=\s{0,3}[^\"\s]+[\s\"]")
        hrefs=re.findall(pattern,soup.prettify())
        for href in hrefs:
            tmpurl = solveurlhref(href)
            if tmpurl not in allatags:
                allatags.append(tmpurl)
        
    except Exception as e:
        print(e)
        pass
    # print(allatags)
    return allatags


def get_and_add(url,webdata, webcount, hashlist, atag):
    print(url)
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
    print(response.status_code)
    print(response.encoding )
    if response.encoding == 'ISO-8859-1':
        response.encoding = requests.utils.get_encodings_from_content(response.text)
        print(response.encoding)

        if len(response.encoding)==0:
            response.encoding = 'GBK'

    # print(response.text)
    print('当前编码 ',response.encoding)
    if response.encoding == ['gbk2312']:
        response.encoding = 'GBK'
    elif response.encoding == ['gb2312']:
        response.encoding = 'GBK'
    elif response.encoding == ['GB2312']:
        response.encoding = 'GBK'
    elif response.encoding == ['gbk']:
        response.encoding = 'GBK'
    elif response.encoding == ['GBK']:
        response.encoding = 'GBK'
    elif type(response.encoding) == list:
        if 'gbk2312' in response.encoding:
            response.encoding = 'GBK'
        if 'gb2312' in response.encoding:
            response.encoding = 'GBK'
        elif 'gbk' in response.encoding:
            response.encoding = 'GBK'
        else:
            response.encoding = 'utf-8'

    
    if response.status_code != 200:
        # sessions.close()
        return False
    print(response.encoding)
    soup = BeautifulSoup(response.text,'html.parser')

    #判断当前url和之前的是否只多了一个参数，如果是，则比较网页内容
    tmpsoup_hash = hash(soup.get_text())
    if tmpsoup_hash in hashlist:
            # print("same hash")
        return False
    tmpatag = return_all_url(soup)

    # sessions.close()
    print('next_url ',url)
    tmp_url_now = response.url
    webdata[tmp_url_now] = response.text
    # writeurlfile(url, response.text)
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('style')]
    for element in soup(text = lambda text: isinstance(text, Comment)):
                element.extract()
    # print(type(soup.get_text()))
    print(soup)
    print(soup.get_text().strip())
    print(webcount)
    for tag in tmpatag:
        atag.put(tag)
    # time.sleep(10)
    return response

maxwebpage = 4

def requesturl(url):
    print(url)
    webinfo={}  # 最后保存的数据
    havegetlist = [] # 已经访问过的网页
    hashlist = []       #访问过的网页的hash值
    atag = queue.Queue(maxsize=0)
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
            tmpurl = tmpurl
            if tmpurl not in havegetlist and samewebsite(abouturl, tmpurl) and count < maxwebpage:
                next_response = get_and_add(tmpurl, webdata, count, hashlist,atag)
                if next_response != False:
                    next_soup = BeautifulSoup(next_response.text,'html.parser')
                    hashlist.append(hash(next_soup.get_text()))
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
    response = get_and_add(url, webdata, havegetcount, hashlist, atag)
    if response == False:
        return False
    next_soup = BeautifulSoup(response.text,'html.parser')
    hashlist.append(hash(next_soup.get_text()))
 # 如果title有问题也要返回false

    def initwebinfo():
        webinfo['title'] = ""
        webinfo['description'] = ""
        webinfo['keywords'] = ""
        webinfo['webtext'] = []

    initwebinfo()
    url_now = response.url      # 当前的url
    url_now_root = geturlroot(url_now)
    print('url_now ',url_now, 'url root ',url_now_root)
    havegetlist.append(url_now)
    havegetlist.append(url_now_root)

    soup = BeautifulSoup(response.text, 'html.parser')
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
        get_bodytext()

    
    # 第一阶段
    #开始获取信息
    get_info()
    if ifbadtitle(webinfo['title']): #如果title 不对提前退出
        return False
    # 可能是欢迎页
    skip_text = ['中文','点击','跳转','进入','访问',
    'chinese','Chinese','CHINESE','china']
    href_text = ['brand','index', 'main','default','info','home','Index','Main',
        'Default','Info','Home','about','About','guide','Guide',
        'chinese','Chinese','CHINESE','china','cn-zh','CN-ZH','zh_cn',
     'find','Find','intro','Intro','.cn','cn.']
    #数据太少  找到所有的a标签 选择合适的访问
    if True:
        soup = BeautifulSoup(response.text, 'html.parser')
        keywordtag = soup.find_all('a')
        # 点击href符合的链接
        for tag in keywordtag:
            if tag.has_attr('href') and tag.get_text():
                for keyword in skip_text:   #访问可能的跳转页面
                    if keyword in tag.get_text():
                        # print(tag)
                        try:
                            next_url = urljoin(url_now_root, tag['href'])
                        except:
                            continue
                        if samewebsite(url_now_root, next_url) and next_url not in havegetlist and havegetcount < maxwebpage: # 需要和当前url一致
                            next_response = get_and_add(next_url, webdata, havegetcount, hashlist,atag)
                            if next_response == False:
                                continue
                            next_soup = BeautifulSoup(next_response.text,'html.parser')
                            hashlist.append(hash(next_soup.get_text()))
                            abouturl = next_response.url          # 当前的url
                            havegetlist.append(abouturl)
                            havegetcount += 1
                            if next_url != abouturl:            #如果当前的url和选中进入的url不一样就要加入访问过的列表
                                havegetlist.append(next_url)
                            tmpsoup = BeautifulSoup(next_response.text, 'html.parser')
                            havegetcount = findaboutwebpage(abouturl, tmpsoup, havegetcount)
                            break
        # atag = return_all_url(soup)
        # print(atag)
        while not atag.empty() and havegetcount < maxwebpage:
            tag = atag.get()
            # print(tag)
        # for tag in atag:
            tmpurl = url_now_root.split("//")[1]
            if tmpurl.split(".")[0] == "www":
                tmpurl = tmpurl.replace("www.","",1)
            tmpurl = tmpurl.split("/")[0]
            try:
                next_url = urljoin(url_now_root, tag) #链接合并
                # print(next_url)
            except:
                continue
            if samewebsite(tmpurl, next_url):
                if havegetcount < maxwebpage:
                    # print(tmpurl, next_url)
                    for keyword in href_text:
                        if keyword in next_url:
                            # print(keyword, next_url)
                            # print('tmpurl ',tmpurl,'next_url ', next_url)
                            if tmpurl in next_url   and next_url not in havegetlist :
                                # print(next_url)
                                next_response = get_and_add(next_url, webdata, havegetcount, hashlist, atag)
                                if next_response == False:
                                    continue
                                next_soup = BeautifulSoup(next_response.text,'html.parser')
                                hashlist.append(hash(next_soup.get_text()))
                                abouturl = next_response.url          # 当前的url
                                havegetlist.append(abouturl)
                                havegetcount += 1
                                if next_url != abouturl:
                                    havegetlist.append(next_url)
                                soup = BeautifulSoup(next_response.text, 'html.parser')
                                havegetcount = findaboutwebpage(abouturl, soup, havegetcount)
    print(havegetcount)
    return True


# url = "qq163.cc"
# url = "yy521.com"
# url = 'gtgqw.com'
# url = 'www.ailab.cn'
# url = 'shuhai.com'

# url = 'yy521.com'
##无法访问
# url = 'gcwdw.com' 
# url = 'byby.xywy.com'
# url = 'apta.gov.cn'  #
# url = 'gxcic.net' #建设网
url = 'jzb.com'
# url = 'hebpta.com.cn'
# url = 'www.ttk-spring.com.cn'
# url = 'qfang.com'



url = 'hx168.com.cn'
requests.packages.urllib3.disable_warnings()
# sessions=requests.session()
# sessions.keep_alive = False
# sessions.mount(url, HTTP20Adapter())

# response = requests.get(url,verify=False,allow_redirects=True,headers = headers)
# requesturl(url)

# tmpurl = url.replace('www.','',1)
httpurl =  'http://' + url
resultdata = requesturl(httpurl)
if resultdata == False:
    if url.split(".")[0]!="www":
        httpurl = 'http://www.' + url
    else:
        httpurl = 'http://' + url.replace('www.','',1)
    resultdata = requesturl(httpurl)
    