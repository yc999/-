#-- coding: utf-8 --
import  requests
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

# sessions=requests.session()
# sessions.mount('http://allegro.pl', HTTP20Adapter())
# response=requests.get('http://www.bigpian.cn')
# response=requests.get('https://nuxechina.com',verify=False)
# response=sessions.get('http://tech.ifeng.com',headers = headers)
# response=sessions.get('http://tech.ifeng.com')


#https://stapharma.com.cn/cn/about-us/facilities/
#https://stapharma.com/about-us/facilities/
# http://www.vslai.com/
# re_text=response.text
# print(re_text)
# print(type(re_text))

# response.encoding='utf-8'
# re_text=response.text
# print(re_text)

# print(re_content)
# print(type(re_content))


# headers={
#     "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
# "Upgrade-Insecure-Requests": "1",
# "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36"
# }
# response=requests.get('http://89ws.com',headers = headers,allow_redirects=False)

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

headers={   
'User-Agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER'
        } 



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

def solvehref(href):
    tmps = href.split('"')
    tmp = tmps[1]

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
    except Exception as e:
        print(e)
        pass
    return allatags


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
    print('当前编码 ',response.encoding)
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
    print(response.encoding)
    
    # sessions.close()
    webdata[url] = response.text
    # writeurlfile(url, response.text)
    soup = BeautifulSoup(response.text,'html.parser')
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('style')]
    for element in soup(text = lambda text: isinstance(text, Comment)):
                element.extract()
    print(type(soup.get_text()))
    # print(soup)
    print(soup.get_text().strip())
    print(webcount)
    return response

maxwebpage = 4


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
    
 # 如果title有问题也要返回false

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
    if ifbadtitle(webinfo['title']):
        return False
    # 可能是欢迎页
    skip_text = ['点击','跳转','进入','访问']
    href_text = ['index', 'main','default','info','home']
    #数据太少  找到所有的a标签 选择合适的访问
    if True:
        soup = BeautifulSoup(response.text, 'html.parser')
        atag = soup.find_all('a')
        # atag = return_all_url(soup)
        # 点击href符合的链接
        for tag in atag:
            # print(tag)
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
        atag = return_all_url(soup)
        # print(atag)
        for tag in atag:
            tmpurl = url_now.replace("http://","",1)
            tmpurl = tmpurl.replace("https://","",1)
            tmpurl = tmpurl.replace("www.","",1)
            for keyword in href_text:
                try:
                    next_url = urljoin(url_now, tag) #寻找可能的相关链接
                except:
                    continue
                if tmpurl in next_url and keyword in next_url and next_url not in havegetlist and samewebsite(url_now, next_url) and havegetcount < maxwebpage:
                    print("next_url, ",next_url)
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
    return True


url = "meilishuo.com"
requests.packages.urllib3.disable_warnings()
# sessions=requests.session()
# sessions.keep_alive = False
# sessions.mount(url, HTTP20Adapter())
# sessions.mount(url, HTTP20Adapter())

# response = requests.get(url,verify=False,allow_redirects=True,headers = headers)
# requesturl(url)

tmpurl = url.replace('www.','',1)
httpurl =  'http://' + url
resultdata = requesturl(httpurl)
if resultdata == False:
    if url.split(".")[0]!="www":
        httpurl = 'http://www.' + url
    else:
        httpurl = 'http://' + url.replace('www.','',1)
    resultdata = requesturl(httpurl)
    