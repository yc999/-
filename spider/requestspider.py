#-- coding: utf-8 --
import  requests
from bs4 import  BeautifulSoup


#Some User Agents
hds=[{'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US; rv:1.9.1.6) Gecko/20091201 Firefox/3.5.6'},\
    {'User-Agent':'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.12 Safari/535.11'},\
    {'User-Agent':'Mozilla/5.0 (compatible; MSIE 10.0; Windows NT 6.2; Trident/6.0)'},\
    {'User-Agent':'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:34.0) Gecko/20100101 Firefox/34.0'},\
    {'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/44.0.2403.89 Chrome/44.0.2403.89 Safari/537.36'},\
    {'User-Agent':'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},\
    {'User-Agent':'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50'},\
    {'User-Agent':'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0'},\
    {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'},\
    {'User-Agent':'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1'},\
    {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11'},\
    {'User-Agent':'Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11'},\
    {'User-Agent':'Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11'}]

    
response=requests.get('http://www.360tuan.com/index.php?mod=user&act=index')
re_text=response.text
re_content=response.content
print(re_text)
print(type(re_text))
print(re_content)
print(type(re_content))
response.encoding='utf-8'
re_text=response.text
print(re_text)


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

    

    response=requests.get('http://www.360tuan.com/index.php?mod=user&act=index')
    
    response.encoding = response.apparent_encoding

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
        makelog(url + " too less message")
    f = open(savefilepath, "w",encoding="utf-8")
    f.write(json.dumps(webinfo, ensure_ascii=False))
    f.close()#关闭文件