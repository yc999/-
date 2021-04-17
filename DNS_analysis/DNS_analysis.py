#-- coding: utf-8 --
import  requests
from bs4 import  BeautifulSoup, Comment
import re
import time
# import eventlet
import os
import sys
import io
import json
from selenium.webdriver.firefox.options import Options
from selenium import webdriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import UnexpectedAlertPresentException
from keras.models import load_model
from gensim.models import KeyedVectors
import numpy as np
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
import random
import meanShift as ms
import mytool
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *

badtitles=['404 Not Found', '找不到',  'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7','Default','已暂停' ,'Server Error','403 Forbidden','禁止访问','载入出错','没有找到',
'无法显示','无法访问','Bad Gateway','正在维护','配置未生效','访问报错','Welcome to nginx','Suspended Domain',
'IIS Windows','Invalid URL','服务器错误','400 Unknown Virtual Host','无法找到','资源不存在',
'Temporarily Unavailable','Database Error','temporarily unavailable','Bad gateway','不再可用','error Page',
'Internal Server Error','升级维护中','Service Unavailable','站点不存在','405','Access forbidden','System Error',
'详细错误','页面载入出错','Error','错误','Connection timed out','域名停靠','网站访问报错','错误提示','临时域名',
'未被授权查看','Test Page','发生错误','非法阻断','链接超时','403 Frobidden','建设中','访问出错','出错啦']

time_limit = 40  #set timeout time 3s


# 判断标题是否正常 
# mytitle 需要判断的title 
# 正常返回 False 不正常返回 True
def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False





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









def prasednsdata(data):
    dnsdata = {}
    parts = data.split("\t")
    dnsdata['tnow'] = parts[0].split(":")[1]
    dnsdata['tbeg'] = parts[1].split(":")[1]
    dnsdata['tend'] = parts[2].split(":")[1]
    dnsdata['count'] = parts[3].split(":")[1]
    tmp = parts[4].split(":")[1]
    tmp1 = tmp.split("+")
    dnsdata['rkey'] = tmp1[0]
    dnsdata['Dnstype'] = tmp1[1]
    dnsdata['data'] = parts[5].split(":")[1]
    return dnsdata


#返回正向域名
def getrkey_domainname(rkey):
    names = rkey.split(".")
    len_names = len(names)
    result_name = names[len_names-1]
    for i in range(len_names-2, -1, -1):
        result_name = result_name + "." + names[i]
    return result_name
    # result_name = result_name + name  




# 步骤1 初始化
# 
# 
# 1.1 加载词向量  
# embeddings_index 为字典  单词 ：下标
# embedding_matrix 词向量数组 

EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330
MAX_SEQUENCE_LENGTH = 10



word2vec_path = '/home/jiangy2/dnswork/glove/Tencent_AILab_ChineseEmbedding.txt'
tc_wv_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
# EMBEDDING_length = 8824330
EMBEDDING_length = len(tc_wv_model.vocab.keys())
print('Found %s word vectors.' % EMBEDDING_length)

embeddings_index = {}
embedding_matrix = np.zeros((EMBEDDING_length + 1, EMBEDDING_DIM))

for counter, key in enumerate(tc_wv_model.vocab.keys()):
    embeddings_index[key] = counter+1
    coefs = np.asarray(tc_wv_model[key], dtype='float32')
    embedding_matrix[counter+1] = coefs

del tc_wv_model


# 1.2 加载模型
modelsave_path = "/home/jiangy2/dnswork/modeldir/LSTMmodel"
LSTM_model = load_model(modelsave_path)




# 1.3 加载停用词
# stopwordslist 保存所有停用词

stopwordslist = []  # 停用词列表
stopwords_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
stopwordslist = mytool.read_stopwords(stopwords_path)



#  1.4 加载cdnlist
cdnlist = []
cdnlist_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
cdnlist = mytool.read_cdnlist(cdnlist_path)



# 1.5 加载 tldlist
tldlist = []
tldlist_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
tldlist = mytool.read_tldlist(tldlist_path)


#2.2 设置分类类别
# classtype 保存了所有的分类信息  子类名 ： 父类目
# class_index 保存了父类名对应的下标

class_index = { '休闲娱乐':0, '生活服务':1, '购物网站':2, '政府组织':3, '综合其他':4, '教育文化':5, '行业企业':6,'网络科技':7,
 '体育健身': 8, '医疗健康':9, '交通旅游':10, '新闻媒体':11}


classtype = { '购物':'购物网站','游戏':'休闲娱乐','旅游':'生活服务','军事':'教育文化','招聘':'生活服务','时尚':'休闲娱乐',
'新闻':'新闻媒体资讯','音乐':'休闲娱乐','健康':'医疗健康','艺术':'教育文化',
'社区':'综合其他','学习':'教育文化','政府':'政府组织','搞笑':'休闲娱乐','银行':'生活服务',
'酷站':'综合其他','视频':'休闲娱乐','电影':'休闲娱乐','文学':'休闲娱乐','体育':'体育健身','科技':'网络科技',
'财经':'生活服务','汽车':'生活服务','房产':'生活服务','摄影':'休闲娱乐','设计':'网络科技','营销':'行业企业',
'电商':'购物网站','外贸':'行业企业','服务':'行业企业','商界':'行业企业','生活':'生活服务'}


def initclass(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
            parts = line.split(",")
            if  not line:
                break
            classtype[parts[0]]=parts[2].strip("\n")


# filepath = "D:/dnswork/sharevm/top.chinaz.txt"
initfilepath = "/home/jiangy2/dnswork/top.chinaz.txt"
initclass(initfilepath)


# 将单词转为词向量的下标,下标从1开始 返回下标的list
def words2index(words):
    index_list = []
    for word in words:
        if word in embeddings_index.keys():  # 单词是否在词向量中
            index_list.append(embeddings_index[word])
    return index_list


def predict_webclass(webdata):
    X_train_text = []
    # if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
    tmp_data = ""
    for data in webdata['webtext']:
        tmp_data=tmp_data + tmp
    len_webtext = len(tmp_data)
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    len_chinese = len(rule.sub('',tmp_data)
    if len_chinese/len_webtext < 0.5:
        return '外语网站'
    if len(webdata['webtext'])>=15:
        X_train_text.append(mytool.get_all_webdata(webdata))
    else:
        return '数据过少'
    #  将文本转为张量
    # X_train 训练数据
    X_train = []
    for sentence in X_train_text:
        tmp_words = mytool.seg_sentence(sentence,stopwordslist)
        X_train.append(words2index(tmp_words))

    # 3 机器学习训练
    model_max_len = 300
    x_train_raw = pad_sequences(X_train, maxlen=model_max_len)
    predicted = LSTM_model.predict(x_train_raw)

    return predicted


# 判断是否为cdn 
# 规则: 在cdnlist中
# 间隔出现多次顶级域名
# 域名长度大于5
# 如果是cdn返回True
# 返回False
def filter_cdn(url):
    for cdn_name in cdnlist:
        if cdn_name in url:
            return True
    names = url.split(".")
    count_names = len(names)
    if count_names >= 6:
        return True
    for name in names[0:count_names-2]:
        if name in cdnlist:
            return True
    return False


dnstpye_value = {'1' : "A", '5':"CNAME", '28':"AAAA"}
# 读取dns数据
dnsdata_path = "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2020-11/pdns_data"
dnsdata_file = open(dnsdata_path, 'r', encoding='utf-8')
while True:
    line = dnsdata_file.readline()
    if  line:
        try:
            dnsdata = prasednsdata(line)
        except:
            continue
        if dnsdata['Dnstype'] not in dnstpye_value: # 只取 A AAAA CNAME记录
            continue
        # print(dnsdata)
        url = getrkey_domainname(dnsdata['rkey'])
        # 过滤url
        
        if filter_cdn(url):
            print(url, " cdn")
            continue
        try:
            httpsurl =  'http://' + url
            resultdata = requesturl(httpsurl)
            if ifbadtitle(resultdata['title']):
                raise Exception("title error")
            # 输入模型 进行判断
            predict_result = predict_webclass(resultdata)
            print(predict_result)
        except:
            try:
                if url.split(".")[0]!="www":
                    httpsurl = 'http://www.' + url
                else:
                    httpsurl = 'http://' + url.replace('www.','',1)
                resultdata = requesturl(httpsurl)
                #网页是否无法访问
                if ifbadtitle(resultdata['title']):
                    raise Exception("title error")
                predict_result = predict_webclass(resultdata)
                print(predict_result)
            except Exception as e:
                print(e)
    else:
        break
