
# -*- coding: utf-8 -*-
import jieba
import numpy as np
import os
import time
import codecs
import re
import jieba.posseg as pseg
from numpy.core.fromnumeric import shape
from paddle.fluid.layers.control_flow import max_sequence_len
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from bs4 import  BeautifulSoup, Comment
from sklearn.metrics import classification_report
import sys
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
import mytool


badtitles = ['404 Not Found', '找不到',  'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7','Default','已暂停' ,'Server Error','403 Forbidden','禁止访问','载入出错','没有找到',
'无法显示','无法访问','Bad Gateway','正在维护','配置未生效','访问报错','Welcome to nginx','Suspended Domain',
'IIS Windows','Invalid URL','服务器错误','400 Unknown Virtual Host','无法找到','资源不存在',
'Temporarily Unavailable','Database Error','temporarily unavailable','Bad gateway','不再可用','error Page',
'Internal Server Error','升级维护中','Service Unavailable','站点不存在','405','Access forbidden','System Error',
'详细错误','页面载入出错','Error','错误','Connection timed out','域名停靠','网站访问报错','错误提示','临时域名',
'未被授权查看','Test Page','发生错误','非法阻断','链接超时','403 Frobidden','建设中','访问出错','出错啦','ACCESS DENIED','系统发生错误','Problem loading page']

def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False

# 1.读取文件，预处理
# 2.分词
# 3.过滤低频词，停用词
# 4.生成tfidf
# 5.训练


# 变量定义
content_train_src = []      #训练集文本列表
opinion_train_stc = []      #训练集类别列表
file_name_src=[]          #训练集文本文件名列表



def get_head(soup):
    head = soup
    webinfo = {}
    webinfo['title'] = ""
    webinfo['description'] = ""
    webinfo['keywords'] = ""
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
    
    if ifbadtitle(webinfo["title"]):
            return False
    result_text = ""
    for text in webinfo:
        result_text += webinfo[text]
    return result_text


# 2.1 读取停用词
# stopwordslist 保存所有停用词
stopwordslist = []  # 停用词列表
stopwords_path = "C:/Users/shinelon/Desktop/linuxfirefox/stopwords-master/stopwords-master/cn_stopwords.txt"
# stopwords_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
stopwordslist = mytool.read_stopwords(stopwords_path)


def filtertext(htmldata):
    """
    # 输入 html文档
    # 返回html中所有的文本 string
    # 如果网页标题包含不正常文本 返回False
    """
    soup = BeautifulSoup(htmldata,'html.parser')
    head_text = get_head(soup)
    if head_text == False:
        return False
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('style')]
    for element in soup(text = lambda text: isinstance(text, Comment)):
        element.extract()
    body = soup.get_text()
    return head_text + body



def Word_pseg(self,word_str):  # 名词提取函数
        words = pseg.cut(word_str)
        word_list = []
        for wds in words:
            # 筛选自定义词典中的词，和各类名词，自定义词库的词在没设置词性的情况下默认为x词性，即词的flag词性为x
            if wds.flag == 'x' and wds.word != ' ' and wds.word != 'ns' \
                    or re.match(r'^n', wds.flag) != None \
                            and re.match(r'^nr', wds.flag) == None:
                word_list.append(wds.word)
        return word_list


def Word_cut_list(word_str):
    #利用正则表达式去掉一些一些标点符号之类的符号。
    word_str = re.sub(r'\s+', ' ', word_str)  # trans 多空格 to空格
    word_str = re.sub(r'\n+', ' ', word_str)  # trans 换行 to空格
    word_str = re.sub(r'\t+', ' ', word_str)  # trans Tab to空格
    word_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
    word_str = re.sub(u"[^\u4E00-\u9FA5]"," ", word_str)
    wordlist = list(jieba.cut(word_str))#jieba.cut  把字符串切割成词并添加至一个列表
    wordlist_N = []
    # chinese_stopwords=self.Chinese_Stopwords()
    for word in wordlist:
        if word not in stopwordslist: #词语的清洗：去停用词
            if word != '\r\n'  and word!=' ' and word != '\u3000'.encode('utf-8').decode('unicode_escape') \
                    and word!='\xa0'.encode('utf-8').decode('unicode_escape'):#词语的清洗：去全角空格
                wordlist_N.append(word)
    return wordlist_N


def segmentWord(cont):
    listseg=[]
    for i in cont:
        Wordp = Word_pseg(i)
        New_str = ''.join(Wordp)
        Wordlist = Word_cut_list(New_str)
        file_string = ''.join(Wordlist)
        listseg.append(file_string)
    return listseg


MAX_SEQUENCE_LENGTH = 0
# max_sequence_lenth = 0

# 读取html文件,并处理成词语
def readtrain(filepath):
    global MAX_SEQUENCE_LENGTH 
    webdatadic = mytool.read_webdata(filepath)
    result_list = []
    for htmldata in webdatadic:
        htmltext = filtertext(webdatadic[htmldata]) # 读取网页中的纯文本
        if htmltext == False:
            continue
        cut_text = Word_cut_list(htmltext) # 生成词列表
        result_list += cut_text
    if len(result_list)> MAX_SEQUENCE_LENGTH:
       MAX_SEQUENCE_LENGTH = len(result_list)
    return result_list


def read_all_data(datapath):
    data = readtrain(datapath)
    if data == []:
        return ""
    cut_text = ' '.join(data)
    # result_data.append(cut_text)
    return cut_text


class_index = { '休闲娱乐':0, '生活服务':1, '购物网站':2, '政府组织':3, '综合其他':4, '教育文化':5, '行业企业':6,'网络科技':7,
 '体育健身': 8, '医疗健康':9, '交通旅游':10, '新闻媒体':11}
classtype = {'购物':'购物网站','游戏':'休闲娱乐','旅游':'生活服务','军事':'教育文化','招聘':'生活服务','时尚':'休闲娱乐',
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


initfilepath = "D:/dnswork/sharevm/top.chinaz.txt"
# initfilepath = "/home/jiangy2/dnswork/top.chinaz.txt"
initclass(initfilepath)


content_train_src = []      # 训练集文本列表
opinion_train_stc = []      # 训练集类别列表


# 获取全部数据集
path = "/home/jiangy2/webdata/"
path = "D:/dnswork/sharevm/topchinaz/"
datapath = "E:/webdata/"


fs = os.listdir(datapath)
i=0
j=0
for subpath in fs:
    filepath = os.path.join(datapath, subpath)
    # print(filepath)
    if (os.path.isdir(filepath)):
        print(subpath)
        webdata_classtype = classtype[subpath]  # 查询父类别
        webdata_class_index = class_index[webdata_classtype] #父类别下标
        webdata_path = os.listdir(filepath)
        for filename in webdata_path:
            i=i+1
            webdatapath = os.path.join(filepath, filename)
            # print(webdatapath)
            webdata = read_all_data(webdatapath)
            if webdata == "":
                continue
            content_train_src.append(webdata)               # 加入数据集 字符串
            opinion_train_stc.append(webdata_class_index)   # 加入标签集

print("opinion_train_stc ", opinion_train_stc)
model_max_len = MAX_SEQUENCE_LENGTH

print("MAX_SEQUENCE_LENGTH ",MAX_SEQUENCE_LENGTH)

# 测试
"""
if test:
    webdata = mytool.read_webdata("E:/webdata/中小学校/haiquan.com.cn.txt")
    for htmldata in webdata:
        htmltext = filtertext(webdata[htmldata])
        # print(htmltext)
        if htmltext == False:
                continue
        cut_text = Word_cut_list(htmltext)
        print(cut_text)
    print(len(cut_text))
    cut_text = ' '.join(cut_text) #拼接成一个字符串
    print(cut_text)
    textlist=[cut_text]

    vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b")

    X = vectorizer.fit_transform(textlist)
    print(vectorizer.get_feature_names())
    print(X.toarray())

    transform = TfidfTransformer()
    Y = transform.fit_transform(X)    # 这里的输入是上面文档的计数矩阵
    print(Y.toarray())                # 输出转换为tf-idf后的 Y 矩阵
"""

# print(content_train_src)

from keras.utils.np_utils import *
from sklearn.model_selection import train_test_split

vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(content_train_src)

transform = TfidfTransformer()
x_tfidf = transform.fit_transform(X)

#训练 用fit_transform
# count_train=vectorizer.fit_transform(content_train)
# tfidf = transform.fit_transform(count_train)

Y_label=to_categorical(opinion_train_stc, len(class_index))
x_train, x_test, y_train, y_test = train_test_split(x_tfidf, opinion_train_stc, test_size=0.2)

print(shape(x_train))
print(shape(y_train))
clf = SVC()
clf.fit(x_train,y_train)

pred_y  = clf.predict(x_test)
print(classification_report(y_test,pred_y))

#测试
# count_test=vectorizer.transform(content_test)
# test_tfidf = transform.transform(count_test)


# tfidftransformer=TfidfTransformer()
# tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(cut_text))  # 先转换成词频矩阵，再计算TFIDF值
# print (tfidf.shape)
# print(tfidf)
# docs = ["原任第一集团军副军长", "在9·3抗战胜利日阅兵中担任“雁门关伏击战英雄连”英模方队领队记者林韵诗继黄铭少将后"]
# new_tfidf = tfidftransformer.transform(vectorizer.transform(docs))
# train=readtrain(train_src_all)
# content=segmentWord(train[0])
# filenamel=train[2]
# opinion=train[1]