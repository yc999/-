
# -*- coding: utf-8 -*-
import jieba
import numpy as np
import os
import time
import codecs
import re
import jieba.posseg as pseg
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from bs4 import  BeautifulSoup, Comment

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

content_train_src = []      #训练集文本列表
opinion_train_stc = []      #训练集类别列表
file_name_src=[]          #训练集文本文件名列表

webdatapath = ""


# webdata = read_webdata("E:/webdata/旅游网站/sh.tuniu.com.txt")


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


stopwordslist = []  # 停用词列表
stopwords_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
stopwordslist = mytool.read_stopwords(stopwords_path)

# 输入 html文档
# 返回html中所有的文本 string
# 如果标题有问题 返回False
def filtertext(htmldata):
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




# 读取训练文件
def readtrain(train_src_list):
    filepath = ""
    webdatadic = mytool.read_webdata(filepath)
    for htmldata in webdatadic:
        htmltext = filtertext(htmldata) #过滤出中文
        if htmltext == False:
            continue
        cut_text = Word_cut_list(htmltext) #生成词列表



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
            if word not in stopwordslist:#词语的清洗：去停用词
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


webdata = mytool.read_webdata("E:/webdata/中小学校/haiquan.com.cn.txt")

for htmldata in webdata:
    # print(htmldata)
    
    htmltext = filtertext(webdata[htmldata])
    # print(htmltext)
    if htmltext == False:
            continue
    cut_text = Word_cut_list(htmltext)
    print(cut_text)

print(len(cut_text))
vectorizer=CountVectorizer()
tfidftransformer=TfidfTransformer()
tfidf = tfidftransformer.fit_transform(vectorizer.fit_transform(cut_text))  # 先转换成词频矩阵，再计算TFIDF值
print (tfidf.shape)
print(tfidf)
docs = ["原任第一集团军副军长", "在9·3抗战胜利日阅兵中担任“雁门关伏击战英雄连”英模方队领队记者林韵诗继黄铭少将后"]
new_tfidf = tfidftransformer.transform(vectorizer.transform(docs))
# train=readtrain(train_src_all)
# content=segmentWord(train[0])
# filenamel=train[2]
# opinion=train[1]