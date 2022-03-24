# -*- coding: utf-8 -*-
# 集成学习
import numpy as np
import os
import time
import jieba.posseg as pseg
from numpy.core.fromnumeric import reshape, shape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from bs4 import  BeautifulSoup, Comment
from sklearn.metrics import classification_report
import sys
from keras.utils.np_utils import *
from sklearn.model_selection import train_test_split
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
import json
from gensim.models import KeyedVectors

import sys
from gensim.models import Word2Vec
# from LoadData import loadData
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
import keras.layers
# from keras.layers import Dropout,Dense,Embedding,LSTM,Activation
from keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Input, Flatten, BatchNormalization,concatenate , Bidirectional, Activation,Masking
import pickle
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary
import re
from gensim import corpora,models,similarities
import tensorflow as tf
from keras import backend as K #转换为张量
from gensim import corpora
# from gensim import models
# from gensim.corpora import Dictionary
import json
import logging
# from keras.engine.topology import Layer
from keras.layers import Layer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *

from MLcode.classcount import X_train_text
import multiprocessing as mp
# import mean_shift as ms
# import matplotlib.pyplot as plt
import jieba
import jieba.posseg as pseg
from numpy.core.fromnumeric import shape
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn import metrics
from bs4 import  BeautifulSoup, Comment
from sklearn.metrics import classification_report
import sys
from keras.utils.np_utils import *
from sklearn.model_selection import train_test_split
import chardet
import io
import codecs


# sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
os.environ['NLS_LANG'] = 'SIMPLIFIED CHINESE_CHINA.UTF8'
os.environ['OPENBLAS_NUM_THREADS'] = '1'


badtitles = ['404 Not Found', '找不到',  'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7','Default','已暂停' ,'Server Error','403 Forbidden','禁止访问','载入出错','没有找到',
'无法显示','无法访问','Bad Gateway','正在维护','配置未生效','访问报错','Welcome to nginx','Suspended Domain',
'IIS Windows','Invalid URL','服务器错误','400 Unknown Virtual Host','无法找到','资源不存在',
'Temporarily Unavailable','Database Error','temporarily unavailable','Bad gateway','不再可用','error Page',
'Internal Server Error','升级维护中','Service Unavailable','站点不存在','405','Access forbidden','System Error',
'详细错误','页面载入出错','Error','错误','Connection timed out','域名停靠','网站访问报错','错误提示','临时域名',
'未被授权查看','Test Page','发生错误','非法阻断','链接超时','403 Frobidden','建设中','访问出错','出错啦','ACCESS DENIED','系统发生错误','Problem loading page']



EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330



word2vec_path = '/public/ycdswork/dnswork/glove/Tencent_AILab_ChineseEmbedding.txt'
stopwords_path = "/public/ycdswork/dnswork/stopwords/cn_stopwords.txt"
webfilepath = "/public/ycdswork/dnswork/httpwebdata/"
file_dir = "/home/yangc/myclass/"
modelsave_path = "/public/ycdswork/modeldir/LSTMmodel"



tc_wv_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
# EMBEDDING_length = 8824330
EMBEDDING_length = len(tc_wv_model.key_to_index)
print('Found %s word vectors.' % EMBEDDING_length)

embeddings_index = {}
embedding_matrix = np.zeros((EMBEDDING_length + 1, EMBEDDING_DIM))
# tc_wv_model.key_to_index
# for counter, key in enumerate(tc_wv_model.vocab.keys()):
for counter, key in enumerate(tc_wv_model.key_to_index):
    # print(counter,key)
    embeddings_index[key] = counter+1
    coefs = np.asarray(tc_wv_model[key], dtype='float32')
    embedding_matrix[counter+1] = coefs


del tc_wv_model



#步骤2 数据预处理
print("step2")

# 2.1 读取停用词
# stopwordslist 保存所有停用词
def read_stopwords(filepath):
    stopwords = []
    if os.path.exists(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwordslist = []
stopwordslist = read_stopwords(stopwords_path)


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
    # body = ''.join(body)
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


#2.2 设置分类类别
# class_index 保存了父类名对应的下标


def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False




def Word_cut_list(word_str):
    #利用正则表达式去掉一些一些标点符号之类的符号。
    word_str = re.sub("[\s+\.\!\[\]\/_,\>\<\-$%^*¿(+\"\']+|[+——；:！·，”。【】《》～¿лΣùòЦ±д£ȫ²αμв»©½йÿλ，。：“？、~@#￥%……&*（）0123456789①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
    wordlist = list(jieba.cut(word_str))  #jieba.cut  把字符串切割成词并添加至一个列表
    # print(wordlist)
    wordlist_N = []
    # chinese_stopwords=self.Chinese_Stopwords()
    for word in wordlist:
        if word not in stopwordslist: #词语的清洗：去停用词
            if word != '\t':
                # if word in embeddings_index: # 词向量中有该词
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

def read_webdata(filepath):
    print(filepath)
    fsize = os.path.getsize(filepath)
    fsize = fsize/float(1024*1024)
    if fsize > 20:
        print(" size too big")
        return {}
    else:
        with open(filepath, 'r', encoding='utf-8') as file_to_read:
            return json.loads(file_to_read.read())

# 读取html文件,并处理成词语
# 其他页面中的词语 如果没有在主页中没出现  才放入列表中, 少于10个词的不考虑
def readtrain(filepath):
    global MAX_SEQUENCE_LENGTH 
    # webdatadic = mytool.read_webdata(filepath)
    try:
        webdatadic = read_webdata(filepath)
    except Exception as r:
        print(r)
        return []
    result_list = []
    result_dic = {}
    webkey = list(webdatadic.keys())  #key 是每个页面的url 
    if len(webkey)>0:
        for htmldata in webkey: #遍历每个页面，切词处理
            htmltext = filtertext(webdatadic[htmldata]) # 读取网页中的纯文本
            if htmltext == False:
                result_dic[htmldata] = []
                continue
            cut_text = Word_cut_list(htmltext) # 生成词列表
            # print("2")
            result_dic[htmldata] = cut_text
        result_list += result_dic[webkey[0]]  # 先保存首页
        # print(result_dic)
        for htmldata in webkey[1:]:
            for word in result_dic[htmldata]:
                # if word not in result_dic[webkey[0]]:  # 只保存不在首页中出现的词语
                    result_list.append(word)
        if len(result_list) < 10:
            return []
        tmpcount = 0
        len_result_list = list(dict.fromkeys(result_list))
        for word in len_result_list:
            # if word in embeddings_index:
                tmpcount += 1
        if tmpcount> MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH = tmpcount
    return result_list


"""
    返回空格分隔的词语串
"""
def read_all_data(datapath):
    data = readtrain(datapath)
    if data == []:
        return ""
    cut_text = ' '.join(data)
    # result_data.append(cut_text)
    return cut_text



taglist = ["a","div","li","span","img","td","p","ul","option","meta","tr","link","input","table",
"tbody","dd","h2","h3","hr","dt"]
tag_dic = {}  # tag对应下标的字典
TAG_length = len(taglist)

for i in range(TAG_length):
    tmpkey = taglist[i]
    tag_dic[tmpkey] = i+1

"""
    返回网页结构的特征
    特征为网页对应标签的下标
    
"""

def getpage_struct(datapath):
    result_list = []
    try:
        webdatadic = read_webdata(datapath)
    except Exception as r:
        print(r)
        return []
    webkey = list(webdatadic.keys())  # key 是每个页面的url 
    if len(webkey)>0:
        for htmldata in webkey:
            tmpdata = webdatadic[htmldata]
            soup = BeautifulSoup(tmpdata,'html.parser')
            body = soup.find("body")
            input_list = []
            if body is not None:
                for child in body.descendants:
                    if child.name in taglist:
                        # input_list.append(child.name)
                        input_list.append(tag_dic[child.name])
                result_list.append(input_list)
            else:
                result_list.append(input_list)
    return result_list
#2.3 读取爬取的网页信息
#读取保存的网页信息


# 保存爬到的数据
i=0
j=0

webfilecontent = {}     # 内容
webfilestruct = {}      # 结构

fs = os.listdir(webfilepath)

#读取所有爬到的网站内容 存到map中
for subpath in fs:
    print(subpath)
    filepath = os.path.join(webfilepath, subpath)
    # print(filepath)
    if (os.path.isdir(filepath)):
        webdata_path = os.listdir(filepath)
        # print(webdata_path)
        for filename in webdata_path:
            i = i +1
            tmpfilename = filename.replace(".txt","")
            # webdata = mytool.read_webdata(os.path.join(filepath, filename))
            webdatapath = os.path.join(filepath, filename)
            # print(webdatapath)
            webdata = read_all_data(webdatapath)
            webstructdata = getpage_struct(webdatapath)
            # print(webdata)
            if tmpfilename not in webfilecontent:
                webfilecontent[tmpfilename] = webdata
                webfilestruct[tmpfilename] = webstructdata
            else:
                print(tmpfilename,subpath)

# i = 39422





def unicode_to_cn(in_str, debug=False):
    out = None
    if isinstance(in_str, bytes):
        temp = str(in_str, encoding='utf-8')
        out = temp.encode('utf-8').decode('unicode_escape')
    else:
        out = in_str.encode('utf-8').decode('unicode_escape')
    return out

# 设置类别的下标
class_index ={}
indexcount = 0


for file in os.listdir(file_dir):
    file = file.replace(".txt",'')
    file_path = os.path.join(file_dir, file)
    if not os.path.isdir(file_path):
        tmp = file.split('.')[0]
        # if len(sys.argv)>1:
        tmp = tmp.replace("#U","\\u")
        print(tmp)
        tmp = unicode_to_cn(tmp)
        print(tmp)
        class_index[tmp] = indexcount
        indexcount = indexcount + 1

print(class_index)



#  从webfilecontent中拿到对应的分类好的文件
content_train_src = []      # 训练集文本列表
struct_train_src = []      # 训练集结构列表
opinion_train_stc = []      # 训练集类别列表
filename_train_src = []     # 训练集对应的域名
for file in os.listdir(file_dir):
    file_path = os.path.join(file_dir, file)
    tmp = file.split('.')[0]
    tmp = tmp.replace("#U","\\u")
    tmp = unicode_to_cn(tmp)
    print(tmp)
    if not os.path.isdir(file_path):
        # print(file_path)
        with open(file_path, 'r', encoding='utf-8') as file_to_read: #读取文件
            while True:
                line = file_to_read.readline()      # 读取内容
                # print(parts)
                if  not line:
                    break
                parts = line.split(",")      #取域名
                if len(parts)>=2:
                    url= parts[1]
                    if url.split(".")[0]=="www":
                        url  =  url.replace('www.','',1)
                    if url  in webfilecontent:
                        if len(webfilecontent[url])<26:
                            print(url)
                        else:
                            content_train_src.append(webfilecontent[url])               # 加入数据集 字符串
                            struct_train_src.append(webfilestruct[url])               # 加入数据集 字符串
                            opinion_train_stc.append(class_index[tmp])   # 加入标签集
                            filename_train_src.append(line)
                    elif 'www.'+ url  in webfilecontent:
                        if len(webfilecontent['www.'+url])<26:
                            print(url)
                        else:
                            content_train_src.append(webfilecontent['www.'+ url])               # 加入数据集 字符串
                            struct_train_src.append(webfilestruct['www.'+ url])
                            opinion_train_stc.append(class_index[tmp])   # 加入标签集
                            filename_train_src.append(line)
                    else:
                        print(url, line)



print("已爬取网页数：") #39422
print(i)
print("有效网页数：")
print(len(content_train_src))   #10766
print(len(struct_train_src))
print(len(opinion_train_stc))
print(len(filename_train_src))


# 把字符串按空格切分
def words2index(words):
    index_list = []
    word_list = []
    tmp_words = words.split(" ")
    for word in tmp_words:
        if word in embeddings_index.keys():  # test if word in embeddings_index
            index_list.append(embeddings_index[word])
            word_list.append(word)
    return index_list, word_list



# 2.4 将文本转为张量
#  原始训练数据
X_train = []             # 只保存文本的单词下标  去重
X_train_full = []             # 只保存文本的单词下标. 不去重
X_train_txt = []        # 只保存文本的单词
X_train_txt_full = []        # 只保存文本的单词
Y_train_index = []        # 只保存标签下标，这样可以通过下标找到filename_train_src和 opinion_train_stc
struct_list = []            # 保存符合条件的结构特征
opinion_list = []       # 保存筛选后的 opinion_train_stc 类别数组
filename_list = []      # filename_train_src
Max_lstm_lenth = 0



# 将单词转为词向量的下标,和对应词,下标从1开始 返回下标的list ，过滤词数较少的网站
short_list = []
Y_filename = []
# for word in content_train_src:
for i in range(len(content_train_src)):
    # tmp_words = mytool.seg_sentence(sentence,stopwordslist)
    tmplist_full, tmpword_list_full = words2index(content_train_src[i])
    tmplist = list(set(tmplist_full))  # delet recur word
    tmpword_list = list(set(tmpword_list_full))  # delet recur word
    if len(tmplist)>Max_lstm_lenth:
        Max_lstm_lenth = len(tmplist)
    if len(tmplist) >20:
        X_train.append(tmplist)
        X_train_full.append(tmplist_full)
        X_train_txt.append(tmpword_list)
        X_train_txt_full.append(tmpword_list_full)
        struct_list.append(struct_train_src[i])
        opinion_list.append(opinion_train_stc[i])
        filename_list.append(filename_train_src[i])
        Y_train_index.append(i)
        # opinion_train_stc.append(words2index(word))
    else:
        short_list.append(tmpword_list)
        Y_filename.append(filename_train_src[i])


# 输出short_list中的内容
for i in range(len(short_list)):
    print(short_list[i])
    print(Y_filename[i])


len(X_train)
len(short_list)
# Max_lstm_lenth

# opinion_train_stc = Y_train


print("有效网页数：") #10423
print(len(X_train))
print(len(X_train_full))
print(len(X_train_txt))
print(len(X_train_txt_full))
print(len(struct_list))
print(len(opinion_list))
print(len(filename_list))
print(len(Y_train_index))


# 记录长度
lstm_lenth_list = []
for word in X_train:
    lstm_lenth_list.append(len(word))

lstm_full_lenth_list = []
for word in X_train_full:
    lstm_full_lenth_list.append(len(word))

len(lstm_lenth_list)
len(lstm_full_lenth_list)


lenth_count = 0
for i in lstm_lenth_list:
    if i>1500:
        lenth_count = lenth_count + 1

# lenth_count
# del embeddings_index


# 3 机器学习训练
# 处理格式特征 最长1000
model_max_len = 1500
model_struct_max_len =1000



# 数据和对应类别的下标一起打乱
# Y=to_categorical(opinion_train_stc, len(class_index))
# x_train, x_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.1)
# x_train, x_test, y_train_index, y_test_index = train_test_split(X_train, Y_train_index, test_size=0.1)

struct_list_input = []
for i in range(len(struct_list)):
    tmp = []
    for j in range(len(struct_list[i])):
        for k in range(len(struct_list[i][j])):
            tmp.append(struct_list[i][j][k])
    struct_list_input.append(tmp)


print(len(struct_list_input[0]))
# 结构后面补0， 后面截断
# X_train_txt_full = pad_sequences(X_train_txt_full, maxlen=model_max_len, padding='post', truncating='post')
X_train_full = pad_sequences(X_train_full, maxlen=model_max_len, padding='post', truncating='post')

print(struct_list_input[0])
print(len(struct_list_input[0]))

# test_train,test_test,test_a_train,test_a_test = train_test_split(test1,test_a ,test_size=0.5)

# x_train, x_test, y_train_index, y_test_index, X_train_full_train, X_train_full_test, \
# X_train_txt_train, X_train_txt_test, \
# X_train_txt_full_train, X_train_txt_full_test, \
# struct_list_train, struct_list_test, \
# svm_y_train, svm_y_test, \
# filename_train, filename_test\
#  = train_test_split(X_train, Y_train_index, X_train_full, 
#  X_train_txt, 
#  X_train_txt_full, 
#  struct_list_input, 
#  opinion_list, 
#  filename_list,test_size=0.1)


x_train = []
x_test = []
y_train_index = []
y_test_index = []
X_train_full_train = []
X_train_full_test = []
X_train_txt_train = []
X_train_txt_test = []
X_train_txt_full_train = []
X_train_txt_full_test = []
struct_list_train = []
struct_list_test = []
svm_y_train = []
svm_y_test = []
filename_train = []
filename_test = []

# 从每一类里面都按 1：9 分为测试集 训练集
for i in range(len(class_index)):
    # print(i)
    tmp_X_train = []
    tmp_Y_train_index = []
    tmp_X_train_full = []
    tmp_X_train_txt = []
    tmp_X_train_txt_full = []
    tmp_struct_list_input = []
    tmp_opinion_list = []
    tmp_filename_list = []
    for tmpindex in range(len(opinion_list)):  # 取出第i类 
        # print(tmpindex)
        if opinion_list[tmpindex] == i:
            # print(tmpindex)
            tmp_X_train.append(X_train[tmpindex])
            tmp_Y_train_index.append(Y_train_index[tmpindex])
            tmp_X_train_full.append(X_train_full[tmpindex])
            tmp_X_train_txt.append(X_train_txt[tmpindex])
            tmp_X_train_txt_full.append(X_train_txt_full[tmpindex])
            tmp_struct_list_input.append(struct_list_input[tmpindex])
            tmp_opinion_list.append(opinion_list[tmpindex])
            tmp_filename_list.append(filename_list[tmpindex])
    tmp_x_train, tmp_x_test, tmp_y_train_index, tmp_y_test_index, tmp_X_train_full_train, tmp_X_train_full_test, \
    tmp_X_train_txt_train, tmp_X_train_txt_test, \
    tmp_X_train_txt_full_train, tmp_X_train_txt_full_test, \
    tmp_struct_list_train, tmp_struct_list_test, \
    tmp_svm_y_train, tmp_svm_y_test, \
    tmp_filename_train, tmp_filename_test\
    = train_test_split(tmp_X_train, tmp_Y_train_index, tmp_X_train_full, 
    tmp_X_train_txt, 
    tmp_X_train_txt_full, 
    tmp_struct_list_input, 
    tmp_opinion_list, 
    tmp_filename_list,test_size=0.1)
    for tmpindex in range(len(tmp_x_train)):
        x_train.append(tmp_x_train[tmpindex])
        y_train_index.append(tmp_y_train_index[tmpindex])
        X_train_full_train.append(tmp_X_train_full_train[tmpindex])
        X_train_txt_train.append(tmp_X_train_txt_train[tmpindex])
        X_train_txt_full_train.append(tmp_X_train_txt_full_train[tmpindex])
        struct_list_train.append(tmp_struct_list_train[tmpindex])
        svm_y_train.append(tmp_svm_y_train[tmpindex])
        filename_train.append(tmp_filename_train[tmpindex])
    for tmpindex in range(len(tmp_x_test)):
        x_test.append(tmp_x_test[tmpindex])
        y_test_index.append(tmp_y_test_index[tmpindex])
        X_train_full_test.append(tmp_X_train_full_test[tmpindex])
        X_train_txt_test.append(tmp_X_train_txt_test[tmpindex])
        X_train_txt_full_test.append(tmp_X_train_txt_full_test[tmpindex])
        struct_list_test.append(tmp_struct_list_test[tmpindex])
        svm_y_test.append(tmp_svm_y_test[tmpindex])
        filename_test.append(tmp_filename_test[tmpindex])

print(shape(x_train))
print(shape(x_test))

#不去重特征


vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b", min_df = 15)
# vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X = vectorizer.fit_transform(content_train_src)
# X = vectorizer.fit_transform(X_train_txt_full)


transform = TfidfTransformer()
x_tfidf = transform.fit_transform(X)


feature_path = '/home/yangc/lstm_model/vectorizer.pkl'
with open(feature_path, 'wb') as fw:
     pickle.dump(vectorizer.vocabulary_, fw)


tfidftransformer_path = '/home/yangc/lstm_model/transform.pkl'
with open(tfidftransformer_path, 'wb') as fw:
     pickle.dump(transform, fw)


# X_train_txt_full_train, X_train_txt_full_test
svm_x_train_txt=[]
svm_x_test_txt=[]

for words in X_train_txt_full_train:
    tmpstr = ""
    for word in words:
        tmpstr = tmpstr + " " + word
    svm_x_train_txt.append(tmpstr)



X_train_vec = vectorizer.transform(svm_x_train_txt)
svm_x_train_input = transform.transform(X_train_vec)
#(9356, 56909)


for words in X_train_txt_full_test:
    tmpstr = ""
    for word in words:
        tmpstr = tmpstr + " " + word
    svm_x_test_txt.append(tmpstr)

X_test_vec = vectorizer.transform(svm_x_test_txt)
svm_x_test_input = transform.transform(X_test_vec)



print(shape(svm_x_train_input)) #(457, 62919)(9356, 56909)
print(shape(svm_y_train))
clf = SVC(probability = True)
clf.fit(svm_x_train_input,svm_y_train)

# print(svm_x_test_input[0])
print(shape(svm_x_test_input)) #(1067, 56909)
print(shape(svm_y_test))

pred_y  = clf.predict(svm_x_test_input)
pred_y_proba  = clf.predict_proba(svm_x_test_input)
print(classification_report(svm_y_test, pred_y))




pred_savepath = "/public/ycdswork/lstm_pred/pred_y_proba.npy"
np.save(pred_savepath, pred_y_proba)



# lstm -----------------------------------------------------------
# lstm
# lstm
# lstm
# lstm
# 处理标签格式 从svm的格式改成lstm需要的格式

lstm_y_train=to_categorical(svm_y_train, len(class_index))       #将标签处理成输入的格式
lstm_y_test=to_categorical(svm_y_test, len(class_index))



# lstm 不需要先取词向量 只用取下标index 需要两份输入 一份结构struct  一份内容content 
#结构输入在前  内容输入在后面 总长度一样 少的部分补0
lstm_input_content_train = []
lstm_input_struct_train = []
for i in range(len(X_train_full_train)):
    tmp = []
    for j in range(len(X_train_full_train[i])):
        if X_train_full_train[i][j] != 0:
            tmp.append(X_train_full_train[i][j])
    tmplen = len(tmp)
    lstm_input_content_train.append(tmp)
    tmp = []
    for j in range(len(struct_list_train[i])):
        if struct_list_train[i][j] != 0:
            tmp.append(struct_list_train[i][j])
    for j in range(tmplen):
        tmp.append(0)
    lstm_input_struct_train.append(tmp)




lstm_input_content_test = []
lstm_input_struct_test = []
for i in range(len(X_train_full_test)):
    tmp = []
    for j in range(len(X_train_full_test[i])):
        if X_train_full_test[i][j] != 0:
            tmp.append(X_train_full_test[i][j])
    tmplen = len(tmp)
    lstm_input_content_test.append(tmp)
    tmp = []
    for j in range(len(struct_list_test[i])):
        if struct_list_test[i][j] != 0:
            tmp.append(struct_list_test[i][j])
    for j in range(tmplen):
        tmp.append(0)
    lstm_input_struct_test.append(tmp)

# lstm处理时 需要在前面补0
lstm_input_content_train = pad_sequences(lstm_input_content_train, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
lstm_input_struct_train = pad_sequences(lstm_input_struct_train, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
lstm_input_content_test = pad_sequences(lstm_input_content_test, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
lstm_input_struct_test = pad_sequences(lstm_input_struct_test, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')



print(shape(lstm_input_struct_train)) #(457, 62919)

# SVM 加入结构特征_________________________________

print(shape(svm_x_train_input)) #(457, 62919)(9356, 56909)

# 稀疏矩阵转稠密矩阵
svm_x_train_input_dense = svm_x_train_input.todense()
svm_x_test_input_dense = svm_x_test_input.todense()
print(shape(svm_x_train_input))
print(shape(svm_x_train_input_dense))
print(shape(svm_x_test_input))
print(shape(svm_x_test_input_dense))

svm_x_train_input_array = svm_x_train_input_dense.getA()
svm_x_test_input_array = svm_x_test_input_dense.getA()

print(shape(svm_x_train_input_array)) 
print(shape(svm_x_test_input_array)) 

# (9356, 56909)
# >>> print(shape(svm_x_test_input_array)) 
# (1067, 56909

svm_x_train_input_struct = []
svm_x_test_input_struct = []



svm_struct_list_input = []
for i in range(len(struct_list)):
    tmp = []
    for j in range(len(struct_list[i])):
        for k in range(len(struct_list[i][j])):
            tmp.append(struct_list[i][j][k])
    svm_struct_list_input.append(tmp)


struct_list_input_list_word = []
for words in svm_struct_list_input:
    tmplist = ""
    for word in words:
        tmplist = tmplist + " " + str(word)
    struct_list_input_list_word.append(tmplist)

# 重新生成svm的网页结构
svm_struct_train=[]
svm_struct_test=[]

for tmp_filename in filename_train:
    for i in range(len(filename_list)):
        if filename_list[i] == tmp_filename:
            svm_struct_train.append(struct_list[i])
            break


for tmp_filename in filename_test:
    for i in range(len(filename_list)):
        if filename_list[i] == tmp_filename:
            svm_struct_test.append(struct_list[i])
            break


struct_vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b")
X_struct = struct_vectorizer.fit_transform(struct_list_input_list_word)
# X = vectorizer.fit_transform(X_train_txt_full)

struct_vectorizer.get_feature_names()

struct_transform = TfidfTransformer()
struct_x_tfidf = struct_transform.fit_transform(X_struct)



struct_feature_path = '/home/yangc/lstm_model/struct_vectorizer.pkl'
with open(struct_feature_path, 'wb') as fw:
     pickle.dump(struct_vectorizer.vocabulary_, fw)


struct_tfidftransformer_path = '/home/yangc/lstm_model/struct_transform.pkl'
with open(struct_tfidftransformer_path, 'wb') as fw:
     pickle.dump(struct_transform, fw)


struct_list_train_word = []
for words in svm_struct_train:
    tmplist = ""
    for word in words:
        tmplist = tmplist + " " + str(word)
    struct_list_train_word.append(tmplist)


struct_list_test_word = []
for words in svm_struct_test:
    tmplist = ""
    for word in words:
        tmplist = tmplist + " " + str(word)
    struct_list_test_word.append(tmplist)


X_train_vec_struct = struct_vectorizer.transform(struct_list_train_word)
svm_x_train_struct = struct_transform.transform(X_train_vec_struct)

X_test_vec_struct = struct_vectorizer.transform(struct_list_test_word)
svm_x_test_struct = struct_transform.transform(X_test_vec_struct)


# 结构特征tfidf矩阵转为list
svm_x_train_struct_dense = svm_x_train_struct.todense()
svm_x_test_struct_dense = svm_x_test_struct.todense()

svm_x_train_struct_dense_array = svm_x_train_struct_dense.getA()
svm_x_test_struct_dense_array = svm_x_test_struct_dense.getA()

del svm_x_train_struct_dense
del svm_x_test_struct_dense

for i in range(shape(svm_x_train_input_dense)[0]):
    tmpinput = []
    for tmpnum in svm_x_train_input_array[i]:
        tmpinput.append(tmpnum)
    for tmpnum in svm_x_train_struct_dense_array[i]:
        tmpinput.append(tmpnum)
    svm_x_train_input_struct.append(tmpinput)


for i in range(shape(svm_x_test_input)[0]):
    tmpinput = []
    for tmpnum in svm_x_test_input_array[i]:
        tmpinput.append(tmpnum)
    for tmpnum in svm_x_test_struct_dense_array[i]:
        tmpinput.append(tmpnum)
    svm_x_test_input_struct.append(tmpinput)


print(shape(svm_x_train_input_array))

print(shape(svm_x_train_input_struct))
print(shape(svm_x_test_input_struct))

import scipy.sparse.csr
#转为稀疏矩阵
svm_x_train_input_struct_csr = scipy.sparse.csr.csr_matrix(svm_x_train_input_struct)
svm_x_test_input_struct_csr = scipy.sparse.csr.csr_matrix(svm_x_test_input_struct)

del svm_x_train_input_struct
del svm_x_test_input_struct

print(shape(svm_x_train_input_struct_csr))
print(shape(svm_x_test_input_struct_csr))


clf_struct = SVC(probability = True)
clf_struct.fit(svm_x_train_input_struct_csr, svm_y_train)


# print(svm_x_test_input[0])
print(shape(svm_y_test))


pred_y_struct  = clf_struct.predict(svm_x_test_input_struct_csr)
pred_y_proba_struct  = clf_struct.predict_proba(svm_x_test_input_struct_csr)
print(classification_report(svm_y_test, pred_y_struct))



pred_savepath = "/public/ycdswork/lstm_pred/pred_y_proba_struct.npy"
np.save(pred_savepath, pred_y_proba_struct)



svm_model_savepath = "/home/yangc/lstm_model/svm_model"

with open(svm_model_savepath,'wb') as f: 
    pickle.dump(clf,f) #将训练好的模型clf存储在变量f中，且保存到本地

with open(svm_model_savepath,'rb') as f:  
    clf_load = pickle.load(f)  #将模型存储在变量clf_load中  
    # print(clf_load.predict(X[0:1000])) #调用模型并预测结果

svm_struct_model_savepath = "/home/yangc/lstm_model/svm_struct_model"

with open(svm_struct_model_savepath,'wb') as f: 
    pickle.dump(clf_struct,f) #将训练好的模型clf存储在变量f中，且保存到本地
# SVM 加入结构特征 结束 ————————————————————————————————————————————————————





STRUCT_EMBEDDING_DIM = 50








# 去重特征 集成方法

struct_list_input = pad_sequences(struct_list_input, maxlen=model_struct_max_len, padding='post',truncating='post')

# lstm 不需要先取词向量 只用取下标index 需要两份输入 一份结构struct  一份内容content 
#结构输入在前  内容输入在后面 总长度一样 少的部分补0
lstm_input_content_nofull_train = []
lstm_input_struct_nofull_train = []
for i in range(len(x_train)):
    tmp = []
    for j in range(len(x_train[i])):
        if x_train[i][j] != 0:
            tmp.append(x_train[i][j])
    tmplen = len(tmp)
    lstm_input_content_nofull_train.append(tmp)
    tmp = []
    for j in range(len(struct_list_train[i])):
        if struct_list_train[i][j] != 0:
            tmp.append(struct_list_train[i][j])
    for j in range(tmplen):
        tmp.append(0)
    lstm_input_struct_nofull_train.append(tmp)

lstm_input_content_nofull_train[0]
lstm_input_struct_nofull_train[0]


lstm_input_content_nofull_test = []
lstm_input_struct_nofull_test = []
for i in range(len(x_test)):
    tmp = []
    for j in range(len(x_test[i])):
        if x_test[i][j] != 0:
            tmp.append(x_test[i][j])
    tmplen = len(tmp)
    lstm_input_content_nofull_test.append(tmp)
    tmp = []
    for j in range(len(struct_list_test[i])):
        if struct_list_test[i][j] != 0:
            tmp.append(struct_list_test[i][j])
    for j in range(tmplen):
        tmp.append(0)
    lstm_input_struct_nofull_test.append(tmp)

# lstm处理时 需要在前面补0
lstm_input_content_nofull_train = pad_sequences(lstm_input_content_nofull_train, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
lstm_input_struct_nofull_train = pad_sequences(lstm_input_struct_nofull_train, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
lstm_input_content_nofull_test = pad_sequences(lstm_input_content_nofull_test, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
lstm_input_struct_nofull_test = pad_sequences(lstm_input_struct_nofull_test, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')


len(lstm_input_content_train[0])












# 3.3 训练  batch size256, learning rate 0.002, and layer number 3
def model_fit_16(model, x_content,x_struct, y):
    return model.fit({'content': x_content, 'struct':x_struct}, y, batch_size=16, epochs=10,  shuffle=True,validation_data=([lstm_input_content_test, lstm_input_struct_test],lstm_y_test))

def model_fit_64(model, x_content,x_struct, y):
    return model.fit({'content': x_content, 'struct':x_struct}, y, batch_size=64, epochs=10,  shuffle=True,validation_data=([lstm_input_content_test, lstm_input_struct_test],lstm_y_test))

# model.fit({'input_1': x1,'input_2': x2}, {'output011': y_1,'output02': y_2, 'output03': y_3}, epochs=50, batch_size=32, validation_split=0.1)

def my_model_fit_16(model, x, y):
    # return model.fit(x, y, batch_size=16, epochs=10, validation_split=0.04, shuffle=True)
    return model.fit(x, y, batch_size=16, epochs=10,  shuffle=True, validation_data=(lstm_input_content_test,lstm_y_test))


def my_model_fit_64(model, x, y):
    # return model.fit(x, y, batch_size=16, epochs=10, validation_split=0.04, shuffle=True)
    return model.fit(x, y, batch_size=64, epochs=10, shuffle=True, validation_data=(lstm_input_content_test, lstm_y_test))

# 去重
def my_model_nofull_fit_16(model, x, y):
    return model.fit(x, y, batch_size=16, epochs=10,  shuffle=True, validation_data=(lstm_input_content_nofull_test,lstm_y_test))


def my_model_nofull_fit_64(model, x, y):
    return model.fit(x, y, batch_size=64, epochs=10, shuffle=True, validation_data=(lstm_input_content_nofull_test, lstm_y_test))

def model_nofull_fit_16(model, x_content,x_struct, y):
    return model.fit({'content': x_content, 'struct':x_struct}, y, batch_size=16, epochs=10,  shuffle=True,validation_data=([lstm_input_content_nofull_test, lstm_input_struct_nofull_test],lstm_y_test))

def model_nofull_fit_64(model, x_content,x_struct, y):
    return model.fit({'content': x_content, 'struct':x_struct}, y, batch_size=64, epochs=10,  shuffle=True,validation_data=([lstm_input_content_nofull_test, lstm_input_struct_nofull_test],lstm_y_test))






my_lstm_model = get_my_lstm_model()
my_lstm_model_train = my_model_fit_64(my_lstm_model, lstm_input_content_train, lstm_y_train)
my_lstm_model_64_pred = my_lstm_model.predict(lstm_input_content_test, batch_size=None, verbose=1, steps=None)

my_lstm_model_64_pred_class = np.argmax(my_lstm_model_64_pred, axis=1)
print(classification_report(svm_y_test, my_lstm_model_64_pred_class))


# my_lstm_model_savepath = "/home/yangc/lstm_model/my_lstm_model"
# my_lstm_model.save(my_lstm_model_savepath)

# del my_lstm_model

# from keras.models import load_model
# my_lstm_model = load_model(my_lstm_model_savepath)

# print(classification_report(svm_y_test, pred_y_struct))
my_lstm_model_16 = get_my_lstm_model()
my_lstm_model_train_16 = my_model_fit_16(my_lstm_model_16, lstm_input_content_train, lstm_y_train)
my_lstm_model_16_pred = my_lstm_model_16.predict(lstm_input_content_test, batch_size=None, verbose=1, steps=None)






# 双特征LSTM 效果不好
def get_lstm_model():
    # define two sets of inputs
    input_content = Input(shape=(model_struct_max_len + model_max_len, ), dtype='int32',name = "content")
    input_struct= Input(shape=(model_struct_max_len + model_max_len,), dtype='int32',name = "struct")
    content_embedding = Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(input_content) #不能训练
    content_embedding = keras.Model(inputs=input_content, outputs=content_embedding)
    struct_embedding = Embedding(input_dim = TAG_length + 1,
                            output_dim =STRUCT_EMBEDDING_DIM,
                            mask_zero = True,
                            trainable=True)(input_struct) #可以训练
    struct_embedding = keras.Model(inputs=input_struct, outputs=struct_embedding)
    concate =  concatenate([content_embedding.output,struct_embedding.output], axis=-1)
    z = LSTM(STRUCT_EMBEDDING_DIM+EMBEDDING_DIM, dropout=0.5)(concate)
    z = Dense(128, activation="relu")(z)
    z = Dense(len(class_index), activation='softmax')(z)
    model = keras.Model(inputs=[content_embedding.input, struct_embedding.input], outputs=z)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 双特征LSTM 效果不好
lstm_model = get_lstm_model()
lstm_model_train = model_fit_64(lstm_model, lstm_input_content_train, lstm_input_struct_train, lstm_y_train)




tf.compat.v1.disable_eager_execution()
class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
        
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


# 单特征双向LSTM
def my_bilstm_model(hidden_size, attention_size):
	# 输入层
    inputs = Input(shape=(model_max_len+model_struct_max_len,), dtype='int32')
    # Embedding层
    x = Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(inputs)
    # BiLSTM层
    x = Bidirectional(LSTM(hidden_size, dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional()(x)
    # Attention层
    x = AttentionLayer(attention_size=attention_size)(x)
    # 输出层
    outputs = Dense(len(class_index), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model








#双特征双向LSTM+ attention
def get_bilstm_model(content_attention_size, struct_attention_size):
    input_content  = Input(shape=(model_struct_max_len + model_max_len, ), dtype='int32',name = "content")
    input_struct= Input(shape=(model_struct_max_len + model_max_len,), dtype='int32', name = "struct")
    content_embedding = Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(input_content) #不能训练
    content_embedding = Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.5, return_sequences=True))(content_embedding)
    content_embedding = AttentionLayer(attention_size=content_attention_size)(content_embedding)
    # content_embedding = LSTM(EMBEDDING_DIM, dropout=0.5)(content_embedding)
    content_LSTM = keras.Model(inputs=input_content, outputs=content_embedding)
    struct_embedding = Embedding(input_dim = TAG_length + 1,
                            output_dim =STRUCT_EMBEDDING_DIM,
                            mask_zero = True,
                            trainable=True)(input_struct) #可以训练
    # struct_embedding = LSTM(STRUCT_EMBEDDING_DIM, dropout=0.5)(struct_embedding)
    struct_embedding = Bidirectional(LSTM(STRUCT_EMBEDDING_DIM, dropout=0.5, return_sequences=True))(struct_embedding)
    struct_embedding = AttentionLayer(attention_size=struct_attention_size)(struct_embedding)
    struct_LSTM = keras.Model(inputs=input_struct, outputs=struct_embedding)
    concate =  concatenate([content_LSTM.output,struct_LSTM.output], axis=-1)
    # z = LSTM(STRUCT_EMBEDDING_DIM+EMBEDDING_DIM, dropout=0.5)(concate)
    z = Dense(128, activation="relu")(concate)
    z = Dense(len(class_index), activation='softmax')(z)
    model = keras.Model(inputs=[content_LSTM.input, struct_LSTM.input], outputs=z)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



bilstmmodel = get_bilstm_model(100,25)
# bilstmmodel_train = model_fit(bilstmmodel, x_train_raw, y_train)
bilstmmodel_train = model_fit_64(bilstmmodel, lstm_input_content_train, lstm_input_struct_train, lstm_y_train)

print(bilstmmodel.evaluate([lstm_input_content_test, lstm_input_struct_test], lstm_y_test))



#去重特征



# 单特征LSTM
my_lstm_model_nofull = get_my_lstm_model()
my_lstm_model_nofull_train = my_model_nofull_fit_64(my_lstm_model_nofull, lstm_input_content_nofull_train, lstm_y_train)

print(my_lstm_model_nofull.evaluate(lstm_input_content_nofull_test, lstm_y_test))


# 双特征LSTM 效果不好
# lstm_model_nofull = get_lstm_model()
# lstm_model_nofull_train = model_fit(lstm_model_nofull, lstm_input_content_nofull_train, lstm_input_struct_nofull_train, lstm_y_train)
# print(lstm_model_nofull.evaluate([lstm_input_content_nofull_test, lstm_input_struct_nofull_test], lstm_y_test))


#  双特征LSTM 复现改进
lstm_model_nofull_1 = get_lstm_model_1()
lstm_model_nofull_train_1 = model_fit(lstm_model_nofull_1, lstm_input_content_nofull_train, lstm_input_struct_nofull_train, lstm_y_train)
print(lstm_model_nofull_1.evaluate([lstm_input_content_nofull_test, lstm_input_struct_nofull_test], lstm_y_test))




tf.compat.v1.disable_eager_execution()  # 避免未知错误
class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
        
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]

from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply,Permute
SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    # if SINGLE_ATTENTION_VECTOR:
    #     a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
    #     a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

from keras import initializers
class AttentionLayer_1(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """
    def __init__(self, **kwargs):
        self.init = initializers.get('glorot_uniform')
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.W = self.add_weight(name='Attention_Weight',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer=self.init,
                                 trainable=True)
        self.b = self.add_weight(name='Attention_Bias',
                                 shape=(input_shape[-1], ),
                                 initializer=self.init,
                                 trainable=True)
        self.u = self.add_weight(name='Attention_Context_Vector',
                                 shape=(input_shape[-1], 1),
                                 initializer=self.init,
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    def call(self, x):
        # refer to the original paper
        # link: https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf
        
        # RNN 구조를 거쳐서 나온 hidden states (x)에 single layer perceptron (tanh activation)
        # 적용하여 나온 벡터가 uit 
        u_it = K.tanh(K.dot(x, self.W) + self.b)
        
        # uit와 uw (혹은 us) 간의 similarity를 attention으로 사용
        # softmax를 통해 attention 값을 확률 분포로 만듬
        a_it = K.dot(u_it, self.u)
        a_it = K.squeeze(a_it, -1)
        a_it = K.softmax(a_it)
        return a_it
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])

class MyAttention(Layer):
    def __init__(self, out_dim,key_size=8, **kwargs):
        super(MyAttention, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.key_size=key_size
    def build(self, input_shape):
        super(MyAttention, self).build(input_shape)
        input_shape = list(input_shape)
        if input_shape[1]==None:
            input_shape[1]=1
        kernel_initializer = 'glorot_uniform'
        kernel_regularizer = None
        kernel_constraint = None
        self.qw = self.add_weight(name='qw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                      trainable=True)
        self.kw = self.add_weight(name='kw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
        self.vw = self.add_weight(name='vw',
                                  shape=(input_shape[1], self.out_dim),
                                  # shape=(1, self.out_dim),
                                  initializer=kernel_initializer,
                                  regularizer=kernel_regularizer,
                                  constraint=kernel_constraint,
                                  trainable=True)
    def call(self, inputs):
        input_size = tf.shape(inputs)
        q = tf.multiply(inputs, self.qw)
        k = K.permute_dimensions(tf.multiply(inputs, self.kw),(0,2,1))
        v = tf.multiply(inputs, self.vw)
        v = tf.reshape(tf.tile(v,[1,input_size[1],1]),(input_size[0],input_size[1],input_size[1],self.out_dim))
        p=tf.matmul(q,k)
        p=tf.reshape(K.softmax(p/np.sqrt(self.key_size)),(input_size[0],input_size[1],input_size[1],1))
        v=tf.reduce_sum(tf.multiply(v, p), 2)
        return v
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1], self.out_dim)


#双特征双向LSTM+ attention
bilstmmodel = get_bilstm_model(100,25)
# bilstmmodel_train = model_fit(bilstmmodel, x_train_raw, y_train)
bilstmmodel_train = model_fit(bilstmmodel, lstm_input_content_nofull_train, lstm_input_struct_nofull_train, lstm_y_train)

print(bilstmmodel.evaluate([lstm_input_content_nofull_test, lstm_input_struct_nofull_test], lstm_y_test))



def my_bilstm_model_1(hidden_size, attention_size):
	# 输入层
    inputs = Input(shape=(model_max_len+model_struct_max_len,), dtype='int32')
    # Embedding层
    x = Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(inputs)
    # BiLSTM层
    x = Bidirectional(LSTM(hidden_size, dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional()(x)
    # Attention层
    x = AttentionLayer()(x)
    # 输出层
    outputs = Dense(len(class_index), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='Adadelta', metrics=['accuracy'])
    return model






# 开始训练++++++++===========================================================

#1 单特征双向LSTM 64 去重
mybilstmmodel_nofull_64 = my_bilstm_model(200,100)
mybilstmmodel_nofull_64_train = my_model_nofull_fit_64(mybilstmmodel_nofull_64, lstm_input_content_nofull_train, lstm_y_train)
mybilstmmodel_nofull_64_pred = mybilstmmodel_nofull_64.predict(lstm_input_content_nofull_test, batch_size=None, verbose=1, steps=None)
mybilstmmodel_nofull_64_pred_class = np.argmax(mybilstmmodel_nofull_64_pred, axis=1)
print(classification_report(svm_y_test, mybilstmmodel_nofull_64_pred_class))



test = classification_report(svm_y_test, mybilstmmodel_nofull_64_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score



#保存模型到 lstm_model 保存测试的预测得分结果到 lstm_pred
model_savepath = "/public/ycdswork/lstm_model/mybilstmmodel_nofull_64"
mybilstmmodel_nofull_64.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/mybilstmmodel_nofull_64_pred.npy"
np.save(pred_savepath, mybilstmmodel_nofull_64_pred)






#2 单特征双向LSTM 64 不去重
mybilstmmodel_64 = my_bilstm_model(200,100)
mybilstmmodel_train = my_model_fit_64(mybilstmmodel_64, lstm_input_content_train, lstm_y_train)
mybilstmmodel_64_pred = mybilstmmodel_64.predict(lstm_input_content_test, batch_size=None, verbose=1, steps=None)

mybilstmmodel_64_pred_class = np.argmax(mybilstmmodel_64_pred, axis=1)
print(classification_report(svm_y_test, mybilstmmodel_64_pred_class))



test = classification_report(svm_y_test, mybilstmmodel_64_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score



model_savepath = "/public/ycdswork/lstm_model/mybilstmmodel_64/mybilstmmodel_64"
mybilstmmodel_64.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/mybilstmmodel_64_pred.npy"
np.save(pred_savepath, mybilstmmodel_64_pred)



#3 单特征双向LSTM 16 去重
mybilstmmodel_nofull_16 = my_bilstm_model(200,100)
mybilstmmodel_nofull_16_train = my_model_nofull_fit_16(mybilstmmodel_nofull_16, lstm_input_content_nofull_train, lstm_y_train)
mybilstmmodel_nofull_16_pred = mybilstmmodel_nofull_16.predict(lstm_input_content_nofull_test, batch_size=None, verbose=1, steps=None)
mybilstmmodel_nofull_16_pred_class = np.argmax(mybilstmmodel_nofull_16_pred, axis=1)
print(classification_report(svm_y_test, mybilstmmodel_nofull_16_pred_class))



test = classification_report(svm_y_test, mybilstmmodel_nofull_16_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score


model_savepath = "/public/ycdswork/lstm_model/mybilstmmodel_nofull_16/mybilstmmodel_nofull_16"
mybilstmmodel_nofull_16.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/mybilstmmodel_nofull_16_pred.npy"
np.save(pred_savepath, mybilstmmodel_nofull_16_pred)




#4 单特征双向LSTM 16 不去重
mybilstmmodel_16 = my_bilstm_model(200,100)
mybilstmmodel_train = my_model_fit_16(mybilstmmodel_16, lstm_input_content_train, lstm_y_train)
mybilstmmodel_16_pred = mybilstmmodel_16.predict(lstm_input_content_test, batch_size=None, verbose=1, steps=None)

mybilstmmodel_16_pred_class = np.argmax(mybilstmmodel_16_pred, axis=1)
print(classification_report(svm_y_test, mybilstmmodel_16_pred_class))



test = classification_report(svm_y_test, mybilstmmodel_16_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass

tmpkey
precision
recall
f1score



model_savepath = "/public/ycdswork/lstm_model/mybilstmmodel_16/mybilstmmodel_16"
mybilstmmodel_16.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/mybilstmmodel_16_pred.npy"
np.save(pred_savepath, mybilstmmodel_16_pred)


# 保存测试

mybilstmmodel_16_model_savepath = "/home/yangc/lstm_model/mybilstmmodel_16"
mybilstmmodel_16.save_weights(mybilstmmodel_16_model_savepath)

# model.load_weights(model_save_path)

# del my_lstm_model

# from keras.models import load_model
# my_lstm_model = load_model(my_lstm_model_savepath)

# print(classification_report(svm_y_test, pred_y_struct))


#测试 加载权重
mybilstmmodel_test = my_bilstm_model(200,100)
model_savepath = "/public/ycdswork/lstm_model/mybilstmmodel_16/mybilstmmodel_16"
mybilstmmodel_test.load_weights(model_savepath)
mybilstmmodel_test_pred = mybilstmmodel_test.predict(lstm_input_content_nofull_test)
mybilstmmodel_test_pred_class = np.argmax(mybilstmmodel_test_pred, axis=1)
list(mybilstmmodel_test_pred_class)




#5 双特征双向LSTM 16 去重

bilstmmodel_nofull_16 = get_bilstm_model(100,25)
# bilstmmodel_train = model_fit(bilstmmodel, x_train_raw, y_train)
bilstmmodel_train = model_fit_16(bilstmmodel_nofull_16, lstm_input_content_nofull_train, lstm_input_struct_nofull_train, lstm_y_train)

bilstmmodel_nofull_16_pred = bilstmmodel_nofull_16.predict([lstm_input_content_nofull_test, lstm_input_struct_nofull_test], batch_size=None, verbose=1, steps=None)

bilstmmodel_nofull_16_pred_class = np.argmax(bilstmmodel_nofull_16_pred, axis=1)
print(classification_report(svm_y_test, bilstmmodel_nofull_16_pred_class))


test = classification_report(svm_y_test, bilstmmodel_nofull_16_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score


model_savepath = "/public/ycdswork/lstm_model/bilstmmodel_nofull_16/bilstmmodel_nofull_16"
bilstmmodel_nofull_16.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/bilstmmodel_nofull_16_pred.npy"
np.save(pred_savepath, bilstmmodel_nofull_16_pred)




#6 双特征双向LSTM 16 不去重

bilstmmodel_16 = get_bilstm_model(100,25)
# bilstmmodel_train = model_fit(bilstmmodel, x_train_raw, y_train)
bilstmmodel_train = model_fit_16(bilstmmodel_16, lstm_input_content_train, lstm_input_struct_train, lstm_y_train)


bilstmmodel_16_pred = bilstmmodel_16.predict([lstm_input_content_test, lstm_input_struct_test], batch_size=None, verbose=1, steps=None)


bilstmmodel_16_pred_class = np.argmax(bilstmmodel_16_pred, axis=1)

print(classification_report(svm_y_test, bilstmmodel_16_pred_class))



test = classification_report(svm_y_test, bilstmmodel_16_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score


model_savepath = "/public/ycdswork/lstm_model/bilstmmodel_16/bilstmmodel_16"
bilstmmodel_16.save_weights(model_savepath)


pred_savepath = "/public/ycdswork/lstm_pred/bilstmmodel_16_pred.npy"
np.save(pred_savepath, bilstmmodel_16_pred)


# 单特征单向LSTM 16 不去重

# 单特征LSTM
def get_my_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False))
    model.add(LSTM(200, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_index), activation='softmax'))
    model.build((None,model_max_len, EMBEDDING_DIM))
    model.summary()
    # tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model


#7 单特征 不去重 16
get_my_lstm_model_16 = get_my_lstm_model()
get_my_lstm_model_16_train = my_model_fit_16(get_my_lstm_model_16, lstm_input_content_train, lstm_y_train)
get_my_lstm_model_16_pred = get_my_lstm_model_16.predict(lstm_input_content_test, batch_size=None, verbose=1, steps=None)

get_my_lstm_model_16_pred_class = np.argmax(get_my_lstm_model_16_pred, axis=1)
print(classification_report(svm_y_test, get_my_lstm_model_16_pred_class))



test = classification_report(svm_y_test, get_my_lstm_model_16_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score


model_savepath = "/public/ycdswork/lstm_model/get_my_lstm_model_16/get_my_lstm_model_16"
get_my_lstm_model_16.save_weights(model_savepath)


pred_savepath = "/public/ycdswork/lstm_pred/get_my_lstm_model_16_pred.npy"
np.save(pred_savepath, get_my_lstm_model_16_pred)



#8  单特征单LSTM 去重 16

get_my_lstm_model_16_nofull = get_my_lstm_model()
get_my_lstm_model_16_nofull_train = my_model_fit_16(get_my_lstm_model_16_nofull, lstm_input_content_nofull_train, lstm_y_train)
get_my_lstm_model_16_nofull_pred = get_my_lstm_model_16_nofull.predict(lstm_input_content_nofull_test, batch_size=None, verbose=1, steps=None)

get_my_lstm_model_16_nofull_pred_class = np.argmax(get_my_lstm_model_16_nofull_pred, axis=1)
print(classification_report(svm_y_test, get_my_lstm_model_16_nofull_pred_class))


test = classification_report(svm_y_test, get_my_lstm_model_16_nofull_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass


tmpkey
precision
recall
f1score



model_savepath = "/public/ycdswork/lstm_model/get_my_lstm_model_16_nofull/get_my_lstm_model_16_nofull"
get_my_lstm_model_16_nofull.save_weights(model_savepath)


pred_savepath = "/public/ycdswork/lstm_pred/get_my_lstm_model_16_nofull_pred.npy"
np.save(pred_savepath, get_my_lstm_model_16_nofull_pred)




#  双特征单LSTM 复现改进
def get_lstm_model_1():
    # define two sets of inputs
    input_content  = Input(shape=(model_struct_max_len + model_max_len, ), dtype='int32',name = "content")
    input_struct= Input(shape=(model_struct_max_len + model_max_len,), dtype='int32', name = "struct")
    content_embedding = Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(input_content) #不能训练
    content_embedding = LSTM(EMBEDDING_DIM, dropout=0.5)(content_embedding)
    content_LSTM = keras.Model(inputs=input_content, outputs=content_embedding)
    struct_embedding = Embedding(input_dim = TAG_length + 1,
                            output_dim =STRUCT_EMBEDDING_DIM,
                            mask_zero = True,
                            trainable=True)(input_struct) #可以训练
    struct_embedding = LSTM(STRUCT_EMBEDDING_DIM, dropout=0.5)(struct_embedding)
    struct_LSTM = keras.Model(inputs=input_struct, outputs=struct_embedding)
    concate =  concatenate([content_LSTM.output,struct_LSTM.output], axis=-1)
    # z = LSTM(STRUCT_EMBEDDING_DIM+EMBEDDING_DIM, dropout=0.5)(concate)
    z = Dense(128, activation="relu")(concate)
    z = Dense(len(class_index), activation='softmax')(z)
    model = keras.Model(inputs=[content_LSTM.input, struct_LSTM.input], outputs=z)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

#9  双特征单LSTM 复现改进 重复 16

lstm_model_1_16 = get_lstm_model_1()
lstm_model_train_1 = model_fit_16(lstm_model_1_16, lstm_input_content_train, lstm_input_struct_train, lstm_y_train)
# print(lstm_model_1.evaluate([lstm_input_content_test, lstm_input_struct_test], lstm_y_test))
my_lstm_model_1_16_pred = lstm_model_1_16.predict([lstm_input_content_test,lstm_input_struct_test], batch_size=None, verbose=0, steps=None)


my_lstm_model_1_16_pred_class = np.argmax(my_lstm_model_1_16_pred, axis=1)
print(classification_report(svm_y_test, my_lstm_model_1_16_pred_class))


test = classification_report(svm_y_test, my_lstm_model_1_16_pred_class, output_dict=True)
precision = []
recall = []
f1score = []
tmpkey = []
for key in test :
    try:
        tmpkey.append(key)
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass

tmpkey
precision
recall
f1score


model_savepath = "/public/ycdswork/lstm_model/lstm_model_1_16/lstm_model_1_16"
lstm_model_1_16.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/my_lstm_model_1_16_pred.npy"
np.save(pred_savepath, my_lstm_model_1_16_pred)




#10  双特征单LSTM 复现改进 去重 16
lstm_model_1_nofull_16 = get_lstm_model_1()
lstm_model_1_nofull_16_train = model_fit_16(lstm_model_1_nofull_16, lstm_input_content_nofull_train, lstm_input_struct_nofull_train, lstm_y_train)
# print(lstm_model_1.evaluate([lstm_input_content_test, lstm_input_struct_test], lstm_y_test))
lstm_model_1_nofull_16_pred = lstm_model_1_nofull_16.predict([lstm_input_content_nofull_test,lstm_input_struct_nofull_test], batch_size=None, verbose=0, steps=None)


lstm_model_1_nofull_16_pred_class = np.argmax(lstm_model_1_nofull_16_pred, axis=1)
print(classification_report(svm_y_test, lstm_model_1_nofull_16_pred_class))
test = classification_report(svm_y_test, lstm_model_1_nofull_16_pred_class,output_dict=True)
precision = []
recall = []
f1score = []
for key in test :
    try:
        precision.append(test[key]['precision'])
        recall.append(test[key]['recall'])
        f1score.append(test[key]['f1-score'])
    except:
        pass

precision
recall
f1score

model_savepath = "/public/ycdswork/lstm_model/lstm_model_1_nofull_16/lstm_model_1_nofull_16"
lstm_model_1_nofull_16.save_weights(model_savepath)

pred_savepath = "/public/ycdswork/lstm_pred/lstm_model_1_nofull_16_pred.npy"
np.save(pred_savepath, lstm_model_1_nofull_16_pred)


pred_savepath = "/public/ycdswork/lstm_pred/svm_y_test.npy"
np.save(pred_savepath, np.array(svm_y_test))

# np.array(svm_y_test)

















# 计算置信度

# mybilstmmodel_16_pred 置信度
# lstm++++++++++++++++=====================================
lstm_score_list = []
lstm_predict_list = []
# all_class_list = []
# svm_y_test


# bilstmmodel_16_pred
for i in range(len(bilstmmodel_16_pred_class)):
    indextmp = bilstmmodel_16_pred_class[i]
    lstm_score_list.append(bilstmmodel_16_pred[i][indextmp])
    lstm_predict_list.append(indextmp)


# mybilstmmodel_16_pred 单特征双向LSTM 16 不去重
# for i in range(len(mybilstmmodel_16_pred_class)):
#     indextmp = mybilstmmodel_16_pred_class[i]
#     lstm_score_list.append(mybilstmmodel_16_pred[i][indextmp])
#     lstm_predict_list.append(indextmp)


# get_my_lstm_model_16_nofull_pred   单特征单LSTM 去重 16
# for i in range(len(get_my_lstm_model_16_nofull_pred_class)):
#     indextmp = get_my_lstm_model_16_nofull_pred_class[i]
#     lstm_score_list.append(get_my_lstm_model_16_nofull_pred[i][indextmp])
#     lstm_predict_list.append(indextmp)

# lstm_model_1_nofull_16_pred   双特征单LSTM 复现改进 去重 16
# for i in range(len(lstm_model_1_nofull_16_pred_class)):
#     indextmp = lstm_model_1_nofull_16_pred_class[i]
#     lstm_score_list.append(lstm_model_1_nofull_16_pred[i][indextmp])
#     lstm_predict_list.append(indextmp)



# #svm 
# for i in range(len(pred_y)):
#     indextmp = pred_y[i]
#     lstm_score_list.append(pred_y_proba[i][indextmp])
#     lstm_predict_list.append(indextmp)

# pred_y_struct  = clf_struct.predict(svm_x_test_input_struct_csr)
# pred_y_proba_struct  = clf_struct.predict_proba(svm_x_test_input_struct_csr)

# #svm struct
# for i in range(len(pred_y_struct)):
#     indextmp = pred_y_struct[i]
#     lstm_score_list.append(pred_y_proba_struct[i][indextmp])
#     lstm_predict_list.append(indextmp)



sort_list = []
for i in range(len(lstm_score_list)):
    tmplist = []
    tmplist.append(lstm_score_list[i])    # 预测得分
    tmplist.append(lstm_predict_list[i]) # 预测类别
    tmplist.append(svm_y_test[i])        #实际类别
    sort_list.append(tmplist)


# 按预测得分大小排序
def takeFirst(elem):
    return elem[0]

sort_list.sort(key=takeFirst,reverse = True)
print("sort_list")
# print(sort_list)



#   2 排序完计算置信度

lstm_sort_score_list = []
for tmp_list in sort_list:
    lstm_sort_score_list.append(tmp_list[0])

lstm_sort_predict_list = []
for tmp_list in sort_list:
    lstm_sort_predict_list.append(tmp_list[1])

lstm_sort_class_list = []
for tmp_list in sort_list:
    lstm_sort_class_list.append(tmp_list[2]) 


lstm_count_true_list = []    # 按顺序统计正确预测的数量
count_number = 0
for i in range(len(lstm_sort_score_list)):
    if lstm_sort_predict_list[i] == lstm_sort_class_list[i]:
        count_number = count_number + 1
    lstm_count_true_list.append(count_number)


print("lstm_count_true_list")
print(lstm_count_true_list)  #   排序完的数组


confidence_n = 100  #


lstm_confidence_list = []
# 获得置信度
for i in range(len(lstm_sort_score_list)):
    tmpcount = 0
    firstpart = 0
    lastpart = 0
    if i<confidence_n + 1:
        firstpart =  confidence_n - i + lstm_count_true_list[i]
    else:
        firstpart = lstm_count_true_list[i] - lstm_count_true_list[i - confidence_n -1]
    if i + confidence_n >= len(lstm_sort_score_list):
        lastpart = lstm_count_true_list[len(lstm_sort_score_list)-1] - lstm_count_true_list[i]
    else:
        lastpart = lstm_count_true_list[i + confidence_n] - lstm_count_true_list[i]
    tmpconfidence = (firstpart + lastpart)/(2*confidence_n + 1)
    lstm_confidence_list.append(tmpconfidence)

print("svm_confidence_list")
print(lstm_confidence_list)


lstm_sort_score_list






#计算集成学习结果++++++++++====================================


# 拟合情况
# #双特征双向
poly_bilstm_struct = [-6.07477365e+06,  4.74061625e+07, -1.67696546e+08,  3.55822651e+08,
       -5.05042054e+08,  5.06358642e+08, -3.69196420e+08,  1.98595018e+08,
       -7.91442029e+07,  2.32716974e+07, -4.98745079e+06,  7.62102600e+05,
       -8.00377562e+04,  5.41738498e+03, -2.07696643e+02,  3.47163360e+00]
# 单特征双向
poly_bilstm = [-7.93121578e+05 , 6.56098835e+06, -2.45394828e+07,  5.49341900e+07,
 -8.21391343e+07,  8.67041440e+07, -6.65922899e+07,  3.77898251e+07,
 -1.59150147e+07,  4.94515599e+06, -1.11406830e+06 , 1.76239489e+05,
 -1.85988392e+04,  1.20992744e+03 ,-4.22032210e+01,  7.78057032e-01]
#单特征单向
poly_lstm = [-8.68881807e+06,  7.05801423e+07, -2.60242939e+08,  5.76332912e+08,
 -8.54876330e+08,  8.96749075e+08, -6.84735605e+08,  3.85978993e+08,
 -1.61215639e+08,  4.96547960e+07, -1.11302488e+07,  1.77447368e+06,
 -1.93936968e+05,  1.36698511e+04, -5.53632103e+02,  9.82589429e+00]
#双特征单向
poly_lstm_struct = [ 1.47573111e+07, -1.24168700e+08,  4.76193560e+08, -1.10213898e+09,
  1.71817907e+09, -1.90689393e+09,  1.55274765e+09, -9.42214384e+08,
  4.28400984e+08, -1.45532335e+08,  3.65280114e+07, -6.63222999e+06,
  8.40639254e+05, -7.00120601e+04,  3.41642736e+03, -7.33203976e+01]
# svm
poly_svm =[ 5.54785065e+05, -5.23793140e+06,  2.16707018e+07, -5.23211021e+07,
  8.24724830e+07, -8.97060316e+07 , 6.92482958e+07, -3.83692447e+07,
  1.52370790e+07, -4.27820625e+06,  8.26024110e+05, -1.04518489e+05,
  7.97593730e+03, -3.15727151e+02,  6.65986798e+00, -3.23336414e-02]

# svm_struct
poly_svm_struct =[-6.83272574e+04,  6.75185611e+04,  1.27815975e+06, -5.50535618e+06,
  1.09145717e+07, -1.28767585e+07,  9.69415689e+06, -4.65556505e+06,
  1.30835412e+06, -1.29872196e+05, -4.18165302e+04,  1.74837752e+04,
 -2.75422345e+03,  2.11447884e+02, -5.29402352e+00,  7.20447959e-02]




p1_poly_bilstm_struct = np.poly1d(poly_bilstm_struct)
p1_poly_bilstm = np.poly1d(poly_bilstm)

p1_poly_lstm = np.poly1d(poly_lstm)
p1_poly_lstm_struct = np.poly1d(poly_lstm_struct)

p1_poly_svm = np.poly1d(poly_svm)
p1_poly_svm_struct = np.poly1d(poly_svm_struct)




# 获得得分
bilstm_struct_score_list = []
bilstm_score_list = []

lstm_score_list = []
lstm_struct_score_list = []

svm_score_list = []
svm_struct_score_list = []




for i in range(len(bilstmmodel_16_pred_class)):
    indextmp = bilstmmodel_16_pred_class[i]
    bilstm_struct_score_list.append(bilstmmodel_16_pred[i][indextmp])
    indextmp = mybilstmmodel_16_pred_class[i]
    bilstm_score_list.append(mybilstmmodel_16_pred[i][indextmp])
    indextmp = get_my_lstm_model_16_nofull_pred_class[i]
    lstm_score_list.append(get_my_lstm_model_16_nofull_pred[i][indextmp])
    indextmp = lstm_model_1_nofull_16_pred_class[i]
    lstm_struct_score_list.append(lstm_model_1_nofull_16_pred[i][indextmp])
    indextmp = pred_y[i]
    svm_score_list.append(pred_y_proba[i][indextmp])
    indextmp = pred_y_struct[i] 
    svm_struct_score_list.append(pred_y_proba_struct[i][indextmp])


# 计算置信度 # 也可以使用yvals=np.polyval(z1,x)

p1_poly_bilstm_struct_confidence = p1_poly_bilstm_struct(bilstm_struct_score_list) 
p1_poly_bilstm_confidence = p1_poly_bilstm(bilstm_score_list)

p1_poly_lstm_confidence = p1_poly_lstm(lstm_score_list) 
p1_poly_lstm_struct_confidence = p1_poly_lstm_struct(lstm_struct_score_list) 

p1_poly_svm_confidence = p1_poly_svm(svm_score_list) 
p1_poly_svm_struct_confidence = p1_poly_svm_struct(svm_struct_score_list)



C_list = [0.95, 0.80, 0.60, 0.35, -1]  # 置信度阈值

#   4 置信度组合 加入集成学习
def result_vote(C_list, confidence_list, confidence_class_list):
    decision_list = []
    decision_dic = {}   #  字典形式  { 类型 ：[计数, 该置信度相加的和] }
    for c in C_list:
        for i in range(len(confidence_list)):
            if confidence_list[i] >= c:
                decision_list.append(i)      #   将置信度加入
        if len(decision_list)>2:
            for class_index in decision_list:
                if confidence_class_list[class_index] in decision_dic:
                    decision_dic[confidence_class_list[class_index]][0] = decision_dic[confidence_class_list[class_index]][0] +1
                    decision_dic[confidence_class_list[class_index]][1] = decision_dic[confidence_class_list[class_index]][1] + confidence_list[class_index]
                else:
                    decision_dic[confidence_class_list[class_index]] = [1,confidence_list[class_index]]
            vote_class = list(decision_dic.keys())  # vote()   # 找出投票最高的类
            vote_count = []
            for tmpclass in vote_class:
                vote_count.append(decision_dic[tmpclass][0])
            if max(vote_count) == min(vote_count):
                vote_score = []
                for tmpclass in vote_class:
                    vote_score.append(decision_dic[tmpclass][1])
                max_score_index = vote_score.index(max(vote_score))
                max_score_class = vote_class[max_score_index]
                return max_score_class
            else:
                max_vote_count_index = vote_count.index(max(vote_count))
                return vote_class[max_vote_count_index]
        elif len(decision_list)==2:
            if confidence_list[decision_list[0]]>confidence_list[decision_list[1]]:
                # print(confidence_class_list[decision_list[0]])
                return confidence_class_list[decision_list[0]]
            else:
                # print(confidence_class_list[decision_list[1]])
                return confidence_class_list[decision_list[1]]
        elif len(decision_list)==1:
            # print(confidence_class_list[decision_list[0]])
            return confidence_class_list[decision_list[0]]

# 改进lstm集成学习
vote_result = []
for i in range(len(bilstmmodel_16_pred_class)):
    tmp_confidence_list = []
    tmp_confidence_class_list = []
    tmp_confidence_list.append(p1_poly_bilstm_struct_confidence[i])
    tmp_confidence_list.append(p1_poly_bilstm_confidence[i])
    tmp_confidence_list.append(p1_poly_svm_confidence[i])
    tmp_confidence_list.append(p1_poly_svm_struct_confidence[i])
    tmp_confidence_class_list.append(bilstmmodel_16_pred_class[i])
    tmp_confidence_class_list.append(mybilstmmodel_16_pred_class[i])
    tmp_confidence_class_list.append(pred_y[i])
    tmp_confidence_class_list.append(pred_y_struct[i])
    tmp_result = result_vote(C_list, tmp_confidence_list, tmp_confidence_class_list)
    vote_result.append(tmp_result)

print(classification_report(svm_y_test, vote_result))



#单lstm集成学习

lstm_vote_result = []
for i in range(len(bilstmmodel_16_pred_class)):
    tmp_confidence_list = []
    tmp_confidence_class_list = []
    tmp_confidence_list.append(p1_poly_lstm_confidence[i])
    tmp_confidence_list.append(p1_poly_lstm_struct_confidence[i])
    tmp_confidence_list.append(p1_poly_svm_confidence[i])
    tmp_confidence_list.append(p1_poly_svm_struct_confidence[i])
    tmp_confidence_class_list.append(get_my_lstm_model_16_nofull_pred_class[i])
    tmp_confidence_class_list.append(lstm_model_1_nofull_16_pred_class[i])
    tmp_confidence_class_list.append(pred_y[i])
    tmp_confidence_class_list.append(pred_y_struct[i])
    tmp_result = result_vote(C_list, tmp_confidence_list, tmp_confidence_class_list)
    lstm_vote_result.append(tmp_result)

print(classification_report(svm_y_test, lstm_vote_result))





# ===================  对爬取的数据进行预测=====================
del mybilstmmodel_nofull_16
del get_my_lstm_model_16
del get_my_lstm_model_16_nofull


spider_webfilepath = "/home/yangc/pdnsdata2/pdnsdata/"


# 保存爬到的数据
i=0
j=0

spider_webfilecontent = {}     # 内容
spider_webfilestruct = {}      # 结构

fs = os.listdir(spider_webfilepath)

#读取所有爬到的网站内容 存到map中
for filename in fs:
    i = i +1
    tmpfilename = filename.replace(".txt","")
    webdatapath = os.path.join(spider_webfilepath, filename)
    webdata = read_all_data(webdatapath)
    webstructdata = getpage_struct(webdatapath)
    if tmpfilename not in spider_webfilecontent:
        spider_webfilecontent[tmpfilename] = webdata
        spider_webfilestruct[tmpfilename] = webstructdata
    else:
        print(tmpfilename)



#  从webfilecontent中拿到对应的分类好的文件
spider_content_train_src = []      # 训练集文本列表
spider_struct_train_src = []      # 训练集结构列表
spider_opinion_train_stc = []      # 训练集类别列表
spider_filename_train_src = []     # 训练集对应的域名

                
for url in  spider_webfilecontent:    
    if len(spider_webfilecontent[url])<26:
        print(url)
    else:
        spider_content_train_src.append(spider_webfilecontent[url])               # 加入数据集 字符串
        spider_struct_train_src.append(spider_webfilestruct[url])               # 加入数据集 字符串
        spider_filename_train_src.append(url)



print("已爬取网页数：")
print(i)
print("有效网页数：")
print(len(spider_content_train_src))   #10901
print(len(spider_struct_train_src))
print(len(spider_opinion_train_stc))
print(len(spider_filename_train_src))




# 2.4 将文本转为张量
#  原始训练数据
spider_X_train = []             # 只保存文本的单词下标  去重
spider_X_train_full = []             # 只保存文本的单词下标. 不去重
spider_X_train_full_txt = []            
spider_X_train_struct = []            
spider_filename_train = []            

spider_Max_lstm_lenth = 0

# 将单词转为词向量的下标,和对应词,下标从1开始 返回下标的list ，过滤词数较少的网站

# for word in content_train_src:
for i in range(len(spider_content_train_src)):
    # tmp_words = mytool.seg_sentence(sentence,stopwordslist)
    tmplist_full, tmpword_list_full = words2index(spider_content_train_src[i])
    if len(tmplist_full)>spider_Max_lstm_lenth:
        spider_Max_lstm_lenth = len(tmplist_full)
    #
    if len(tmplist_full)> 20:
        spider_X_train_full.append(tmplist_full)
        spider_X_train_full_txt.append(tmpword_list_full)
        spider_X_train_struct.append(spider_struct_train_src[i])
        # opinion_train_stc.append(words2index(word))
        spider_filename_train.append(spider_filename_train_src[i])

len(spider_X_train_full)
len(spider_X_train_full_txt)
len(spider_X_train_struct)
len(spider_filename_train)


spider_struct_list_input = []
for i in range(len(spider_X_train_struct)):
    tmp = []
    for j in range(len(spider_X_train_struct[i])):
        for k in range(len(spider_X_train_struct[i][j])):
            tmp.append(spider_X_train_struct[i][j][k])
    spider_struct_list_input.append(tmp)

len(spider_struct_list_input)


# svm

spider_svm_x_train_txt=[]

for words in spider_X_train_full_txt:
    tmpstr = ""
    for word in words:
        tmpstr = tmpstr + " " + word
    spider_svm_x_train_txt.append(tmpstr)



spider_X_train_vec = vectorizer.transform(spider_svm_x_train_txt)
spider_svm_x_train_input = transform.transform(spider_X_train_vec)
#(9356, 56909)
spider_svm_proba  = clf.predict_proba(spider_svm_x_train_input)
spider_svm_proba_class = np.argmax(spider_svm_proba, axis=1)



# svm struct

print(shape(spider_svm_x_train_input)) 

# 文本特征tfidf 稀疏矩阵转稠密矩阵
spider_svm_x_train_input_dense = spider_svm_x_train_input.todense()
spider_svm_x_train_input_array = spider_svm_x_train_input_dense.getA()

print(shape(spider_svm_x_train_input_array)) 



spider_struct_list_train_word = []
for words in spider_struct_list_input:
    tmplist = ""
    for word in words:
        tmplist = tmplist + " " + str(word)
    spider_struct_list_train_word.append(tmplist)



spider_X_train_vec_struct = struct_vectorizer.transform(spider_struct_list_train_word)
spider_svm_x_train_struct = struct_transform.transform(spider_X_train_vec_struct)


# 结构特征tfidf矩阵转为list
spider_svm_x_train_struct_dense = spider_svm_x_train_struct.todense()

spider_svm_x_train_struct_dense_array = spider_svm_x_train_struct_dense.getA()

shape(spider_svm_x_train_input_array)
shape(spider_svm_x_train_struct_dense_array)


# 将 文本和结构 tf-idf 值 合并
spider_svm_x_train_input_struct = []
for i in range(shape(spider_svm_x_train_input_dense)[0]):
    tmpinput = []
    for tmpnum in spider_svm_x_train_input_array[i]:
        tmpinput.append(tmpnum)
    for tmpnum in spider_svm_x_train_struct_dense_array[i]:
        tmpinput.append(tmpnum)
    spider_svm_x_train_input_struct.append(tmpinput)

del spider_svm_x_train_input_dense
del spider_svm_x_train_input_array
del spider_svm_x_train_struct_dense
del spider_svm_x_train_struct_dense_array

print(shape(spider_svm_x_train_input_struct))

import scipy.sparse.csr

#转为稀疏矩阵
spider_svm_x_train_input_struct_csr = scipy.sparse.csr.csr_matrix(spider_svm_x_train_input_struct)

# del svm_x_train_input_struct

print(shape(spider_svm_x_train_input_struct_csr))




spider_svm_struct_proba  = clf_struct.predict_proba(spider_svm_x_train_input_struct_csr)
spider_svm_struct_proba_class = np.argmax(spider_svm_struct_proba, axis=1)



spider_svm_proba_savepath = "/home/yangc/svm_result/spider_svm_proba.npy"
spider_svm_struct_proba_savepath = "/home/yangc/svm_result/spider_svm_struct_proba.npy"
np.save( spider_svm_proba_savepath, spider_svm_proba)
np.save( spider_svm_struct_proba_savepath, spider_svm_struct_proba)

# lstm

shape(spider_X_train_full)
shape(spider_struct_list_input)


spider_X_train_full = pad_sequences(spider_X_train_full, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')
spider_lstm_input_struct_train = pad_sequences(spider_struct_list_input, maxlen=model_max_len+model_struct_max_len, padding='pre', truncating='post')


# 双向双特征
spider_bilstmmodel_16_pred = bilstmmodel_16.predict([spider_X_train_full, spider_lstm_input_struct_train], batch_size=None, verbose=1, steps=None)
# 双向单特征
spider_mybilstmmodel_16_pred = mybilstmmodel_16.predict(spider_X_train_full, batch_size=None, verbose=1, steps=None)




spider_bilstmmodel_16_pred_savepath = "/home/yangc/svm_result/spider_bilstmmodel_16_pred.npy"
spider_mybilstmmodel_16_pred_savepath = "/home/yangc/svm_result/spider_mybilstmmodel_16_pred.npy"
np.save( spider_bilstmmodel_16_pred_savepath, spider_bilstmmodel_16_pred)
np.save( spider_mybilstmmodel_16_pred_savepath, spider_mybilstmmodel_16_pred)


# pred_y_proba_savepath = "/home/yangc/svm_result/pred_y_proba.npy"
# pred_y_proba_struct_savepath = "/home/yangc/svm_result/pred_y_proba_struct.npy"
# spider_svm_struct_proba=np.load(pred_y_proba_struct_savepath)
# spider_svm_proba=np.load(pred_y_proba_savepath)


# 双向双特征
spider_bilstmmodel_16_pred_class = np.argmax(spider_bilstmmodel_16_pred, axis=1)
# 双向单特征
spider_mybilstmmodel_16_pred_class = np.argmax(spider_mybilstmmodel_16_pred, axis=1)


# 双向双特征
spider_bilstmmodel_16_pred_score_list = [] 
# 双向单特征
spider_mybilstmmodel_16_pred_score_list = []
spider_svm_struct_proba_score_list = []
spider_svm_proba_score_list = []
for i in range(len(spider_bilstmmodel_16_pred_class)):
    indextmp = spider_bilstmmodel_16_pred_class[i]
    spider_bilstmmodel_16_pred_score_list.append(spider_bilstmmodel_16_pred[i][indextmp])
    indextmp = spider_mybilstmmodel_16_pred_class[i]
    spider_mybilstmmodel_16_pred_score_list.append(spider_mybilstmmodel_16_pred[i][indextmp])
    indextmp = spider_svm_struct_proba_class[i]
    spider_svm_struct_proba_score_list.append(spider_svm_struct_proba[i][indextmp])
    indextmp = spider_svm_proba_class[i]
    spider_svm_proba_score_list.append(spider_svm_proba[i][indextmp])


# 计算置信度 # 也可以使用yvals=np.polyval(z1,x)

spider_p1_poly_bilstm_struct_confidence = p1_poly_bilstm_struct(spider_bilstmmodel_16_pred_score_list) 
spider_p1_poly_bilstm_confidence = p1_poly_bilstm(spider_mybilstmmodel_16_pred_score_list)


spider_p1_poly_svm_confidence = p1_poly_svm(spider_svm_proba_score_list) 
spider_p1_poly_svm_struct_confidence = p1_poly_svm_struct(spider_svm_struct_proba_score_list)


# 改进lstm集成学习
spider_vote_result = []
for i in range(len(spider_p1_poly_bilstm_struct_confidence)):
    tmp_confidence_list = []
    tmp_confidence_class_list = []
    tmp_confidence_list.append(spider_p1_poly_bilstm_struct_confidence[i])
    tmp_confidence_list.append(spider_p1_poly_bilstm_confidence[i])
    tmp_confidence_list.append(spider_p1_poly_svm_confidence[i])
    tmp_confidence_list.append(spider_p1_poly_svm_struct_confidence[i])
    tmp_confidence_class_list.append(spider_bilstmmodel_16_pred_class[i])
    tmp_confidence_class_list.append(spider_mybilstmmodel_16_pred_class[i])
    tmp_confidence_class_list.append(spider_svm_proba_class[i])
    tmp_confidence_class_list.append(spider_svm_struct_proba_class[i])
    tmp_result = result_vote(C_list, tmp_confidence_list, tmp_confidence_class_list)
    spider_vote_result.append(tmp_result)


spider_filename_train=[]


# for word in content_train_src:
for i in range(len(spider_content_train_src)):
    # tmp_words = mytool.seg_sentence(sentence,stopwordslist)
    tmplist_full, tmpword_list_full = words2index(spider_content_train_src[i])
    if len(tmplist_full)> 20:
        # opinion_train_stc.append(words2index(word))
        spider_filename_train.append(spider_filename_train_src[i])

len(spider_vote_result)
print(len(spider_filename_train))



save_result_dic = {}
for i in range(len(spider_vote_result)):
    save_result_dic[spider_filename_train[i]] = spider_vote_result[i]

save_result_dic['www.aibanji.com']
save_result_dic['com.browurl.com']

f6 = open("/home/yangc/svm_result/spider_result_dic",'w')
f6.write(str(save_result_dic))
f6.close()


f = open('/home/yangc/svm_result/spider_result_dic','r')
a = f.read()
dict_hi = eval(a)
f.close()

spider_webfilecontent['omomgy.xyz']
spider_webfilecontent['www.lxsw.xyz']

save_result_dic['omomgy.xyz']

spider_vote_result
spider_filename_train_src

test = set(spider_vote_result)


from collections import Counter


test=Counter(spider_vote_result)



for key in save_result_dic:
    if save_result_dic[key]==23:
        print(key)


spider_webfilecontent['oceanvstheworld.knoji.com']


























# 作图
# win作图
import matplotlib.pyplot as plt
import numpy as np

list1 = []
list2 = []
list3 = []
list4 = []

list1 = list1[0:51]
list2 = list2[0:51]
# list3 = list3[0:51]
# list4 = list4[0:51]

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('f1-score对比')
plt.xlabel("类型序号")
plt.ylabel("f1-score")
x=np.arange(0,len(list1))
x[0]=0
my_x_ticks = np.arange(0, 51, 1)
plt.xticks(my_x_ticks)
plt.plot(x,list1,label='方法1',color='g',linewidth=2,linestyle='--')#添加linestyle设置线条类型
plt.plot(x,list2,label='方法2',color='b',linewidth=2,linestyle='-') #添加linestyle设置线条类型
# plt.plot(x,list3,label='3',color='r',linewidth=2,linestyle='-.')#添加linestyle设置线条类型
# plt.plot(x,list4,label='4',color='k',linewidth=2,linestyle=':')#添加linestyle设置线条类型
# plt.plot(x,list2,label='list2',color='b',linewidth=5,linestyle='--')
plt.legend()
plt.grid()#添加网格
plt.show()




#单特征双向lstm

y1 = [0.8053642873514599, 0.7297426908230563, 0.742268058875591]
yy1 = [0.8001975697833917, 0.7806935332708529,  0.7758857130314184]

y2 = [0.8210981669859513, 0.7541939463069859, 0.7652271904257378 ]
yy2 = [0.8301419311024624, 0.8181818181818182, 0.8133636236708899]

y3 = [0.7994694874253448, 0.7539550884956746,0.7668486981964514 ]
yy3 = [0.8141938396450904, 0.8059981255857545, 0.8019807621703887]

y4 = [0.8454708432821596, 0.8046104292808364, 0.811962673224768 ]
yy4 = [0.8615768110307197, 0.8491096532333646, 0.8481855061160365]

name = ['precision', 'recall', 'f1-score']
x = np.arange(3)
total_width, n = 0.4, 4     # 有多少个类型，只需更改n即可
width = 0.2

plt.bar(x, y1,  width=width, label='label1',color='red')
plt.bar(x + width, y2, width=width, label='label2',color='deepskyblue',tick_label=name)
plt.bar(x + 2 * width, y3, width=width, label='label3', color='green')
plt.bar(x + 3 * width, y4, width=width, label='label4', color='k')

plt.xticks()
plt.legend()  # 防止label和图像重合显示不出来
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.ylabel('value')
plt.xlabel('line')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率
# plt.rcParams['figure.figsize'] = (15.0, 8.0)  # 尺寸
plt.title("title")
plt.show()




list3 = [0.7297426908230563, ]  #    macro avg  recall
list4 = [0.7806935332708529, ]          #weighted avg   recall



# 单特征LSTM
def get_my_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False))
    model.add(LSTM(200, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(class_index), activation='softmax'))
    model.build((None,model_max_len, EMBEDDING_DIM))
    model.summary()
    # tf.config.experimental_run_functions_eagerly(True)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

#双特征双向LSTM+ attention
def get_bilstm_model(content_attention_size, struct_attention_size):
    input_content  = Input(shape=(model_struct_max_len + model_max_len, ), dtype='int32',name = "content")
    input_struct= Input(shape=(model_struct_max_len + model_max_len,), dtype='int32', name = "struct")
    content_embedding = Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(input_content) #不能训练
    content_embedding = Bidirectional(LSTM(EMBEDDING_DIM, dropout=0.5, return_sequences=True))(content_embedding)
    content_embedding = AttentionLayer(attention_size=content_attention_size)(content_embedding)
    # content_embedding = LSTM(EMBEDDING_DIM, dropout=0.5)(content_embedding)
    content_LSTM = keras.Model(inputs=input_content, outputs=content_embedding)
    struct_embedding = Embedding(input_dim = TAG_length + 1,
                            output_dim =STRUCT_EMBEDDING_DIM,
                            mask_zero = True,
                            trainable=True)(input_struct) #可以训练
    # struct_embedding = LSTM(STRUCT_EMBEDDING_DIM, dropout=0.5)(struct_embedding)
    struct_embedding = Bidirectional(LSTM(STRUCT_EMBEDDING_DIM, dropout=0.5, return_sequences=True))(struct_embedding)
    struct_embedding = AttentionLayer(attention_size=struct_attention_size)(struct_embedding)
    struct_LSTM = keras.Model(inputs=input_struct, outputs=struct_embedding)
    concate =  concatenate([content_LSTM.output,struct_LSTM.output], axis=-1)
    # z = LSTM(STRUCT_EMBEDDING_DIM+EMBEDDING_DIM, dropout=0.5)(concate)
    z = Dense(128, activation="relu")(concate)
    z = Dense(len(class_index), activation='softmax')(z)
    model = keras.Model(inputs=[content_LSTM.input, struct_LSTM.input], outputs=z)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# old
tmp_y_train = []
tmp_y_test_ = []


for i in y_train_index:
    tmp_y_train.append(opinion_train_stc[i])


for i in y_test_index:
    tmp_y_test_.append(opinion_train_stc[i])

y_train=to_categorical(tmp_y_train, len(class_index))       #将标签处理成输入的格式
y_test=to_categorical(tmp_y_test_, len(class_index))


x_train_raw = pad_sequences(x_train, maxlen=model_max_len)
x_test_raw = pad_sequences(x_test, maxlen=model_max_len)



tf.compat.v1.disable_eager_execution()  # 避免未知错误
class AttentionLayer(Layer):
    def __init__(self, attention_size=None, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config
        
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size
        
        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                initializer='uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                initializer='uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        self.V = K.reshape(self.V, (-1, 1))
        H = K.tanh(K.dot(inputs, self.W) + self.b)
        score = K.softmax(K.dot(H, self.V), axis=1)
        outputs = K.sum(score * inputs, axis=1)
        return outputs
    
    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]



def create_classify_model(hidden_size, attention_size):
	# 输入层
    inputs = Input(shape=(model_max_len,), dtype='int32')
    # Embedding层
    x = Embedding(input_dim = EMBEDDING_length  + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=700,
                            mask_zero = True,
                            trainable=False)(inputs)
    # BiLSTM层
    x = Bidirectional(LSTM(hidden_size, dropout=0.2, return_sequences=True))(x)
    # x = Bidirectional()(x)
    # Attention层
    x = AttentionLayer(attention_size=attention_size)(x)
    # 输出层
    outputs = Dense(len(class_index), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.summary() # 输出模型结构和参数数量
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# 3.3 训练
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=16, epochs=10, validation_split=0.04)

model = create_classify_model(150,150)
model_train = model_fit(model, x_train_raw, y_train)



print(model.evaluate(x_test_raw, y_test))






# 1 取词向量  embedding_matrix
# X_train_full_train_embedding = []
# for i in range(len(X_train_full_train)):
#     tmp = []
#     for j in range(len(X_train_full_train[i])):
#         tmp.append(embedding_matrix[X_train_full_train[i][j]])
#     X_train_full_train_embedding.append(tmp)


# print(len(X_train_full_train_embedding[0][0]))
# print(len(X_train_full_train))
# print(len(X_train_full_train_embedding))


# X_train_full_train_embedding = np.array(X_train_full_train_embedding)
# print(len(X_train_full_train_embedding))
# tmplen = len(X_train_full_train_embedding)
# X_train_full_train_embedding = X_train_full_train_embedding.reshape(tmplen, -1)
# len(X_train_full_train_embedding[0])
# len(X_train_full_train_embedding)



# X_train_full_test_embedding = []
# for i in range(len(X_train_full_test)):
#     tmp = []
#     for j in range(len(X_train_full_test[i])):
#         tmp.append(embedding_matrix[X_train_full_test[i][j]])
#     X_train_full_test_embedding.append(tmp)

# X_train_full_test_embedding = np.array(X_train_full_test_embedding)
# print(len(X_train_full_test_embedding))
# tmplen = len(X_train_full_test_embedding)
# X_train_full_test_embedding = X_train_full_test_embedding.reshape(tmplen, -1)
# # 2 合并特征 svm不用处理0的情况， lstm需要
# # 文本和结构特征依次加入
# svm_input_train = []
# for i in range(len(X_train_full_train_embedding)):
#     tmp = []
#     for j in range(len(X_train_full_train_embedding[i])):
#         tmp.append(X_train_full_train_embedding[i][j])
#     for j in range(len(struct_list_train[i])):
#         tmp.append(struct_list_train[i][j])
#     svm_input_train.append(tmp)

# print(len(svm_input_train[0]))

# svm_input_test = []
# for i in range(len(X_train_full_test_embedding)):
#     tmp = []
#     for j in range(len(X_train_full_test_embedding[i])):
#         tmp.append(X_train_full_test_embedding[i][j])
#     for j in range(len(struct_list_test[i])):
#         tmp.append(struct_list_test[i][j])
#     svm_input_test.append(tmp)



# # 开始训练
# # svm
# svm_model = SVC(probability = True)
# svm_model.fit(svm_input_train, svm_y_train)

# # print(x_test[0])
# # print(shape(x_test))
# pred_y  = svm_model.predict(svm_input_test)
# # pred_y_proba  = svm_model.predict_proba(x_test)
# print(classification_report(svm_y_test,pred_y))


#svm使用 TF-IDF ----_______-_________________
# svm_Y_label_train=to_categorical(svm_y_train, len(class_index))
# svm_Y_label_test=to_categorical(svm_y_test, len(class_index))