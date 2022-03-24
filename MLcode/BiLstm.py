# -*- coding: utf-8 -*-
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
import io
import tensorflow as tf
from keras import backend as K #转换为张量
import codecs
from gensim import corpora
# from gensim import models
# from gensim.corpora import Dictionary
import json
import logging
import os
# from keras.engine.topology import Layer
from keras.layers import Layer
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *

from MLcode.classcount import X_train_text
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
# print(os.path.realpath('./MLcode/'))
import random
import meanShift as ms
import mytool
import multiprocessing as mp
# import mean_shift as ms
# import matplotlib.pyplot as plt
import jieba
import re
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


# data = np.genfromtxt('D:\GitHubcode\-\MLcode\data.csv', delimiter=',')


# mean_shifter = ms.MeanShift()
# mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = 1)




# 步骤1 加载词向量  
# embeddings_index 为字典  单词 ：下标
# embedding_matrix 词向量数组 

# MAX_SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330


# word2vec_path = '/home/yc/dnswork/glove/Tencent_AILab_ChineseEmbedding.txt'
# stopwords_path = "/home/yc/dnswork/stopwords/cn_stopwords.txt"
# webfilepath = "/home/yc/dnswork/httpwebdata/"
# file_dir = "/home/yc/dnswork/myclass/myclass/"
# modelsave_path = "/home/yc/dnswork/modeldir/LSTMmodel"

# if len(sys.argv)>1:
#     word2vec_path = '/public/ycdswork/dnswork/glove/Tencent_AILab_ChineseEmbedding.txt'
#     stopwords_path = "/public/ycdswork/dnswork/stopwords/cn_stopwords.txt"
#     webfilepath = "/public/ycdswork/dnswork/httpwebdata/"
#     file_dir = "/home/yangc/myclass/"
#     modelsave_path = "/public/ycdswork/modeldir/LSTMmodel"

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


# download.10jqka.com.cn 1Gb

#步骤2 数据预处理
print("step2")


# stopwords_path = "/public/ycdswork/dnswork/stopwords/cn_stopwords.txt"
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


# 返回将字符串切成词语list返回，并去除空格等符号
    # word_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
    # word_str = re.sub(r'\s+', ' ', word_str)  # trans 多空格 to空格
    # word_str = re.sub(r'\n+', ' ', word_str)  # trans 换行 to空格
    # word_str = re.sub(r'\t+', ' ', word_str)  # trans Tab to空格
    # word_str = re.sub(u"[^\u4E00-\u9FA5]"," ", word_str)
def Word_cut_list(word_str):
    #利用正则表达式去掉一些一些标点符号之类的符号。
    word_str = re.sub("[\s+\.\!\[\]\/_,\>\<\-$%^*(+\"\']+|[+——；:！·，”。【】《》～д£ȫ²αμв»©½йÿλ，。：“？、~@#￥%……&*（）0123456789①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
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
    # print("0")
    # print(webkey)
    if len(webkey)>0:
        for htmldata in webkey: #遍历每个页面，切词处理
            # print("1")
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


#2.3 读取爬取的网页信息
#读取保存的网页信息


#test
# testpath = "/public/ycdswork/dnswork/httpwebdata/酒店宾馆/szqinglv.cn.txt"

# fs = os.listdir(webfilepath)
# filepath = os.path.join(webfilepath, fs[0])
# webdata_path = os.listdir(filepath)
# webdatapath = os.path.join(filepath, webdata_path[0])
# print(webdatapath)
# webdata = read_all_data(webdatapath)
# print(webdata)


# 保存爬到的数据
i=0
j=0

webfilecontent = {} 
fs = os.listdir(webfilepath)

#读取所有爬到的网站内容 存到map中
for subpath in fs:
    # print(subpath)
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
            # print(webdata)
            if tmpfilename not in webfilecontent:
                webfilecontent[tmpfilename] = webdata
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

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf8')

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
        # temp = str(tmp, encoding='utf-8')
        # out = temp.encode('utf-8').decode('unicode_escape')
        tmp = unicode_to_cn(tmp)
        print(tmp)
        class_index[tmp] = indexcount
        indexcount = indexcount + 1

print(class_index)



#从webfilecontent中拿到对应的分类好的文件
content_train_src = []      # 训练集文本列表
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
                            opinion_train_stc.append(class_index[tmp])   # 加入标签集
                            filename_train_src.append(line)
                    elif 'www.'+ url  in webfilecontent:
                        if len(webfilecontent['www.'+url])<26:
                            print(url)
                        else:
                            content_train_src.append(webfilecontent['www.'+ url])               # 加入数据集 字符串
                            opinion_train_stc.append(class_index[tmp])   # 加入标签集
                            filename_train_src.append(line)
                    else:
                        print(url, line)



# for word in webfilecontent['www.dwjq.com.cn']:
#     print(word)
#     if word in embeddings_index.keys():  # test if word in embeddings_index
#         print("true")


print("已爬取网页数：")
print(i)


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
X_train = []
X_train_txt = []
Y_train_index = []        # 只保存下标
Y_filename = []
Max_lstm_lenth = 0

# 将单词转为词向量的下标,和对应词,下标从1开始 返回下标的list 可以去重



short_list = []
# for word in content_train_src:
for i in range(len(content_train_src)):
    # tmp_words = mytool.seg_sentence(sentence,stopwordslist)
    tmplist, tmpword_list = words2index(content_train_src[i])
    tmplist = list(set(tmplist))  # delet recur word
    tmpword_list = list(set(tmpword_list))  # delet recur word
    if len(tmplist)>Max_lstm_lenth:
        Max_lstm_lenth = len(tmplist)
    if len(tmplist) >25:
        X_train.append(tmplist)
        X_train_txt.append(tmpword_list)
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

X_train_txt[0]
Y_train_index[0]
# opinion_train_stc = Y_train


print("有效网页数：")
print(len(X_train))



lstm_lenth_list = []
for word in X_train:
    lstm_lenth_list.append(len(word))


len(lstm_lenth_list)

lenth_count = 0
for i in lstm_lenth_list:
    if i>1500:
        lenth_count = lenth_count + 1

# lenth_count
# del embeddings_index


# 3 机器学习训练
model_max_len = 1500



# 数据和对应类别的下标一起打乱
# Y=to_categorical(opinion_train_stc, len(class_index))
# x_train, x_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.1)
x_train, x_test, y_train_index, y_test_index = train_test_split(X_train, Y_train_index, test_size=0.1)

tmp_y_train = []
tmp_y_test_ = []


for i in y_train_index:
    tmp_y_train.append(opinion_train_stc[i])


for i in y_test_index:
    tmp_y_test_.append(opinion_train_stc[i])

y_train=to_categorical(tmp_y_train, len(class_index))
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


# 3.3 训练
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=16, epochs=10, validation_split=0.04)

bilstmmodel = create_classify_model(150,150)
bilstmmodel_train = model_fit(bilstmmodel, x_train_raw, y_train)



print(bilstmmodel.evaluate(x_test_raw, y_test))



