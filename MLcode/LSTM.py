# -*- coding: utf-8 -*-

import sys
from gensim.models import Word2Vec
# from LoadData import loadData
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential
# from keras.layers import Dropout,Dense,Embedding,LSTM,Activation
from keras.layers import Dense, LSTM, Embedding, Dropout, Conv1D, MaxPooling1D, Bidirectional, Activation,Masking
import pickle
from sklearn.model_selection import train_test_split
from gensim.corpora.dictionary import Dictionary

import re
from gensim import corpora,models,similarities
import sys
import io
import jieba
import codecs
from gensim import corpora
from gensim.models import LdaModel
# from gensim import models
# from gensim.corpora import Dictionary
import json
import logging




# 步骤1 读取预训练词向量  
# embeddings_index 为字典  单词 ：下标
# embedding_matrix 词向量数组 

EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330
MAX_SEQUENCE_LENGTH = 10

filepath = 'Tencent_AILab_ChineseEmbedding.txt'
f = open(filepath)
embeddings_index = {}
embedding_matrix = np.zeros((EMBEDDING_length + 1, EMBEDDING_DIM))
i = 1
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = i
    embedding_matrix[i] = coefs
    i = i+1
print('Found %s word vectors.' % i)
f.close()



#步骤2 数据预处理

# 2.1 读取停用词
# stopwordslist 保存所有停用词

stopwordslist = []  # 停用词列表
def read_stopwords(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
stopwordslist =read_stopwords("C:/Users/shinelon/Desktop/linuxfirefox/stopwords-master/stopwords-master/cn_stopwords.txt")


#2.2 设置分类类别
# classtype 保存了所有的分类信息  子类名 ： 父类目
# class_index 保存了父类名对应的下标

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
            classtype[parts[0]]=parts[2].strip('\n')


filepath = "D:/dnswork/sharevm/top.chinaz.txt"
initclass(filepath)


#2.3 读取爬取的网页信息
# 数据存入 X_train_text 网页中所有语句合成一句
# 标签下标存入 Y_train_text
X_train_text = []
Y_train = []
jieba.setLogLevel(logging.INFO)
#读取保存的网页信息
def read_webdata(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        return json.loads(file_to_read.read())
        # print(webdata)

# 读取网页数据,过滤非汉字
def get_all_webdata(webdata):
    tmp_data = ""
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    for data in webdata['title']:
        tmp = rule.sub('',data)
        tmp_data=tmp_data + tmp
    for data in webdata['description']:
        tmp = rule.sub('',data)
        tmp_data=tmp_data + tmp
    for data in webdata['keywords']:
        tmp = rule.sub('',data)
        tmp_data=tmp_data + tmp
    for data in webdata['webtext']:
        tmp = rule.sub('',data)
        tmp_data=tmp_data + tmp
    for data in webdata['abouttext']:
        tmp = rule.sub('',data)
        tmp_data=tmp_data + tmp
    return tmp_data

import os
path = "D:/dnswork/sharevm/topchinaz/"
fs = os.listdir(path)
i=0
j=0
for subpath in fs:
    filepath = os.path.join(path, subpath)
    # print(filepath)
    if (os.path.isdir(filepath)):
        webdata_classtype = classtype[subpath]  # 查询父类别
        webdata_class_index = class_index[webdata_classtype] #父类别下标
        webdata_path = os.listdir(filepath)
        for filename in webdata_path:
            i=i+1
            webdata = read_webdata(os.path.join(filepath, filename))
            if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
                if len(webdata['webtext'])>=15:
                    j=j+1
                    X_train_text.append(get_all_webdata(webdata))
                    Y_train.append(webdata_class_index)

print(i)
print(j)


# 2.4 将文本转为张量
# X_train 训练数据
X_train = []

# 分割句子变成词语 bool_cut_all 是否全模式
def seg_sentence(sentence , bool_cut_all=False):
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    sentence = rule.sub('',sentence)
    sentence_seged = jieba.lcut(sentence.strip(),cut_all=bool_cut_all)
    wordlist = []
    for word in sentence_seged:
        if word not in stopwordslist:
            if word != '\t':
                wordlist.append(word)
    return wordlist

# 将单词转为词向量的下标,下标从1开始 返回下标的list
def words2index(words):
    index_list = []
    for word in words:
        if word in embeddings_index.keys():  # 单词是否在词向量中
            index_list.append(embeddings_index[word])
    return index_list

for sentence in X_train_text:
    tmp_words = seg_sentence(sentence)
    X_train.append(words2index(tmp_words))








# 3 机器学习训练

# def get_lstm_model(max_features, embed_size):
def get_lstm_model():
    model = Sequential()
    # model.add(Masking(mask_value= [-1. -1.      -1.   -1.  ],input_shape=(3,4)))
    model.add(Embedding(EMBEDDING_length + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=MAX_SEQUENCE_LENGTH,
                            mask_zero = True,
                            trainable=False))
    model.add(LSTM(3, recurrent_dropout=0.1))
    model.add(Dropout(0.25))
    model.add(Dense(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

MAX_NB_WORDS = 50000
# 每条cut_review最大的长度
MAX_SEQUENCE_LENGTH = 250
# 设置Embeddingceng层的维度
EMBEDDING_DIM = 100

data1 = np.random.random(size=( 5,3, 4)) # batch_size = 1, timespan = 100
print(data1)
data1[1,2]=-1
print(data1)
y_train = np.random.random(size=(5))
print(y_train)
# model = get_lstm_model(max_features, embed_size)
model = get_lstm_model()
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=1, epochs=2, validation_split=0.2)
model_train = model_fit(model, data1, y_train)
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
# tokenizer.fit_on_texts(df['cut_review'].values)

