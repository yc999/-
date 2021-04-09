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
import codecs
from gensim import corpora
# from gensim import models
# from gensim.corpora import Dictionary
import json
import logging
import os
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *
import numpy as np
# print(os.path.realpath('./MLcode/'))
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
import random
import meanShift as ms
import mytool

# import mean_shift as ms
import matplotlib.pyplot as plt

















# 步骤1 加载词向量  
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




#步骤2 数据预处理

# 2.1 读取停用词
# stopwordslist 保存所有停用词

stopwordslist = []  # 停用词列表
stopwords_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
stopwordslist =mytool.read_stopwords(stopwords_path)


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


# filepath = "D:/dnswork/sharevm/top.chinaz.txt"
initfilepath = "/home/jiangy2/dnswork/top.chinaz.txt"
initclass(initfilepath)



#2.3 读取爬取的网页信息
# 数据存入 X_train_text 网页中所有语句合成一句
# 标签下标存入 Y_train
X_train_text = []
Y_train = []

#读取保存的网页信息
# path = "D:/dnswork/sharevm/topchinaz/"
path = "/home/jiangy2/webdata/"
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
            webdata = mytool.read_webdata(os.path.join(filepath, filename))
            if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
                if len(webdata['webtext'])>=15:
                    j=j+1
                    X_train_text.append(mytool.get_all_webdata(webdata))
                    Y_train.append(webdata_class_index)

print("已爬取网页数：")
print(i)
print("有效网页数：")
print(j)







# 2.4 将文本转为张量
# X_train 训练数据
X_train = []

# 将单词转为词向量的下标,下标从1开始 返回下标的list
def words2index(words):
    index_list = []
    for word in words:
        if word in embeddings_index.keys():  # 单词是否在词向量中
            index_list.append(embeddings_index[word])
    return index_list

for sentence in X_train_text:
    tmp_words = mytool.seg_sentence(sentence,stopwordslist)
    X_train.append(words2index(tmp_words))











# 3 机器学习训练
model_max_len = 300


# 3.1 定义模型
def get_lstm_model():
    model = Sequential()
    model.add(Embedding(input_dim = EMBEDDING_length + 1,
                            output_dim =EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            # input_length=200,
                            mask_zero = True,
                            trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(class_index), activation='softmax'))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    return model

model = get_lstm_model()
model_cluster = get_lstm_model()




# 3.2 划分数据集
# 3.2.1 划分测试训练集
# X_padded=pad_sequences(X_train, maxlen=300)
# Y=to_categorical(Y_train, len(class_index))
# x_train, x_test, y_train, y_test = train_test_split(X_padded, Y, test_size=0.2)

Y=to_categorical(Y_train, len(class_index))
x_train, x_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.2)

x_train_raw = pad_sequences(x_train, maxlen=model_max_len)
x_test_raw = pad_sequences(x_test, maxlen=model_max_len)



# 3.2.2 处理聚类后的数据

#在每个类中选取等比例个词 
# 某一类取词数 = model_max_len * 该类词数/总词数 （最多不超过该类词数）
# 比如maxlen = 10 总共有100个词 10个类 每个类都是10个词 那么每个类中取1个词 //优化：离中心点近的权重越大，被选择概率越大
# 返回下标
def get_mean_shift_result(mean_shift_result):
    cluster_assignments = mean_shift_result.cluster_ids
    sum_point = len(cluster_assignments) # 总词数
    # 词数小于最大长度 直接返回
    if sum_point <= model_max_len:
        raise Exception("get_mean_shift_result len error")
    # 将各类词的下标存入字典中 cluster_dic{类别：[下标]} 
    cluster_dic = {}
    for index, cluster in enumerate(cluster_assignments):
        if cluster  in cluster_dic:
            cluster_dic[cluster].append(index)
        else:
            cluster_dic[cluster]=[index]
    result_point = []
    # 将每一类词下标取出放入list中
    for key in cluster_dic:
        cluster_count = len(cluster_dic[key]) # 该类词数
        choose_count = model_max_len * cluster_count / sum_point
        cluster_choose_result = random.sample(cluster_dic[key], int(choose_count))
        result_point = result_point + cluster_choose_result
    # 不足maxlen 随机添加未被选中的下标
    len_result_point = len(result_point)
    while len_result_point < model_max_len:
        random_number = random.randint(0,sum_point-1)
        if random_number not in result_point:
            result_point.append(random_number)
            len_result_point = len_result_point + 1
    return result_point


mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')
kernel_bandwidth = [10]*EMBEDDING_DIM     # 带宽参数

# 返回聚类结果
def get_cluster_result(cluster_data, kernel_bandwidth):
    result = []
    for data_index in cluster_data:
        data = []
        if len(data_index) > model_max_len:
            # 将下标转为数据点坐标值，然后聚类
            for index in data_index:
                data.append(embedding_matrix[index])
            mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = kernel_bandwidth)
            index_result = get_mean_shift_result(mean_shift_result)
            data = []
            # 将聚类后的结果转为对应下标
            for index in index_result:
                data.append(data_index[index])
            result.append(data)
        else:   #词数不够 model_max_len 直接返回
            result.append(data_index)
    return result

#训练集聚类 x_train_cluster
x_train_cluster = get_cluster_result(x_train, kernel_bandwidth)
#测试集聚类 x_test_cluster
x_test_cluster = get_cluster_result(x_test, kernel_bandwidth)

x_train_cluster = pad_sequences(x_train_cluster, maxlen=model_max_len)
x_test_cluster = pad_sequences(x_test_cluster, maxlen=model_max_len)




# 3.3 训练
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=10, epochs=5, validation_split=0.1)
model_train = model_fit(model, x_train_raw, y_train)
cluster_model_train = model_fit(model, x_train_cluster, y_train)


# 3.4 测试
print(model.evaluate(x_test_raw, y_test))
print(model.evaluate(x_test_cluster, y_test))

