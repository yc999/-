# -*- coding: utf-8 -*-
# 把部分已确定的网站进行训练，对未确定的网站进行分类
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
import multiprocessing as mp
# import mean_shift as ms
# import matplotlib.pyplot as plt
os.environ['OPENBLAS_NUM_THREADS'] = '1'




# data = np.genfromtxt('D:\GitHubcode\-\MLcode\data.csv', delimiter=',')


# mean_shifter = ms.MeanShift()
# mean_shift_result = mean_shifter.cluster(data, kernel_bandwidth = 1)






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
stopwordslist = mytool.read_stopwords(stopwords_path)


#2.2 设置分类类别
readclasspath = "/home/jiangy2/dnswork/topchinaz-confirm/"
# readpath = "D:/dnswork/sharevm/topchinaz/"
# readpath = "E:/webdata/"
fs = os.listdir(readclasspath)   #读取url目录
class_index={}
weburllist = [] #保存要训练的网站
classlist = []  #保存要训练网站对应的标签
for i,filename in enumerate(fs):
    filepath = readclasspath + filename
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
            parts = line.split(",")
            if  not line:
                break
            weburllist.append(parts[1])
            classlist.append(parts[0])
    tmp = filename.split(".")[0]
    class_index[tmp] = i

print(class_index)

# classtype 保存了所有的分类信息  子类名 ： 父类目
# class_index 保存了父类名对应的下标








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







# 3.2 划分数据集
# 3.2.1 划分测试训练集
# X_padded=pad_sequences(X_train, maxlen=300)
# Y=to_categorical(Y_train, len(class_index))
# x_train, x_test, y_train, y_test = train_test_split(X_padded, Y, test_size=0.2)

Y=to_categorical(Y_train, len(class_index))
x_train, x_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.1)

x_train_raw = pad_sequences(x_train, maxlen=model_max_len)
x_test_raw = pad_sequences(x_test, maxlen=model_max_len)





# 3.3 训练
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=10, epochs=5, validation_split=0.1)

model = get_lstm_model()
model_train = model_fit(model, x_train_raw, y_train)


# 3.4 测试
print(model.evaluate(x_test_raw, y_test))

modelsave_path = "/home/jiangy2/dnswork/modeldir/LSTMmodel"
model.save(modelsave_path)










