import os
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
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
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import *
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
import sys
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
for counter, key in enumerate(tc_wv_model.key_to_index):
    # print(counter,key)
    embeddings_index[key] = counter+1
    coefs = np.asarray(tc_wv_model[key], dtype='float32')
    embedding_matrix[counter+1] = coefs
    

del tc_wv_model


# download.10jqka.com.cn 1Gb

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


# 返回将字符串切成词语list返回，并去除空格等符号
    # word_str = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——；！，”。《》，。：“？、~@#￥%……&*（）1234567①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
    # word_str = re.sub(r'\s+', ' ', word_str)  # trans 多空格 to空格
    # word_str = re.sub(r'\n+', ' ', word_str)  # trans 换行 to空格
    # word_str = re.sub(r'\t+', ' ', word_str)  # trans Tab to空格
    # word_str = re.sub(u"[^\u4E00-\u9FA5]"," ", word_str)
def Word_cut_list(word_str):
    #利用正则表达式去掉一些一些标点符号之类的符号。
    word_str = re.sub("[\s+\.\!\[\]\/_,\>\<\-$%^*(+\"\']+|[+——；:！·，”。【】《》，。：“？、~@#￥%……&*（）0123456789①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
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
                        if len(webfilecontent[url])<30:
                            print(url)
                        else:
                            content_train_src.append(webfilecontent[url])               # 加入数据集 字符串
                            opinion_train_stc.append(class_index[tmp])   # 加入标签集
                    elif 'www.'+ url  in webfilecontent:
                        if len(webfilecontent['www.'+url])<30:
                            print(url)
                        else:
                            content_train_src.append(webfilecontent['www.'+ url])               # 加入数据集 字符串
                            opinion_train_stc.append(class_index[tmp])   # 加入标签集
                    else:
                        print(url, line)



print("已爬取网页数：")
print(i)
print("有效网页数：")
print(len(content_train_src))







# 2.4 将文本转为张量
#  原始训练数据
X_train = []
Max_lstm_lenth = 0
# 将单词转为词向量的下标,下标从1开始 返回下标的list 可以去重
def words2index(words):
    global Max_lstm_lenth
    index_list = []
    for word in words:
        if word in embeddings_index.keys():  # test if word in embeddings_index
            index_list.append(embeddings_index[word])
    index_list = list(set(index_list))  # delet recur word
    if len(index_list)>Max_lstm_lenth:
        Max_lstm_lenth = len(index_list)
    return index_list


for word in content_train_src:
    # tmp_words = mytool.seg_sentence(sentence,stopwordslist)
    X_train.append(words2index(word))



lstm_lenth_list = []
for word in X_train:
    lstm_lenth_list.append(len(word))






# 3 机器学习训练
model_max_len = 1300
warnings.filterwarnings("ignore")


# 配置参数

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001
    

class ModelConfig(object):
    embeddingSize = 200
    
    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 2  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.8  # multi head attention 中的dropout
    
    dropoutKeepProb = 0.5 # 全连接层的dropout
    l2RegLambda = 0.0
    
    
class Config(object):
    sequenceLength = 200  # 取了所有序列长度的均值
    batchSize = 128
    
    dataSource = "../data/preProcess/labeledTrain.csv"
    
    stopWordSource = "../data/english"
    
    numClasses = 16  # 二分类设置为1，多分类设置为类别的数目
    
    rate = 0.8  # 训练集的比例
    
    training = TrainingConfig()
    
    model = ModelConfig()

    
# 实例化配置参数对象
config = Config()




# 数据预处理的类，生成训练集和测试集

class Dataset(object):
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource  
        
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate
        
        self._stopWordDict = {}
        
        self.trainReviews = []
        self.trainLabels = []
        
        self.evalReviews = []
        self.evalLabels = []
        
        self.wordEmbedding =None
        
        self.labelList = []
        
    def _readData(self, filePath):
        """
        从csv文件中读取数据集
        """
        
        df = pd.read_csv(filePath)
        
        if self.config.numClasses == 1:
            labels = df["sentiment"].tolist()
        elif self.config.numClasses > 1:
            labels = df["rate"].tolist()
            
        review = df["review"].tolist()
        reviews = [line.strip().split() for line in review]

        return reviews, labels
    
    def _labelToIndex(self, labels, label2idx):
        """
        将标签转换成索引表示
        """
        labelIds = [label2idx[label] for label in labels]
        return labelIds
    
    def _wordToIndex(self, reviews, word2idx):
        """
        将词转换成索引
        """
        reviewIds = [[word2idx.get(item, word2idx["UNK"]) for item in review] for review in reviews]
        return reviewIds
        
    def _genTrainEvalData(self, x, y, word2idx, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        for review in x:
            if len(review) >= self._sequenceLength:
                reviews.append(review[:self._sequenceLength])
            else:
                reviews.append(review + [word2idx["PAD"]] * (self._sequenceLength - len(review)))
            
        trainIndex = int(len(x) * rate)
        
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(y[:trainIndex], dtype="float32")
        
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(y[trainIndex:], dtype="float32")

        return trainReviews, trainLabels, evalReviews, evalLabels
        
    def _genVocabulary(self, reviews, labels):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        
        allWords = [word for review in reviews for word in review]
        
        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]
        
        wordCount = Counter(subWords)  # 统计词频
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        
        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= 5]
        
        vocab, wordEmbedding = self._getWordEmbedding(words)
        self.wordEmbedding = wordEmbedding
        
        word2idx = dict(zip(vocab, list(range(len(vocab)))))
        
        uniqueLabel = list(set(labels))
        label2idx = dict(zip(uniqueLabel, list(range(len(uniqueLabel)))))
        self.labelList = list(range(len(uniqueLabel)))
        
        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/word2idx.json", "w", encoding="utf-8") as f:
            json.dump(word2idx, f)
        
        with open("../data/wordJson/label2idx.json", "w", encoding="utf-8") as f:
            json.dump(label2idx, f)
        
        return word2idx, label2idx
            
    def _getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        
        wordVec = gensim.models.KeyedVectors.load_word2vec_format("../word2vec/word2Vec.bin", binary=True)
        vocab = []
        wordEmbedding = []
        
        # 添加 "pad" 和 "UNK", 
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        
        for word in words:
            try:
                vector = wordVec.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")
                
        return vocab, np.array(wordEmbedding)
    
    def _readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
            
    def dataGen(self):
        """
        初始化训练集和验证集
        """
        
        # 初始化停用词
        self._readStopWord(self._stopWordSource)
        
        # 初始化数据集
        reviews, labels = self._readData(self._dataSource)
        
        # 初始化词汇-索引映射表和词向量矩阵
        word2idx, label2idx = self._genVocabulary(reviews, labels)
        
        # 将标签和句子数值化
        labelIds = self._labelToIndex(labels, label2idx)
        reviewIds = self._wordToIndex(reviews, word2idx)
        
        # 初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self._genTrainEvalData(reviewIds, labelIds, word2idx, self._rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels
        
        
data = Dataset(config)
data.dataGen()


# 输出batch数据集

def nextBatch(x, y, batchSize):
        """
        生成batch数据集，用生成器的方式输出
        """
    
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]
        
        numBatches = len(x) // batchSize

        for i in range(numBatches):
            start = i * batchSize
            end = start + batchSize
            batchX = np.array(x[start: end], dtype="int64")
            batchY = np.array(y[start: end], dtype="float32")
            
            yield batchX, batchY



# 生成位置嵌入
def fixedPositionEmbedding(batchSize, sequenceLen):
    embeddedPosition = []
    for batch in range(batchSize):
        x = []
        for step in range(sequenceLen):
            a = np.zeros(sequenceLen)
            a[step] = 1
            x.append(a)
        embeddedPosition.append(x)
    
    return np.array(embeddedPosition, dtype="float32")


# 模型构建

class Transformer(object):
    """
    Transformer Encoder 用于文本分类
    """
    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        # self.inputY = tf.placeholder(tf.int32, [None], name="inputY")
        self.inputY = tf.placeholder(tf.float32, [None, config.numClasses], name="inputY")
        
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.embeddedPosition = tf.placeholder(tf.float32, [None, config.sequenceLength, config.sequenceLength], name="embeddedPosition")
        
        self.config = config
        
        # 定义l2损失
        l2Loss = tf.constant(0.0)
        
        # 词嵌入层, 位置向量的定义方式有两种：一是直接用固定的one-hot的形式传入，然后和词向量拼接，在当前的数据集上表现效果更好。另一种
        # 就是按照论文中的方法实现，这样的效果反而更差，可能是增大了模型的复杂度，在小数据集上表现不佳。
        
        with tf.name_scope("embedding"):
            
            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec") ,name="W",trainable =False)
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embedded = tf.nn.embedding_lookup(self.W, self.inputX)
            self.embeddedWords = tf.concat([self.embedded, self.embeddedPosition], -1)

        with tf.name_scope("transformer"):
            for i in range(config.model.numBlocks):
                with tf.name_scope("transformer-{}".format(i + 1)):
            
                    # 维度[batch_size, sequence_length, embedding_size]
                    multiHeadAtt = self._multiheadAttention(rawKeys=self.inputX, queries=self.embeddedWords,
                                                            keys=self.embeddedWords)
                    # 维度[batch_size, sequence_length, embedding_size]
                    self.embeddedWords = self._feedForward(multiHeadAtt, 
                                                           [config.model.filters, config.model.embeddingSize + config.sequenceLength])
                
            outputs = tf.reshape(self.embeddedWords, [-1, config.sequenceLength * (config.model.embeddingSize + config.sequenceLength)])

        outputSize = outputs.get_shape()[-1].value


        
        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.dropoutKeepProb)
    
        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            
            outputB= tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(outputs, outputW, outputB, name="logits")
            
            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=-1, name="predictions")
        
        # 计算二元交叉熵损失
        with tf.name_scope("loss"):
            
            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(tf.reshape(self.inputY, [-1, 1]), 
                                                                                                    dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.inputY)
                
            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss
            
    def _layerNormalization(self, inputs, scope="layerNorm"):
        # LayerNorm层和BN层有所不同
        epsilon = self.config.model.epsilon

        inputsShape = inputs.get_shape() # [batch_size, sequence_length, embedding_size]

        paramsShape = inputsShape[-1:]

        # LayerNorm是在最后的维度上计算输入的数据的均值和方差，BN层是考虑所有维度的
        # mean, variance的维度都是[batch_size, sequence_len, 1]
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)

        beta = tf.Variable(tf.zeros(paramsShape))

        gamma = tf.Variable(tf.ones(paramsShape))
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)
        
        outputs = gamma * normalized + beta

        return outputs

    def _multiheadAttention(self, rawKeys, queries, keys, numUnits=None, causality=False, scope="multiheadAttention"):
        # rawKeys 的作用是为了计算mask时用的，因为keys是加上了position embedding的，其中不存在padding为0的值
        
        numHeads = self.config.model.numHeads
        keepProp = self.config.model.keepProp
        
        if numUnits is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            numUnits = queries.get_shape().as_list()[-1]

        # tf.layers.dense可以做多维tensor数据的非线性映射，在计算self-Attention时，一定要对这三个值进行非线性映射，
        # 其实这一步就是论文中Multi-Head Attention中的对分割后的数据进行权重映射的步骤，我们在这里先映射后分割，原则上是一样的。
        # Q, K, V的维度都是[batch_size, sequence_length, embedding_size]
        Q = tf.layers.dense(queries, numUnits, activation=tf.nn.relu)
        K = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)
        V = tf.layers.dense(keys, numUnits, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, numHeads, axis=-1), axis=0) 
        K_ = tf.concat(tf.split(K, numHeads, axis=-1), axis=0) 
        V_ = tf.concat(tf.split(V, numHeads, axis=-1), axis=0)

        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similary = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        scaledSimilary = similary / (K_.get_shape().as_list()[-1] ** 0.5)

        # 在我们输入的序列中会存在padding这个样的填充词，这种词应该对最终的结果是毫无帮助的，原则上说当padding都是输入0时，
        # 计算出来的权重应该也是0，但是在transformer中引入了位置向量，当和位置向量相加之后，其值就不为0了，因此在添加位置向量
        # 之前，我们需要将其mask为0。虽然在queries中也存在这样的填充词，但原则上模型的结果之和输入有关，而且在self-Attention中
        # queryies = keys，因此只要一方为0，计算出的权重就为0。
        # 具体关于key mask的介绍可以看看这里： https://github.com/Kyubyong/transformer/issues/3

        # 利用tf，tile进行张量扩张， 维度[batch_size * numHeads, keys_len] keys_len = keys 的序列长度
        keyMasks = tf.tile(rawKeys, [numHeads, 1]) 

        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        keyMasks = tf.tile(tf.expand_dims(keyMasks, 1), [1, tf.shape(queries)[1], 1])

        # tf.ones_like生成元素全为1，维度和scaledSimilary相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(scaledSimilary) * (-2 ** (32 + 1))

        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        maskedSimilary = tf.where(tf.equal(keyMasks, 0), paddings, scaledSimilary) # 维度[batch_size * numHeads, queries_len, key_len]

        # 在计算当前的词时，只考虑上文，不考虑下文，出现在Transformer Decoder中。在文本分类时，可以只用Transformer Encoder。
        # Decoder是生成模型，主要用在语言生成中
        if causality:
            diagVals = tf.ones_like(maskedSimilary[0, :, :])  # [queries_len, keys_len]
            tril = tf.contrib.linalg.LinearOperatorTriL(diagVals).to_dense()  # [queries_len, keys_len]
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(maskedSimilary)[0], 1, 1])  # [batch_size * numHeads, queries_len, keys_len]

            paddings = tf.ones_like(masks) * (-2 ** (32 + 1))
            maskedSimilary = tf.where(tf.equal(masks, 0), paddings, maskedSimilary)  # [batch_size * numHeads, queries_len, keys_len]

        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(maskedSimilary)

        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(weights, V_)

        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, numHeads, axis=0), axis=2)
        
        outputs = tf.nn.dropout(outputs, keep_prob=keepProp)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layerNormalization(outputs)
        return outputs

    def _feedForward(self, inputs, filters, scope="multiheadAttention"):
        # 在这里的前向传播采用卷积神经网络
        
        # 内层
        params = {"inputs": inputs, "filters": filters[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # 外层
        params = {"inputs": outputs, "filters": filters[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}

        # 这里用到了一维卷积，实际上卷积核尺寸还是二维的，只是只需要指定高度，宽度和embedding size的尺寸一致
        # 维度[batch_size, sequence_length, embedding_size]
        outputs = tf.layers.conv1d(**params)

        # 残差连接
        outputs += inputs

        # 归一化处理
        outputs = self._layerNormalization(outputs)

        return outputs
    
    def _positionEmbedding(self, scope="positionEmbedding"):
        # 生成可训练的位置向量
        batchSize = self.config.batchSize
        sequenceLen = self.config.sequenceLength
        embeddingSize = self.config.model.embeddingSize
        
        # 生成位置的索引，并扩张到batch中所有的样本上
        positionIndex = tf.tile(tf.expand_dims(tf.range(sequenceLen), 0), [batchSize, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        positionEmbedding = np.array([[pos / np.power(10000, (i-i%2) / embeddingSize) for i in range(embeddingSize)] 
                                      for pos in range(sequenceLen)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        positionEmbedding[:, 0::2] = np.sin(positionEmbedding[:, 0::2])
        positionEmbedding[:, 1::2] = np.cos(positionEmbedding[:, 1::2])

        # 将positionEmbedding转换成tensor的格式
        positionEmbedding_ = tf.cast(positionEmbedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        positionEmbedded = tf.nn.embedding_lookup(positionEmbedding_, positionIndex)

        return positionEmbedded



"""
定义各类性能指标
"""

def mean(item: list) -> float:
    """
    计算列表中元素的平均值
    :param item: 列表对象
    :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def binary_precision(pred_y, true_y, positive=1):
    """
    二类的精确率计算
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    pred_corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == positive:
            pred_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    prec = corr / pred_corr if pred_corr > 0 else 0
    return prec


def binary_recall(pred_y, true_y, positive=1):
    """
    二类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param positive: 正例的索引表示
    :return:
    """
    corr = 0
    true_corr = 0
    for i in range(len(pred_y)):
        if true_y[i] == positive:
            true_corr += 1
            if pred_y[i] == true_y[i]:
                corr += 1

    rec = corr / true_corr if true_corr > 0 else 0
    return rec


def binary_f_beta(pred_y, true_y, beta=1.0, positive=1):
    """
    二类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param beta: beta值
    :param positive: 正例的索引表示
    :return:
    """
    precision = binary_precision(pred_y, true_y, positive)
    recall = binary_recall(pred_y, true_y, positive)
    try:
        f_b = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
    except:
        f_b = 0
    return f_b


def multi_precision(pred_y, true_y, labels):
    """
    多类的精确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    多类的召回率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    多类的f beta值
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :param labels: 标签列表
    :param beta: beta值
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    得到二分类的性能指标
    :param pred_y:
    :param true_y:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):

    """
    得到多分类的性能指标
    :param pred_y:
    :param true_y:
    :param labels:
    :param f_beta:
    :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta




# 训练模型

# 生成训练集和验证集
trainReviews = data.trainReviews
trainLabels = data.trainLabels
evalReviews = data.evalReviews
evalLabels = data.evalLabels

wordEmbedding = data.wordEmbedding
labelList = data.labelList

embeddedPosition = fixedPositionEmbedding(config.batchSize, config.sequenceLength)

# 定义计算图
with tf.Graph().as_default():

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  

    sess = tf.Session(config=session_conf)
    
    # 定义会话
    with sess.as_default():
        transformer = Transformer(config, wordEmbedding)
        
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(transformer.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        
        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        
        outDir = os.path.abspath(os.path.join(os.path.curdir, "summarys"))
        print("Writing to {}\n".format(outDir))
        
        lossSummary = tf.summary.scalar("loss", transformer.loss)
        summaryOp = tf.summary.merge_all()
        
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
        
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
        
        
        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        
        # 保存模型的一种方式，保存为pb文件
        savedModelPath = "../model/transformer/savedModel"
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)
            
        sess.run(tf.global_variables_initializer())

        def trainStep(batchX, batchY):
            """
            训练函数
            """   
            feed_dict = {
              transformer.inputX: batchX,
              transformer.inputY: batchY,
              transformer.dropoutKeepProb: config.model.dropoutKeepProb,
              transformer.embeddedPosition: embeddedPosition
            }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, transformer.loss, transformer.predictions],
                feed_dict)
            
            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

                
            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                              labels=labelList)
                
            trainSummaryWriter.add_summary(summary, step)
            
            return loss, acc, prec, recall, f_beta

        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
              transformer.inputX: batchX,
              transformer.inputY: batchY,
              transformer.dropoutKeepProb: 1.0,
              transformer.embeddedPosition: embeddedPosition
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, transformer.loss, transformer.predictions],
                feed_dict)
            
            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

                
            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY,
                                                              labels=labelList)
                
            trainSummaryWriter.add_summary(summary, step)
            
            return loss, acc, prec, recall, f_beta
        
        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])
                
                currentStep = tf.train.global_step(sess, globalStep) 
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    
                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []
                    
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)
                        
                    time_str = datetime.datetime.now().isoformat()
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str, currentStep, mean(losses), 
                                                                                                       mean(accs), mean(precisions),
                                                                                                       mean(recalls), mean(f_betas)))
                    
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, "../model/Transformer/model/my-model", global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
                    
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(transformer.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(transformer.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(transformer.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

        builder.save()