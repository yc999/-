
# -*- coding: utf-8 -*-
import jieba
import numpy as np
import os
import time
import codecs
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
# stopwords_path = "C:/Users/shinelon/Desktop/linuxfirefox/stopwords-master/stopwords-master/cn_stopwords.txt"
stopwords_path = "/home/jiangy2/dnswork/stopwords/cn_stopwords.txt"
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


# 返回将字符串切成词语list返回，并去除空格等符号
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
# 其他页面中的词语 如果没有在主页中没出现  才放入列表中, 少于10个词的不考虑
def readtrain(filepath):
    global MAX_SEQUENCE_LENGTH 
    webdatadic = mytool.read_webdata(filepath)
    result_list = []
    result_dic = {}
    webkey = list(webdatadic.keys())
    for htmldata in webkey:
        htmltext = filtertext(webdatadic[htmldata]) # 读取网页中的纯文本
        if htmltext == False:
            result_dic[htmldata] = []
            continue
        cut_text = Word_cut_list(htmltext) # 生成词列表
        result_dic[htmldata] = cut_text
    result_list += result_dic[webkey[0]]  # 先保存首页
    for htmldata in webkey[1:]:
        for word in result_dic[htmldata]:
            if word not in result_dic[webkey[0]]:  # 只保存不在首页中出现的词语
                result_list.append(word)
    if len(result_list) < 10:
        return []
    if len(result_list)> MAX_SEQUENCE_LENGTH:
       MAX_SEQUENCE_LENGTH = len(result_list)
    return result_list


def read_all_data(datapath):
    """
    返回空格分隔的词语串
    """
    data = readtrain(datapath)
    if data == []:
        return ""
    cut_text = ' '.join(data)
    # result_data.append(cut_text)
    return cut_text

class_list = [ '休闲娱乐', '生活服务', '购物网站', '政府组织', '综合其他', '教育文化', '行业企业','网络科技',
 '体育健身', '医疗健康', '交通旅游', '新闻媒体']
class_index = { '休闲娱乐':0, '生活服务':1, '购物网站':2, '政府组织':3, '综合其他':4, '教育文化':5, '行业企业':6,'网络科技':7,
 '体育健身': 8, '医疗健康':9, '交通旅游':10, '新闻媒体':11}


classtype =  {'购物':'购物网站','游戏':'休闲娱乐','旅游':'生活服务','军事':'教育文化','招聘':'生活服务','时尚':'休闲娱乐',
    '新闻':'新闻媒体资讯','音乐':'休闲娱乐','健康':'医疗健康','艺术':'教育文化',  '社区':'综合其他',
    '学习':'教育文化','政府':'政府组织','搞笑':'休闲娱乐','银行':'生活服务',
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


# initfilepath = "D:/dnswork/sharevm/top.chinaz.txt"
initfilepath = "/home/jiangy2/dnswork/top.chinaz.txt"
initclass(initfilepath)


content_train_src = []      # 训练集文本列表
opinion_train_stc = []      # 训练集类别列表


# 获取全部数据集
# path = "/home/jiangy2/webdata/"
# path = "D:/dnswork/sharevm/topchinaz/"
# datapath = "E:/webdata/"

datapath = "/home/jiangy2/httpwebdata/"


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

# print("opinion_train_stc ", opinion_train_stc)
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



vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b",min_df = 15)
X = vectorizer.fit_transform(content_train_src)

transform = TfidfTransformer()
x_tfidf = transform.fit_transform(X)

#训练 用fit_transform
# count_train=vectorizer.fit_transform(content_train)
# tfidf = transform.fit_transform(count_train)

Y_label=to_categorical(opinion_train_stc, len(class_index))
# x_train, x_test, y_train, y_test = train_test_split(x_tfidf, opinion_train_stc, test_size=0.1)
X_train, X_test, y_train, y_test = train_test_split(content_train_src, opinion_train_stc, test_size=0.1)  #划分词语

X_train_vec = vectorizer.transform(X_train)
x_train = transform.transform(X_train_vec)

X_test_vec = vectorizer.transform(X_test)
x_test = transform.transform(X_test_vec)

print(shape(x_train)) #(457, 62919)
print(shape(y_train))
clf = SVC()
clf.fit(x_train,y_train)

print(x_test[0])
print(shape(x_test))
pred_y  = clf.predict(x_test)
print(classification_report(y_test,pred_y))






"""
------------------------------------------------------
"""



from gensim.models import KeyedVectors

EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330

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



# 2.4 将文本转为张量
# X_train 训练数据
X_train_index = []   # 训练集下标
X_test_index = []    # 测试集下标

# 将单词转为词向量的下标,下标从1开始 返回下标的list
def words2index(words):
    index_list = []
    for word in words:
        if word in embeddings_index.keys():  # 单词是否在词向量中
            index_list.append(embeddings_index[word])
        else:
            print(word + "不存在于词向量中")
    return index_list


for sentence in X_train:
    tmp_words = sentence.split(" ")
    X_train_index.append(words2index(tmp_words))

for sentence in X_test:
    tmp_words = sentence.split(" ")
    X_test_index.append(words2index(tmp_words))

import os
import time
import datetime
import random

import warnings
from collections import Counter
from math import sqrt

import gensim
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
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
    numHeads = 4  # Attention 的头数
    numBlocks = 2  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.8  # multi head attention 中的dropout
    
    dropoutKeepProb = 0.5 # 全连接层的dropout
    l2RegLambda = 0.0
    
    
class Config(object):
    # sequenceLength = 200  # 取了所有序列长度的均值
    sequenceLength = model_max_len//2
    batchSize = 128
    
    dataSource = "../data/preProcess/labeledTrain.csv"
    
    stopWordSource = "../data/english"
    
    numClasses = len(class_list)  # 二分类设置为1，多分类设置为类别的数目
    
    rate = 0.8  # 训练集的比例
    
    training = TrainingConfig()
    
    model = ModelConfig()

    
# 实例化配置参数对象
config = Config()


# 输出batch数据集

def nextBatch(x, y, batchSize):
        """
        # 生成batch数据集，用生成器的方式输出

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
# 定义各类性能指标
"""


def mean(item: list):
    """
    # 计算列表中元素的平均值
    # :param item: 列表对象
    # :return:
    """
    res = sum(item) / len(item) if len(item) > 0 else 0
    return res


def accuracy(pred_y, true_y):
    """
    # 计算二类和多类的准确率
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :return:
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
    # 二类的精确率计算
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :param positive: 正例的索引表示
    # :return:
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
    # 二类的召回率
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :param positive: 正例的索引表示
    # :return:
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
    # 二类的f beta值
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :param beta: beta值
    # :param positive: 正例的索引表示
    # :return:
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
    # 多类的精确率
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :param labels: 标签列表
    # :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    precisions = [binary_precision(pred_y, true_y, label) for label in labels]
    prec = mean(precisions)
    return prec


def multi_recall(pred_y, true_y, labels):
    """
    # 多类的召回率
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :param labels: 标签列表
    # :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    recalls = [binary_recall(pred_y, true_y, label) for label in labels]
    rec = mean(recalls)
    return rec


def multi_f_beta(pred_y, true_y, labels, beta=1.0):
    """
    # 多类的f beta值
    # :param pred_y: 预测结果
    # :param true_y: 真实结果
    # :param labels: 标签列表
    # :param beta: beta值
    # :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]

    f_betas = [binary_f_beta(pred_y, true_y, beta, label) for label in labels]
    f_beta = mean(f_betas)
    return f_beta


def get_binary_metrics(pred_y, true_y, f_beta=1.0):
    """
    # 得到二分类的性能指标
    # :param pred_y:
    # :param true_y:
    # :param f_beta:
    # :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = binary_recall(pred_y, true_y)
    precision = binary_precision(pred_y, true_y)
    f_beta = binary_f_beta(pred_y, true_y, f_beta)
    return acc, recall, precision, f_beta


def get_multi_metrics(pred_y, true_y, labels, f_beta=1.0):
    """
    # 得到多分类的性能指标
    # :param pred_y:
    # :param true_y:
    # :param labels:
    # :param f_beta:
    # :return:
    """
    acc = accuracy(pred_y, true_y)
    recall = multi_recall(pred_y, true_y, labels)
    precision = multi_precision(pred_y, true_y, labels)
    f_beta = multi_f_beta(pred_y, true_y, labels, f_beta)
    return acc, recall, precision, f_beta


# 训练模型


# 生成训练集和验证集
# trainReviews = data.trainReviews
trainReviews = X_train_index
# trainLabels = data.trainLabels
trainLabels = to_categorical(y_train, len(class_index))


evalReviews = X_test_index
# evalLabels = data.evalLabels
trainLabels=to_categorical(y_test, len(class_index))

# wordEmbedding = data.wordEmbedding
wordEmbedding = embedding_matrix
labelList = class_list 

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
            # 训练函数
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

        # builder.save()


