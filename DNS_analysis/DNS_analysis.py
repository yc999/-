#-- coding: utf-8 --
import  requests
import re
# import eventlet
import os
import sys
import io
import json
from selenium.webdriver.firefox.options import Options
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

from selenium import webdriver


time_limit = 40  #set timeout time 3s







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
cdnlist_path = "/home/jiangy2/dnswork/cdnlist/cdnlist.txt"
cdnlist = mytool.read_cdnlist(cdnlist_path)



# 1.5 加载 tldlist
tldlist = []
tldlist_path = "/home/jiangy2/dnswork/cdnlist/tldlist.txt"
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

class_list = [ '休闲娱乐', '生活服务', '购物网站', '政府组织', '综合其他', '教育文化', '行业企业','网络科技',
 '体育健身', '医疗健康', '交通旅游', '新闻媒体']

def predict_webclass(webdata):
    X_train_text = []
    tmp_data = ""
    for data in webdata['webtext']:
        tmp_data=tmp_data + data
    len_webtext = len(tmp_data)
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    len_chinese = len(rule.sub('',tmp_data))
    if len_chinese/len_webtext < 0.5:
        return "外语网站"
    if len(webdata['webtext'])>=15:
        X_train_text.append(mytool.get_all_webdata(webdata))
    else:
        return "数据过少"
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
    predicted = class_list[np.argmax(predicted)]
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
        for cdn in cdnlist:
            if '.'+name == cdn:
                return True
    return False


dnstpye_value = {'1' : "A", '5':"CNAME", '28':"AAAA"}

# 读取dns数据
# dnsdata_path = "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2020-11/pdns_data"
dnsdata_path = "/home/jiangy2/dnswork/cdnlist/pdns_data"

dnsdata_file = open(dnsdata_path, 'r', encoding='utf-8')
while True:
    line = dnsdata_file.readline()
    try:
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
                resultdata = mytool.requesturl(httpsurl,browser, time_limit)
                if mytool.ifbadtitle(resultdata['title']):
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
                    resultdata = mytool.requesturl(httpsurl, browser, time_limit)
                    #网页是否无法访问
                    if mytool.ifbadtitle(resultdata['title']):
                        raise Exception("title error")
                    predict_result = predict_webclass(resultdata)
                    print(predict_result)
                except Exception as e:
                    print(e)
        else:
            break
    except:
        pass
