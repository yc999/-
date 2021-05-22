# -*- coding: utf-8 -*-
# 估计每一类大概多少个网站
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
from numpy.lib.shape_base import expand_dims
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



#  85 
# 107 110 112 114  125


class_index = {'商务服务': 0, '教育资讯': 1, '动漫网站': 2, '返利比价': 3, '票务预订': 4, '出国留学': 5,
   '家电数码': 6, '音乐网站': 7, '健康资讯': 8, '门户网站': 9, '生活百科': 10, '虚拟现实': 11, 
   '旅行社': 12, '搜索引擎': 13, '视频电影': 14, '医院诊所': 15, '信息公司': 16, '广播电视': 17,
   '医学知识': 18, '房产网站': 19, '汽车配件': 20, ' 外语综合': 21, '在线教育': 22, '图书展馆': 23, 
   '新闻报刊': 24, '减肥瘦身': 25, '建筑材料': 26, '分类信息': 27, '招商加盟': 28, '电力水务': 29,
   '家居建材': 30, '网络硬盘': 31, '博客网站': 32, '健康保健': 33, '药品交易': 34, '日化用品': 35, 
   '培训机构': 36, 'QQ个性': 37, '棋牌健身': 38, '农林畜牧渔': 39, '极限运动': 40, 
   '域名主机': 41, '医疗器械': 42, '社科文艺': 43, '生育避孕': 44, '网络安全': 45,
    '汽车网站': 46, '聊天交友': 47, '地方网站': 48, '体育院校': 49, '家政服务': 50, 
    '金融财经': 51, '事业单位': 52, '军事国防': 53, 
   '体育用品': 54, '福彩体彩': 55, '酒店宾馆': 56, '常用查询': 57, '政府门户': 58, '团购网站': 59, 
   '设计素材': 60, '交易平台': 61, '电商服务': 62, '美容整形': 63, '收藏爱好': 64, '星座运势': 65, 
   '技术编程': 66, '广电通信': 67, '物流运输': 68, '旅游网站': 69, '男科医院': 70, '商业百货': 71, 
   '宠物玩具': 72, '驾校学车': 73, '广告营销': 74, '中小学校': 75, '邮件通信': 76, '银行保险': 77, 
   '创业投资': 78, '不孕不育': 79, '应用工具': 80, '购物分享': 81, '网贷平台': 82, '求职招聘': 83, 
   '社交网站': 84, '电子元器件': 85, '医疗职业': 86, '企业信息': 87, '纺织皮革': 88, '食品饮料': 89, 
   '汽车厂商': 90, '高等院校': 91, '电子商务': 92, '心理健康': 93, '政府职能': 94, '机械工业': 95, 
   '电脑硬件': 96, '影楼婚嫁': 97, '组织机构': 98, '家居小商品': 99, '幽默笑话': 100, '广告联盟': 101,
   '站长资源': 102, '化工能源': 103, '武术搏击': 104, '五金电工': 105, '论坛综合': 106, '球类运动': 107,
   '卫星地图': 108, '图片摄影': 109, '法律法规': 110, '水暖安防': 111, '母婴网站': 112, '服装配饰': 113,
   '网址导航': 114, '数据分析': 115, '教育考试': 116, '科学探索': 117, '单车汽摩': 118, '药品药学': 119, 
   '天文历史': 120, '包装印刷': 121, '医疗药企': 122, '游戏网站': 123, '明星粉丝': 124, '娱乐时尚': 125, 
   '证券网站': 126, '体育综合': 127, '妇幼医院': 128, '手机数码': 129, 'IT资讯': 130, '电子支付': 131, 
   '软件下载': 132, '石化能源': 133, '百科 辞典': 134, '餐饮美食': 135, '公共团体': 136, '交通地图': 137, 
   '违法网站': 138, '小说网站': 139, '电商网站': 140}

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
            if len(parts)>1:
                weburllist.append(parts[1].strip())
                classlist.append(parts[0])
    tmp = filename.split(".")[0]
    class_index[tmp] = i

print(weburllist)

# classtype 保存了所有的分类信息  子类名 ： 父类目
# class_index 保存了父类名对应的下标



#2.3 读取爬取的网页信息
# 数据存入 X_train_text 网页中所有语句合成一句
# 标签下标存入 Y_train
X_train_text = []
Y_train = []

#读取保存的网页信息
# path = "D:/dnswork/sharevm/topchinaz/"



i=0
j=0
def find_weburldata(weburl):
    path = "/home/jiangy2/webdata/"
    fs = os.listdir(path)
    for subpath in fs:
        filepath = os.path.join(path, subpath)
        if (os.path.isdir(filepath)):
            webdata_path = os.listdir(filepath)
            for filename in webdata_path:
                fileurl = filename.replace(".txt","")
                fileurl = fileurl.replace("www.","")
                stripfilename = weburl.replace("www.","")
                if fileurl == stripfilename:
                    webdata = mytool.read_webdata(os.path.join(filepath, filename))
                    if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
                        if len(webdata['webtext'])>=15:
                            return mytool.get_all_webdata(webdata)          
    newpath = "/home/jiangy2/dnswork/newwebdata/"
    newfs = os.listdir(newpath)
    for subpath in newfs:
        filepath = os.path.join(newpath, subpath)
        if (os.path.isdir(filepath)):
            webdata_path = os.listdir(filepath)
            for filename in webdata_path:
                fileurl = filename.replace(".txt","")
                fileurl = fileurl.replace("www.","")
                stripfilename = weburl.replace("www.","")
                if fileurl == stripfilename:
                    webdata = mytool.read_webdata(os.path.join(filepath, filename))
                    if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
                        if len(webdata['webtext'])>=15:
                            return mytool.get_all_webdata(webdata)
    return ""


for index,weburl in enumerate(weburllist):
    i = i+1
    webdata = find_weburldata(weburl)
    if webdata != "":
        j = j + 1
        X_train_text.append(webdata)
        Y_train.append(class_index[classlist[index]])


print("已爬取网页数：")
print(i)
print("有效网页数：")
print(j)

ids = list(set(Y_train))





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
model_max_len = 350


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
# x_train, x_test, y_train, y_test = train_test_split(X_train, Y, test_size=0.1)


x_train_raw = pad_sequences(X_train, maxlen=model_max_len)
# x_test_raw = pad_sequences(x_test, maxlen=model_max_len)





# 3.3 训练
def model_fit(model, x, y):
    return model.fit(x, y, batch_size=10, epochs=25, validation_split=0.1)


model = get_lstm_model()
model_train = model_fit(model, x_train_raw, Y)


# 3.4 测试
# print(model.evaluate(x_test_raw, y_test))

modelsave_path = "/home/jiangy2/dnswork/modeldir/tmp_LSTMmodel"
model.save(modelsave_path)




#开始预测
class_list = list(class_index.keys())


def predict_webclass(webdata,LSTM_model):
    X_train_text = []
    tmp_data = ""
    for data in webdata['webtext']:
        tmp_data=tmp_data + data
    len_webtext = len(tmp_data)
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    len_chinese = len(rule.sub('',tmp_data))
    if len_chinese/len_webtext < 0.2:
        return ""
    X_train_text.append(mytool.get_all_webdata(webdata))
    #  将文本转为张量
    # X_train 训练数据
    X_train = []
    for sentence in X_train_text:
        tmp_words = mytool.seg_sentence(sentence,stopwordslist)
        X_train.append(words2index(tmp_words))
    # 3 机器学习训练
    x_train_raw = pad_sequences(X_train, maxlen=model_max_len)
    predicted = LSTM_model.predict(x_train_raw)
    predicted = class_list[np.argmax(predicted)]
    return predicted


#读取文件
result_dic={}  #保存结果


newpath = "/home/jiangy2/dnswork/newwebdata/"
newfs = os.listdir(newpath)
for subpath in newfs:
    filepath = os.path.join(newpath, subpath)
    if (os.path.isdir(filepath)):
        print(filepath)
        webdata_path = os.listdir(filepath)
        for filename in webdata_path:
            fileurl = filename.replace(".txt","")
            fileurl = fileurl.replace("www.","")
            webdata = mytool.read_webdata(os.path.join(filepath, filename))
            if fileurl not in result_dic.keys():
                if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
                    if len(webdata['webtext'])>=15:
                        result = predict_webclass(webdata,model)
                        if result != "":
                            print(result)
                            result_dic[fileurl] = result

   
newpath = "/home/jiangy2/webdata/" 
newfs = os.listdir(newpath)

for subpath in newfs:
    filepath = os.path.join(newpath, subpath)
    if (os.path.isdir(filepath)):
        print(filepath)
        webdata_path = os.listdir(filepath)
        for filename in webdata_path:
            fileurl = filename.replace(".txt","")
            fileurl = fileurl.replace("www.","")
            webdata = mytool.read_webdata(os.path.join(filepath, filename))
            if fileurl not in result_dic.keys():
                if webdata['title'] != "" and webdata['description'] != "" and webdata['keywords'] != "":
                    if len(webdata['webtext'])>=15:
                        result = predict_webclass(webdata,model)
                        if result != "":
                            print(result)
                            result_dic[fileurl] = result

           
#统计结果数量                 
resultcount_dic={}

for key in result_dic.keys():
    print(key)
    if result_dic[key]  in resultcount_dic:
        resultcount_dic[result_dic[key]] = resultcount_dic[result_dic[key]] + 1
    else:
        resultcount_dic[result_dic[key]] = 1

sum = 0
for key in resultcount_dic:
    sum = sum + resultcount_dic[key]

for key in class_list:
    if key not in resultcount_dic:
        print(key)









result_dic['alipay.com']
result_dic['shfft.com']


result_dic['58wuji.com']
result_dic['onlinedown.net']
result_dic['downcc.com']


result_dic['mine999.cn']
result_dic['sinopec.com']
result_dic['cnpc.com.cn']
result_dic['pvc123.com']


result_dic['dict.cn']
result_dic['iciba.com']
result_dic['dict.youdao.com']
result_dic['chazidian.com']


result_dic['xiachufang.com']
result_dic['meishij.net']
result_dic['cy8.com.cn']
result_dic['shipuxiu.com']
result_dic['haocai777.com']
result_dic['ttmeishi.com']


result_dic['12371.cn']
result_dic['sac.net.cn']
result_dic['ccopyright.com.cn']
result_dic['rmzxb.com.cn']
result_dic['gwyoo.com']



result_dic['8684.cn']
result_dic['mapbar.com']
result_dic['amap.com']
result_dic['city8.com']
result_dic['keyunzhan.com']



result_dic['hebyyjzcgw.cn']
result_dic['box-z.com']
result_dic['jianbaishi.com']
result_dic['cjkq.ne']
result_dic['tech-marts.com']



result_dic['cqxfyy.com']
result_dic['bjhms.com']
result_dic['bjebhw.com']
result_dic['kedayy120.com']
result_dic['dx66.cn']
result_dic['dx021.com']
result_dic['dlyfj.com']
result_dic['fuke120.com']
result_dic['syjqzyy.com']
result_dic['keyunzhan.com']


result_dic['biquge.info']
result_dic['jjwxc.net']
result_dic['qidian.com']
result_dic['faloo.com']
result_dic['zongheng.com']
result_dic['xxsy.net']


result_dic['jd.com']
result_dic['1688.com']
result_dic['taobao.com']
result_dic['kaola.com']
result_dic['dangdang.com']
result_dic['suning.com']



