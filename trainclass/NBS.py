#-- coding: utf-8 --
from sklearn.datasets import fetch_20newsgroups  # 从sklearn.datasets里导入新闻数据抓取器 fetch_20newsgroups
from sklearn.model_selection import  train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # 从sklearn.feature_extraction.text里导入文本特征向量化模块
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import wordninja
import numpy as np

namelist = []

classtype = {'购物':'购物网站','游戏':'休闲娱乐','旅游':'生活服务','军事':'教育文化','招聘':'生活服务','时尚':'休闲娱乐',
'新闻':'新闻媒体资讯','音乐':'休闲娱乐','健康':'医疗健康','艺术':'教育文化','社区':'综合其他','学习':'教育文化','政府':'政府组织',
'搞笑':'休闲娱乐','银行':'生活服务','酷站':'综合其他','视频':'休闲娱乐','电影':'休闲娱乐','文学':'休闲娱乐','体育':'体育健身',
'科技':'网络科技','财经':'生活服务','汽车':'生活服务','房产':'生活服务','摄影':'休闲娱乐','设计':'网络科技','营销':'行业企业',
'电商':'购物网站','外贸':'行业企业','服务':'行业企业','商界':'行业企业','生活':'生活服务'}
# "购物"：'购物网站', '游戏':'休闲娱乐','旅游':'交通旅游','军事':'教育文化','招聘':'生活服务','新闻':'新闻媒体','音乐':'休闲娱乐'
def initclass(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
                # print(line)
            parts = line.split(",")
            if  not line:
                break
            classtype[parts[0]]=parts[2].strip('\n')

def initstep():
    filepath = "D:/dnswork/sharevm/top.chinaz.txt"
    initclass(filepath)
    # filepath = "D:/dnswork/sharevm/alexaclass.txt"
    # initclass(filepath)


def getnamedata():
    resultdata = []
    resultclass = []
    data=[]
    classname=[]
    datapath = "D:/dnswork/sharevm/hao.66360.cn.txt"
    data, classname = getclassifydata(datapath)
    resultdata = resultdata +data
    resultclass = resultclass + classname

    datapath = "D:/dnswork/sharevm/123.sogou.com.txt"
    data, classname= getclassifydata(datapath)
    resultdata = resultdata +data
    resultclass = resultclass + classname

    # datapath = "D:/dnswork/sharevm/hao123.txt"
    # data, classname= getclassifydata(datapath)

    # resultdata = resultdata +data
    # resultclass = resultclass + classname

    # datapath = "D:/dnswork/sharevm/hao.360.com.txt"
    # data, classname= getclassifydata(datapath)
    # resultdata = resultdata +data
    # resultclass = resultclass + classname

    # path = "D:/dnswork/sharevm/alexclassify/"
    # fs = os.listdir(path)
    # for filename in fs :
    #     # print(filename)
    #     data, classname= getclassifydata(path + filename)
    #     resultdata = resultdata +data
    #     resultclass = resultclass + classname
    
    path = "D:/dnswork/sharevm/topchinaz/"
    fs = os.listdir(path)
    for filename in fs :
        # print(filename)
        data, classname=getclassifydata(path + filename)
        resultdata = resultdata +data
        resultclass = resultclass + classname
    return resultdata, resultclass

def getclassifydata(path):
    data= []
    classname = []
    with open(path, 'r', encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
            # print(line)
            parts = line.split(",")
            if  line =="":
                break
            if  line =="\n":
                break
            if len(parts)>=2:
                name = parts[1]
                nameparts = name.split("//")
                if len(nameparts)==2:
                    name = nameparts[1]
                nameparts = name.split("www.")
                if len(nameparts)==2:
                    name = nameparts[1]
                nameparts = name.split("/")
                if len(nameparts) >=2:
                    name = nameparts[0]
                if name!='':
                    if name not in namelist:
                        # print(name)
                        namelist.append(name)
                        data.append(name)
                        classname.append(classtype[parts[0]])
    return data, classname

def createVocabList(dataSet):
    """
    获取去重后的词汇表

    :param dataSet:
    :return:
    """
    vocabSet = set()
    for document in dataSet:
        vocabSet |= set(document)
    return list(vocabSet)

def wordsToVector(dataList, vocabularys):
    """
    将原始数据向量化，向量的每个元素为1或0

    :param vocabularys: createVocabList返回的列表
    :param dataList: 切分的词条列表
    :return: 文档向量,词集模型
    """
    vector = [0] * len(vocabularys)
    for word in dataList:  # 遍历每个词条
        if word in vocabularys:  # 如果词条存在于词汇表中，则置1
            vector[vocabularys.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return vector  # 返回文档向量

initstep()
#1.数据获取
# news = fetch_20newsgroups(subset='all')
# print(len(news.data)  )# 输出数据的条数：18846
# print(news.data[0])
# print(news)
namedata, nameclass = getnamedata()
print(len(namedata))

# dataset = createVocabList(namedata)
# print(len(dataset))

#2.数据预处理：训练集和测试集分割，文本特征向量化
# X_train,X_test,y_train,y_test = train_test_split(news.data,news.target,test_size=0.25,random_state=33) # 随机采样25%的数据样本作为测试集
X_train,X_test,y_train,y_test = train_test_split(namedata,nameclass,test_size=0.25) # 随机采样25%的数据样本作为测试集


xsplit_train=[]
for pername in X_train:
    inputname =""
    name =  wordninja.split(pername)
    for i in name:
        inputname = inputname +" "+i
    xsplit_train.append(inputname)
    
xsplit_test=[]
for pername in X_test:
    inputname =""
    name =  wordninja.split(pername)
    for i in name:
        inputname = inputname +" "+i
    xsplit_test.append(inputname)      

# 文本特征向量化
vec = CountVectorizer()
xvec_train = vec.fit_transform(xsplit_train)
# print(X_train)
Xvec_test = vec.transform(xsplit_test)
# print(vec.get_feature_names())

# X_train = np.array(X_train).reshape(1, -1)
# tfidf = TfidfVectorizer()
# tfidf.fit(fit_trainlist)
# # traindata = []
# # for x in X_train:
# #     traindata.append( tfidf.transform(x))
# traindata = tfidf.fit_transform(my_train)
# testdata=[]
# for x in X_test:
#     testdata.append( tfidf.transform(x))

# testdata = tfidf.transform(X_test)
# print (X_train[0:10] ) #查看训练样本
# print (X_test[0:10])  #查看标签

#3.使用朴素贝叶斯进行训练
mnb = MultinomialNB()   # 使用默认配置初始化朴素贝叶斯
mnb.fit(xvec_train,y_train)    # 利用训练数据对模型参数进行估计
y_predict = mnb.predict(Xvec_test)     # 对参数进行预测


#查看预测错误情况
# print(y_predict)
for index,value in enumerate(y_predict):
    if value != y_test[index]:
        print(X_test[index], value, y_test[index])


#4.获取结果报告
# print ('The Accuracy of Naive Bayes Classifier is:', mnb.score(X_test,y_test))
print ('The Accuracy of Naive Bayes Classifier is:', mnb.score(Xvec_test,y_test))
# print (classification_report(y_test, y_predict, target_names = news.target_names))
print (classification_report(y_test, y_predict))