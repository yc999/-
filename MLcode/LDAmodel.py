#-- coding: utf-8 --
# 把疑似的网页内容通过LDA模型聚类

import numpy as np
import pandas as pd
import re
import os
import sys

sys.path.append(os.path.realpath('./MLcode'))
sys.path.append(os.path.realpath('../MLcode'))
import mytool
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

classtype = {'购物':'购物网站','游戏':'休闲娱乐','旅游':'生活服务','军事':'教育文化','招聘':'生活服务',
'时尚':'休闲娱乐','新闻':'新闻媒体资讯','音乐':'休闲娱乐','健康':'医疗健康','艺术':'教育文化',
'社区':'综合其他','学习':'教育文化','政府':'政府组织','搞笑':'休闲娱乐','银行':'生活服务',
'酷站':'综合其他','视频':'休闲娱乐','电影':'休闲娱乐','文学':'休闲娱乐','体育':'体育健身','科技':'网络科技',
'财经':'生活服务','汽车':'生活服务','房产':'生活服务','摄影':'休闲娱乐','设计':'网络科技','营销':'行业企业',
'电商':'购物网站','外贸':'行业企业','服务':'行业企业','商界':'行业企业','生活':'生活服务'}



jieba.setLogLevel(logging.INFO)



# 读取停用词
stopwordslist = []
stopwordslist_path = '/home/jiangy2/dnswork/stopwords/cn_stopwords.txt'
stopwordslist =mytool.read_stopwords(stopwordslist_path)


# 读取数据
## 读取疑似的网页名 存入 filenamelist
webfilename = '../../firststep_list.txt'
filenamelist = []
with open(webfilename, 'r', encoding='utf-8') as file_to_read:
    # with open('test1.txt', 'r') as f1:
    filenamelist = file_to_read.readlines()
for i in range(0, len(filenamelist)):
    filenamelist[i] = filenamelist[i].rstrip('\n')

print("疑似网页共：" + len(filenamelist))

filepath = '/home/jiangy2/webdata'
## 读取文件夹 存入 dirlist
dirlist = []
for root, dirs, files in os.walk(filepath):
         # root 表示当前正在访问的文件夹路径
         # dirs 表示该文件夹下的子目录名list
         # files 表示该文件夹下的文件list
        # 遍历文件
        #  for f in files:
        #      print(os.path.join(root, f))
        # 遍历路径下所有的文件夹
        for d in dirs:
            dirlist.append(os.path.join(root, d))
            # print(os.path.join(root, d))

## 读取疑似的网页数据
webdata_list = []
for dirpath in dirlist:
    print(dirpath)
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            if f.replace(".txt","") in filenamelist: ###
                data = mytool.read_webdata(os.path.join(root, f))
                # print(os.path.join(root, f))
                # 网页数据存入一个list
                target_data = mytool.get_all_webdata(data)
                #分词
                tmp_words = mytool.seg_sentence(target_data, stopwordslist)
                webdata_list.append(tmp_words)

print(webdata_list[0])
print("读取疑似网页内容共：" + len(webdata_list))


#构建词频矩阵，训练LDA模型
dictionary = corpora.Dictionary(webdata_list)
# corpus[0]: [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1),...]
# corpus是把每条新闻ID化后的结果，每个元素是新闻中的每个词语，在字典中的ID和频率
corpus = [dictionary.doc2bow(text) for text in webdata_list]

lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=3)
topic_list = lda.print_topics(3)
print("3个主题的单词分布为：\n")
for topic in topic_list:
    print(topic)



# raw_documents = [
#     '0无偿居间介绍买卖毒品的行为应如何定性',
#     '1吸毒男动态持有大量毒品的行为该如何认定',
#     '2如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
#     '3为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
#     '4将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
#     '5为获报酬帮人购买毒品的行为该如何认定',
#     '6毒贩出狱后再次够买毒品途中被抓的行为认定',
#     '7虚夸毒品功效劝人吸食毒品的行为该如何认定',
#     '8妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
#     '9一方未签字办理的结婚登记是否有效',
#     '10夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
#     '11结婚前对方父母出资购买的住房写我们二人的名字有效吗',
#     '12身份证被别人冒用无法登记结婚怎么办？',
#     '13同居后又与他人登记结婚是否构成重婚罪',
#     '14未办登记只举办结婚仪式可起诉离婚吗',
#     '15同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
# ]





