#-- coding: utf-8 --
import numpy as np
import pandas as pd
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

classtype = {'购物':'购物网站','游戏':'休闲娱乐','旅游':'生活服务','军事':'教育文化','招聘':'生活服务',
'时尚':'休闲娱乐','新闻':'新闻媒体资讯','音乐':'休闲娱乐','健康':'医疗健康','艺术':'教育文化',
'社区':'综合其他','学习':'教育文化','政府':'政府组织','搞笑':'休闲娱乐','银行':'生活服务',
'酷站':'综合其他','视频':'休闲娱乐','电影':'休闲娱乐','文学':'休闲娱乐','体育':'体育健身','科技':'网络科技',
'财经':'生活服务','汽车':'生活服务','房产':'生活服务','摄影':'休闲娱乐','设计':'网络科技','营销':'行业企业',
'电商':'购物网站','外贸':'行业企业','服务':'行业企业','商界':'行业企业','生活':'生活服务'}

stopwordslist = []  # 停用词列表


jieba.setLogLevel(logging.INFO)
def initclass(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        while True:
            line = file_to_read.readline()
            parts = line.split(",")
            if  not line:
                break
            classtype[parts[0]]=parts[2].strip('\n')

def initstep():
    filepath = "D:/dnswork/sharevm/top.chinaz.txt"
    initclass(filepath)


#读取停用词
def read_stopwords(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

#读取保存的网页信息
def read_webdata(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        return json.loads(file_to_read.read())
        # print(webdata)

# 分割句子变成词语 bool_cut_all 是否全模式
def seg_sentence(sentence , bool_cut_all=False):
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    sentence = rule.sub('',sentence)
    sentence_seged = jieba.lcut(sentence.strip(),cut_all=bool_cut_all)
    print(sentence_seged)
    outstr = ''
    for word in sentence_seged:
        # word = word.lower()
        if word not in stopwordslist:
            if word != '\t':
                 outstr += word
                 outstr += " "
    return outstr

stopwordslist = read_stopwords("C:/Users/shinelon/Desktop/linuxfirefox/stopwords-master/stopwords-master/cn_stopwords.txt")
webdata = read_webdata("E:/webdata/旅游网站/sh.tuniu.com.txt")
print(webdata['title'])
titlewordslist = seg_sentence(webdata['title']) #需要词嵌入时用False 传统机器学习用True
print(titlewordslist)
titlewordslist = seg_sentence("第224期",True) #需要词嵌入时用False 传统机器学习用True
print(titlewordslist)

raw_documents = [
    '0无偿居间介绍买卖毒品的行为应如何定性',
    '1吸毒男动态持有大量毒品的行为该如何认定',
    '2如何区分是非法种植毒品原植物罪还是非法制造毒品罪',
    '3为毒贩贩卖毒品提供帮助构成贩卖毒品罪',
    '4将自己吸食的毒品原价转让给朋友吸食的行为该如何认定',
    '5为获报酬帮人购买毒品的行为该如何认定',
    '6毒贩出狱后再次够买毒品途中被抓的行为认定',
    '7虚夸毒品功效劝人吸食毒品的行为该如何认定',
    '8妻子下落不明丈夫又与他人登记结婚是否为无效婚姻',
    '9一方未签字办理的结婚登记是否有效',
    '10夫妻双方1990年按农村习俗举办婚礼没有结婚证 一方可否起诉离婚',
    '11结婚前对方父母出资购买的住房写我们二人的名字有效吗',
    '12身份证被别人冒用无法登记结婚怎么办？',
    '13同居后又与他人登记结婚是否构成重婚罪',
    '14未办登记只举办结婚仪式可起诉离婚吗',
    '15同居多年未办理结婚登记，是否可以向法院起诉要求离婚'
]
corpora_documents = []
for item_text in raw_documents:
    # print(item_text)
    item_str = jieba.lcut(item_text)
    # print(item_str)
    corpora_documents.append(item_str)

print(corpora_documents)

dictionary = corpora.Dictionary(corpora_documents)
corpus = [ dictionary.doc2bow(text) for text in corpora_documents ]
print(corpus)
# lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# for words in titlewordslist:
#     print(words)

# for line in raw_documents:
#         line = line.split('\t')[1]
#         line = re.sub(r'[^\u4e00-\u9fa5]+','',line)
#         line_seg = seg_depart(line.strip())
#         outputs.write(line_seg.strip() + '\n')





