#-- coding: utf-8 --
#自定义函数

#阅读文本
import json
def read_webdata(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        return json.loads(file_to_read.read())


import re
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


import jieba
# 分割句子变成词语 
# stopwordslist 停用词表
# bool_cut_all 分词模式是否为全模式
def seg_sentence(sentence , stopwordslist = [] , bool_cut_all=False):
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    sentence = rule.sub('',sentence)
    sentence_seged = jieba.lcut(sentence.strip(),cut_all=bool_cut_all)
    wordlist = []
    for word in sentence_seged:
        if  stopwordslist:
            if word not in stopwordslist:
                if word != '\t':
                    wordlist.append(word)
    return wordlist

import os
def read_stopwords(filepath):
    stopwords = []
    if os.path.exists(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
