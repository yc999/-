import keras.backend as K
from keras.layers import LSTM, Input
import numpy as np
# I = Input(shape=(None, 200)) # unknown timespan, fixed feature size
# lstm = LSTM(20)
# f = K.function(inputs=[I], outputs=[lstm(I)])

# from keras import backend as K
import tensorflow as tf

# inputs = K.constant([1,2,3])
inputs = K.constant([[1, 2, 3], [1, 2, 0], [-1., -1., -1.]])

mask_value = [-1., -1., -1.]
output_mask = K.not_equal(inputs, mask_value)
print(output_mask)
# print(output_mask.eval())
# with tf.compat.v1.Session() as sess:
#     print(output_mask.eval())

# 输出结果如下：

# Tensor("NotEqual:0", shape=(3, 3), dtype=bool)
# [[ True  True  True]
#  [ True  True False]
#  [False False False]]

# docs = ['Well done!',
#         'Good work',
#         'Great effort',
#         'nice work',
#         'Excellent!',
#         'Weak',
#         'Poor effort!',
#         'not good',
#         'poor work',
#         'Could have done better.']
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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def seg_sentence(sentence , bool_cut_all=False):
    rule = re.compile(u"[^\u4E00-\u9FA5]")
    sentence = rule.sub('',sentence)
    sentence_seged = jieba.lcut(sentence.strip(),cut_all=bool_cut_all)
    # print(sentence_seged)
    # outstr = ''
    wordlist = []
    for word in sentence_seged:
        # word = word.lower()
        if word not in stopwordslist:
            if word != '\t':
                wordlist.append(word)
                #  outstr += word
                #  outstr += " "
    return wordlist
def read_stopwords(filepath):
    stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords
def read_webdata(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        return json.loads(file_to_read.read())

stopwordslist = read_stopwords("C:/Users/shinelon/Desktop/linuxfirefox/stopwords-master/stopwords-master/cn_stopwords.txt")
webdata = read_webdata("E:/webdata/旅游网站/sh.tuniu.com.txt")
# print(webdata['title'])
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
    # for data in webdata['webtext']:
    #     tmp = rule.sub('',data)
    #     tmp_data=tmp_data + tmp
    # for data in webdata['abouttext']:
    #     tmp = rule.sub('',data)
    #     tmp_data=tmp_data + tmp
    return tmp_data

print(seg_sentence(get_all_webdata(webdata),False))
# titlewordslist = seg_sentence(webdata['title']) #需要词嵌入时用False 传统机器学习用True
# print(titlewordslist)
