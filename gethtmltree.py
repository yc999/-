# -*- coding: utf-8 -*-
import sys
from bs4 import  BeautifulSoup, Comment
import os
import json



htmlfile_path = "E:/服务器数据/dnswork/httpwebdata/IT资讯/cena.com.cn.txt"

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

webdatadic = read_webdata(htmlfile_path)
webkey = list(webdatadic.keys())
htmldata = webdatadic[webkey[0]]

soup = BeautifulSoup(htmldata,'html.parser')

# print(soup)
body = soup.find("body")
# print(body)
bodysilb = body.find_next_siblings() #返回后面所有兄弟节点的列表
# print(bodysilb)
# for child in body.children:
#     print(child.name)
#     print()
#  <a>, <div>, <li>, <span>, <img>, <td>, 
# <p>, <ul>, <option>, <meta>, <tr>, <link>, <input>, 
# <table>, <tbody>, <dd>, <h2>, <h3>, <hr>, and <dt>.
taglist = ["a","div","li","span","img","td","p","ul","option","meta","tr","link","input","table",
"tbody","dd","h2","h3","hr","dt"]

input_list = []
for child in body.descendants:
    if child.name in taglist:
        # print(child.name)
        input_list.append(child.name)
# bodychild = body.children() #返回后面所有兄弟节点的列表
# print(bodychild)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer(token_pattern=r"(?u)\b\w+\b")
# X = vectorizer.fit_transform(X_train)
X = vectorizer.fit_transform(taglist)

X_train_vec = vectorizer.transform(input_list)
print("X_train_vec")
print(X_train_vec[0])

print(type(X_train_vec))


from sklearn.svm import SVC
import numpy as np


def func(x, a, b, c):
    return a * np.exp(-b * x) + c


np.random.seed(1719)
y_noise =  np.random.normal(size=5)
# for i in range(100):
x1 = np.random.randint(4, size=(5,5,5))
x2 = np.random.randint(4,8, size=(5,5,5))
x3 = np.random.randint(8,12, size=(5,5,5))
xtrain = []
xtrain.append(x1)
xtrain.append(x2)
xtrain.append(x3)
xtrain = np.array(xtrain)
y = [1,1,1,1,1,
     2,2,2,2,2,
     3,3,3,3,3]
# print(x1)
# print(x2)
# print(x3)

# clf = SVC(probability = True)
clf = SVC()
clf.fit(xtrain,y)
