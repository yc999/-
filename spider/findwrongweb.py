#-- coding: utf-8 --
# 找到爬过的非目标网页
import os

filepath = ''
# fs = os.listdir(filepath)   #读取url目录


#读取文件夹
dirlist = []
for root, dirs, files in os.walk(filepath):
         # root 表示当前正在访问的文件夹路径
         # dirs 表示该文件夹下的子目录名list
         # files 表示该文件夹下的文件list

        # 遍历文件
        #  for f in files:
        #      print(os.path.join(root, f))
        # 遍历所有的文件夹
        for d in dirs:
            dirlist.append(os.path.join(root, d))
            print(os.path.join(root, d))


import json
def read_webdata(filepath):
    with open(filepath, 'r', encoding='utf-8') as file_to_read:
        return json.loads(file_to_read.read())     
# 读取文件
for dirpath in dirlist:
    for f in files:
        data = read_webdata(os.path.join(root, f))
        print(os.path.join(root, f))

        #处理文件



import jieba

rule = re.compile(u"[^\u4E00-\u9FA5]")
sentence = rule.sub('',sentence)
sentence_seged = jieba.lcut(sentence.strip(),cut_all=bool_cut_all)
