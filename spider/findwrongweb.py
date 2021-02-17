#-- coding: utf-8 --
# 找到爬过的非目标网页
import sys
# print(sys.path)
import os
print('\n')
# print(os.path.realpath('./MLcode/'))
sys.path.append(os.path.realpath('./MLcode'))
import mytool

print(sys.path)
filepath = '/home/jiangy2/webdata'
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

count_words_dic = {}

# 读取停用词
stopwordslist = []
stopwordslist_path = '/home/jiangy2/dnswork/stopwords/cn_stopwords.txt'
stopwordslist =mytool.read_stopwords(stopwordslist_path)

# 读取文件
for dirpath in dirlist:
    for f in files:
        data = mytool.read_webdata(os.path.join(root, f))
        print(os.path.join(root, f))
        # 网页数据存入一个list
        target_data = mytool.get_all_webdata(data)
        #分词
        tmp_words = mytool.seg_sentence(target_data, stopwordslist)
        #统计词出现次数
        for word in tmp_words:
            if word in count_words_dic:
                count_words_dic[word] = count_words_dic[word]+1
            else:
                count_words_dic[word] = 0

print(count_words_dic)

        #处理文件
        # 文本分词





# import jieba
