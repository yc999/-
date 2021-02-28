#-- coding: utf-8 --
# 找到爬过的非目标网页
import sys
# print(sys.path)
import os
# print(os.path.realpath('./MLcode/'))
sys.path.append(os.path.realpath('./MLcode'))
sys.path.append(os.path.realpath('../MLcode'))
import mytool

filepath = '/home/jiangy2/webdata'
# fs = os.listdir(filepath)   #读取url目录

wrongwordslist = [ '葡京', '奥门', '投注', '色情']
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
            # print(os.path.join(root, d))

count_words_dic = {}

# 读取停用词
stopwordslist = []
stopwordslist_path = '/home/jiangy2/dnswork/stopwords/cn_stopwords.txt'
stopwordslist =mytool.read_stopwords(stopwordslist_path)

# 读取文件
firststep_list = []
for dirpath in dirlist:
    print(dirpath)
    # for f in files:
    for root, dirs, files in os.walk(dirpath):
        for f in files:
            data = mytool.read_webdata(os.path.join(root, f))
            # print(os.path.join(root, f))
            # 网页数据存入一个list
            target_data = mytool.get_all_webdata(data)
            #分词
            tmp_words = mytool.seg_sentence(target_data, stopwordslist)
            #统计词出现次数
            for word in tmp_words:
                if word in wrongwordslist:
                    # print(word)
                    # print(os.path.join(root, f))
                    firststep_list.append(f)
                    break
            # if word in count_words_dic:
            #     count_words_dic[word] = count_words_dic[word]+1
            # else:
            #     count_words_dic[word] = 0
print(firststep_list)
filename = '../../firststep_list.txt'
f = open(filename,'w', encoding='utf-8')
for url in firststep_list:
        f.write(url.replace(".txt","") + '\n')
f.close()
# print(count_words_dic)
# savefilepath = "./firststep.txt"
# f = open(savefilepath, "w", encoding="utf-8")
# f.write(json.dumps(webinfo, ensure_ascii=False))
# f.close()#关闭文件