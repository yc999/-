# 使用svm 得到 网站分类情况


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
import pickle
import numpy as np
import os


from bs4 import  BeautifulSoup, Comment
import re
import jieba
import jieba.posseg as pseg
import json
import numpy as np
# 加载tfidf

feature_path = '/home/yangc/lstm_model/vectorizer.pkl'
tfidftransformer_path = '/home/yangc/lstm_model/transform.pkl'



vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open(feature_path, "rb")))
# 加载TfidfTransformer
# tfidftransformer_path = 'models/tfidftransformer.pkl'
transform = pickle.load(open(tfidftransformer_path, "rb"))
#测试用transform，表示测试数据，为list





struct_feature_path = '/home/yangc/lstm_model/struct_vectorizer.pkl'
struct_tfidftransformer_path = '/home/yangc/lstm_model/struct_transform.pkl'

struct_vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b", decode_error="replace", vocabulary=pickle.load(open(struct_feature_path, "rb")))
struct_transform = pickle.load(open(struct_tfidftransformer_path, "rb"))



badtitles = ['404 Not Found', '找不到',  'null', 'Not Found','阻断页','Bad Request','Time-out','No configuration',
'TestPage','IIS7','Default','已暂停' ,'Server Error','403 Forbidden','禁止访问','载入出错','没有找到',
'无法显示','无法访问','Bad Gateway','正在维护','配置未生效','访问报错','Welcome to nginx','Suspended Domain',
'IIS Windows','Invalid URL','服务器错误','400 Unknown Virtual Host','无法找到','资源不存在',
'Temporarily Unavailable','Database Error','temporarily unavailable','Bad gateway','不再可用','error Page',
'Internal Server Error','升级维护中','Service Unavailable','站点不存在','405','Access forbidden','System Error',
'详细错误','页面载入出错','Error','错误','Connection timed out','域名停靠','网站访问报错','错误提示','临时域名',
'未被授权查看','Test Page','发生错误','非法阻断','链接超时','403 Frobidden','建设中','访问出错','出错啦','ACCESS DENIED','系统发生错误','Problem loading page']


word2vec_path = '/public/ycdswork/dnswork/glove/Tencent_AILab_ChineseEmbedding.txt'
stopwords_path = "/public/ycdswork/dnswork/stopwords/cn_stopwords.txt"

file_dir = "/home/yangc/myclass/"
modelsave_path = "/public/ycdswork/modeldir/LSTMmodel"

# 2.1 读取停用词
# stopwordslist 保存所有停用词
def read_stopwords(filepath):
    stopwords = []
    if os.path.exists(filepath):
        stopwords = [line.strip() for line in open(filepath, 'r', encoding='utf-8').readlines()]
    return stopwords

stopwordslist = []
stopwordslist = read_stopwords(stopwords_path)


def get_head(soup):
    head = soup
    webinfo = {}
    webinfo['title'] = ""
    webinfo['description'] = ""
    webinfo['keywords'] = ""
    if webinfo['title'] == "" or webinfo['title'] == None:
        try:
            webinfo['title'] += head.title.string.strip()
        except:
            pass
    try:
        webinfo['description'] += head.find('meta',attrs={'name':'description'})['content']
    except:
        pass
    try:
        webinfo['description'] += head.find('meta',attrs={'name':'Description'})['content']
    except:
        pass
    try:
        webinfo['description'] += head.find('meta',attrs={'name':'DESCRIPTION'})['content']
    except:
        pass
    try:
        webinfo['keywords'] += head.find('meta',attrs={'name':'keywords'})['content']
    except:
        pass
    try:
        webinfo['keywords'] += head.find('meta',attrs={'name':'Keywords'})['content']
    except:
        pass
    try:
        webinfo['keywords'] += head.find('meta',attrs={'name':'KEYWORDS'})['content']
    except:
        pass
    if ifbadtitle(webinfo["title"]):
            return False
    result_text = ""
    for text in webinfo:
        result_text += webinfo[text]
    return result_text



def ifbadtitle(mytitle):
    for badtitle in badtitles:
        if badtitle in mytitle:
            return True
    return False


def filtertext(htmldata):
    """
    # 输入 html文档
    # 返回html中所有的文本 string
    # 如果网页标题包含不正常文本 返回False
    """
    soup = BeautifulSoup(htmldata,'html.parser')
    head_text = get_head(soup)
    if head_text == False:
        return False
    [s.extract() for s in soup('script')]
    [s.extract() for s in soup('style')]
    for element in soup(text = lambda text: isinstance(text, Comment)):
        element.extract()
    body = soup.get_text()
    # body = ''.join(body)
    return head_text + body



def Word_pseg(self,word_str):  # 名词提取函数
        words = pseg.cut(word_str)
        word_list = []
        for wds in words:
            # 筛选自定义词典中的词，和各类名词，自定义词库的词在没设置词性的情况下默认为x词性，即词的flag词性为x
            if wds.flag == 'x' and wds.word != ' ' and wds.word != 'ns' \
                    or re.match(r'^n', wds.flag) != None \
                            and re.match(r'^nr', wds.flag) == None:
                word_list.append(wds.word)
        return word_list



def Word_cut_list(word_str):
    #利用正则表达式去掉一些一些标点符号之类的符号。
    word_str = re.sub("[\s+\.\!\[\]\/_,\>\<\-$%^*¿(+\"\']+|[+——；:！·，”。【】《》～¿лΣùòЦ±д£ȫ²αμв»©½йÿλ，。：“？、~@#￥%……&*（）0123456789①②③④)]+".encode('utf-8').decode("utf8"), " ".encode('utf-8').decode("utf8"), word_str)
    wordlist = list(jieba.cut(word_str))  #jieba.cut  把字符串切割成词并添加至一个列表
    # print(wordlist)
    wordlist_N = []
    # chinese_stopwords=self.Chinese_Stopwords()
    for word in wordlist:
        if word not in stopwordslist: #词语的清洗：去停用词
            if word != '\t':
                # if word in embeddings_index: # 词向量中有该词
                    wordlist_N.append(word)
    return wordlist_N


def segmentWord(cont):
    listseg=[]
    for i in cont:
        Wordp = Word_pseg(i)
        New_str = ''.join(Wordp)
        Wordlist = Word_cut_list(New_str)
        file_string = ''.join(Wordlist)
        listseg.append(file_string)
    return listseg



MAX_SEQUENCE_LENGTH = 0
# max_sequence_lenth = 0

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



# 读取html文件,并处理成词语
# 其他页面中的词语 如果没有在主页中没出现  才放入列表中, 少于10个词的不考虑
def readtrain(filepath):
    global MAX_SEQUENCE_LENGTH 
    # webdatadic = mytool.read_webdata(filepath)
    try:
        webdatadic = read_webdata(filepath)
    except Exception as r:
        print(r)
        return []
    result_list = []
    result_dic = {}
    webkey = list(webdatadic.keys())  #key 是每个页面的url 
    if len(webkey)>0:
        for htmldata in webkey: #遍历每个页面，切词处理
            htmltext = filtertext(webdatadic[htmldata]) # 读取网页中的纯文本
            if htmltext == False:
                result_dic[htmldata] = []
                continue
            cut_text = Word_cut_list(htmltext) # 生成词列表
            # print("2")
            result_dic[htmldata] = cut_text
        result_list += result_dic[webkey[0]]  # 先保存首页
        # print(result_dic)
        for htmldata in webkey[1:]:
            for word in result_dic[htmldata]:
                # if word not in result_dic[webkey[0]]:  # 只保存不在首页中出现的词语
                    result_list.append(word)
        if len(result_list) < 10:
            return []
        tmpcount = 0
        len_result_list = list(dict.fromkeys(result_list))
        for word in len_result_list:
            # if word in embeddings_index:
                tmpcount += 1
        if tmpcount> MAX_SEQUENCE_LENGTH:
            MAX_SEQUENCE_LENGTH = tmpcount
    return result_list


"""
    返回空格分隔的词语串
"""
def read_all_data(datapath):
    data = readtrain(datapath)
    if data == []:
        return ""
    cut_text = ' '.join(data)
    # result_data.append(cut_text)
    return cut_text



taglist = ["a","div","li","span","img","td","p","ul","option","meta","tr","link","input","table",
"tbody","dd","h2","h3","hr","dt"]
tag_dic = {}  # tag对应下标的字典
TAG_length = len(taglist)

for i in range(TAG_length):
    tmpkey = taglist[i]
    tag_dic[tmpkey] = i+1

"""
    返回网页结构的特征
    特征为网页对应标签的下标
    
"""

def getpage_struct(datapath):
    result_list = []
    try:
        webdatadic = read_webdata(datapath)
    except Exception as r:
        print(r)
        return []
    webkey = list(webdatadic.keys())  # key 是每个页面的url 
    if len(webkey)>0:
        for htmldata in webkey:
            tmpdata = webdatadic[htmldata]
            soup = BeautifulSoup(tmpdata,'html.parser')
            body = soup.find("body")
            input_list = []
            if body is not None:
                for child in body.descendants:
                    if child.name in taglist:
                        # input_list.append(child.name)
                        input_list.append(tag_dic[child.name])
                result_list.append(input_list)
            else:
                result_list.append(input_list)
    return result_list
#2.3 读取爬取的网页信息
#读取保存的网页信息



webfilepath = "/home/yangc/pdnsdata2/pdnsdata/"
# /home/yangc/pdnsdata2/pdnsdata
# 保存爬到的数据
i=0
j=0

webfilecontent = {}     # 内容
webfilestruct = {}      # 结构

fs = os.listdir(webfilepath)

#读取所有爬到的网站内容 存到map中
for filename in fs:
    i = i +1
    tmpfilename = filename.replace(".txt","")
    # webdata = mytool.read_webdata(os.path.join(filepath, filename))
    webdatapath = os.path.join(webfilepath, filename)
    # print(webdatapath)
    webdata = read_all_data(webdatapath)
    webstructdata = getpage_struct(webdatapath)
    # print(webdata)
    if tmpfilename not in webfilecontent:
        webfilecontent[tmpfilename] = webdata
        webfilestruct[tmpfilename] = webstructdata
    else:
        print(tmpfilename)

# i = 39422



def unicode_to_cn(in_str, debug=False):
    out = None
    if isinstance(in_str, bytes):
        temp = str(in_str, encoding='utf-8')
        out = temp.encode('utf-8').decode('unicode_escape')
    else:
        out = in_str.encode('utf-8').decode('unicode_escape')
    return out

# 设置类别的下标
class_index ={}
indexcount = 0


for file in os.listdir(file_dir):
    file = file.replace(".txt",'')
    file_path = os.path.join(file_dir, file)
    if not os.path.isdir(file_path):
        tmp = file.split('.')[0]
        # if len(sys.argv)>1:
        tmp = tmp.replace("#U","\\u")
        print(tmp)
        tmp = unicode_to_cn(tmp)
        print(tmp)
        class_index[tmp] = indexcount
        indexcount = indexcount + 1

print(class_index)




#  从webfilecontent中拿到对应的分类好的文件
content_train_src = []      # 训练集文本列表
struct_train_src = []      # 训练集结构列表
opinion_train_stc = []      # 训练集类别列表
filename_train_src = []     # 训练集对应的域名

                
for url in  webfilecontent:
    if len(webfilecontent[url])<26:
        print(url)
    else:
        content_train_src.append(webfilecontent[url])               # 加入数据集 字符串
        struct_train_src.append(webfilestruct[url])               # 加入数据集 字符串
        filename_train_src.append(url)
                   

print("已爬取网页数：") #39422
print(i)
print("有效网页数：")
print(len(content_train_src))   #10901
print(len(struct_train_src))
print(len(opinion_train_stc))
print(len(filename_train_src))



X_train_vec = vectorizer.transform(content_train_src)
svm_x_train_input = transform.transform(X_train_vec)


# 加载svm


svm_model_savepath = "/home/yangc/lstm_model/svm_model"

with open(svm_model_savepath,'rb') as f:  
    clf = pickle.load(f)  #将模型存储在变量clf_load中  
    # print(clf_load.predict(X[0:1000])) #调用模型并预测结果

svm_struct_model_savepath = "/home/yangc/lstm_model/svm_struct_model"

with open(svm_struct_model_savepath,'rb') as f:  
    clf_struct = pickle.load(f)  #将模型存储在变量clf_load中  
    # print(clf_load.predict(X[0:1000])) #调用模型并预测结果

# 预测
pred_y  = clf.predict(svm_x_train_input)
pred_y_proba  = clf.predict_proba(svm_x_train_input)


for i in pred_y:
    print(i)



struct_list_input = []
for i in range(len(struct_train_src)):
    tmp = []
    for j in range(len(struct_train_src[i])):
        for k in range(len(struct_train_src[i][j])):
            tmp.append(struct_train_src[i][j][k])
    struct_list_input.append(tmp)





# print(shape(svm_x_train_input)) #(457, 62919)(9356, 56909)
# print(shape(lstm_input_struct_train)) #(457, 62919)

# 稀疏矩阵转稠密矩阵
svm_x_train_input_dense = svm_x_train_input.todense()


svm_x_train_input_array = svm_x_train_input_dense.getA()

# (10901, 56909) 数据2




tmplist = ""

for word in struct_list_input[0]:
    tmplist = tmplist + " " + str(word)
    print(tmplist)





struct_list_input_list_word = []
for words in struct_list_input:
    tmplist = ""
    for word in words:
        tmplist = tmplist + " " + str(word)
    struct_list_input_list_word.append(tmplist)


X_train_vec_struct = struct_vectorizer.transform(struct_list_input_list_word)
svm_x_train_struct = struct_transform.transform(X_train_vec_struct)



# 结构特征tfidf矩阵转为list
svm_x_train_struct_dense = svm_x_train_struct.todense()

svm_x_train_struct_dense_array = svm_x_train_struct_dense.getA()

svm_x_train_input_struct = []

for i in range(len(svm_x_train_input_dense)):
    tmpinput = []
    for tmpnum in svm_x_train_input_array[i]:
        tmpinput.append(tmpnum)
    for tmpnum in svm_x_train_struct_dense_array[i]:
        tmpinput.append(tmpnum)
    svm_x_train_input_struct.append(tmpinput)





import scipy.sparse.csr
#转为稀疏矩阵
svm_x_train_input_struct_csr = scipy.sparse.csr.csr_matrix(svm_x_train_input_struct)




pred_y_struct  = clf_struct.predict(svm_x_train_input_struct_csr)
pred_y_proba_struct  = clf_struct.predict_proba(svm_x_train_input_struct_csr)

pred_y_proba_struct[0][3]


#保存
pred_y_struct_savepath = "/home/yangc/svm_result/pred_y_struct.npy"
pred_y_proba_struct_savepath = "/home/yangc/svm_result/pred_y_proba_struct.npy"
np.save(pred_y_struct_savepath, pred_y_struct)
np.save( pred_y_proba_struct_savepath, pred_y_proba_struct)


# pred_y
# pred_y_proba

pred_y_savepath = "/home/yangc/svm_result/pred_y.npy"
pred_y_proba_savepath = "/home/yangc/svm_result/pred_y_proba.npy"
np.save( pred_y_savepath, pred_y)
np.save( pred_y_proba_savepath, pred_y_proba)


# 加载

# data_b=np.load(pred_y_proba_struct_savepath)

from gensim.models import KeyedVectors




EMBEDDING_DIM = 200  #词向量长度
EMBEDDING_length = 8824330



tc_wv_model = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
# EMBEDDING_length = 8824330
EMBEDDING_length = len(tc_wv_model.key_to_index)
print('Found %s word vectors.' % EMBEDDING_length)

embeddings_index = {}
embedding_matrix = np.zeros((EMBEDDING_length + 1, EMBEDDING_DIM))
# tc_wv_model.key_to_index
# for counter, key in enumerate(tc_wv_model.vocab.keys()):
for counter, key in enumerate(tc_wv_model.key_to_index):
    # print(counter,key)
    embeddings_index[key] = counter+1
    coefs = np.asarray(tc_wv_model[key], dtype='float32')
    embedding_matrix[counter+1] = coefs


del tc_wv_model





# 把字符串按空格切分
def words2index(words):
    index_list = []
    word_list = []
    tmp_words = words.split(" ")
    for word in tmp_words:
        if word in embeddings_index.keys():  # test if word in embeddings_index
            index_list.append(embeddings_index[word])
            word_list.append(word)
    return index_list, word_list








# 加载模型

#测试 加载权重  单特征双向lstm
mybilstmmodel_test = my_bilstm_model(200,100)
model_savepath = "/public/ycdswork/lstm_model/mybilstmmodel_16/mybilstmmodel_16"
mybilstmmodel_test.load_weights(model_savepath)



mybilstmmodel_test_pred = mybilstmmodel_test.predict(spider_X_train_full)


mybilstmmodel_test_pred_class = np.argmax(mybilstmmodel_test_pred, axis=1)
list(mybilstmmodel_test_pred_class)
