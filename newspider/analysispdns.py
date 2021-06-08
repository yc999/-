#-- coding: utf-8 --
# 找到 dns 解析路径 比如 首先访问 baidu 返回的是 CNAME 记录， 那么会解析CNAME对应的域名
#  cname -> A,AAAA
#  A,AAAA
import json
import numpy as np
import  requests
import re
# import eventlet
import os
import sys
import io
sys.path.append(os.path.realpath('./Clustering'))
sys.path.append(os.path.realpath('../Clustering'))
sys.path.append(os.path.realpath('./spider'))
sys.path.append(os.path.realpath('../spider'))
import random
import mytool


dnstpye_value = {'1' : "A", '5':"CNAME", '28':"AAAA"}

# 读取dns数据
# dnsdata_path = "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2020-11/pdns_data"
dnsdata_path = "/home/jiangy2/dnswork/cdnlist/pdns_data"
dnsdata_file = open(dnsdata_path, 'r', encoding='utf-8')

while True:
    line = dnsdata_file.readline()
    if  line:
        try:
            dnsdata = mytool.prasednsdata(line)
        except:
            continue