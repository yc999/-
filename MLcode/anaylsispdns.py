# 分析 pdns

from calendar import month
import re
# import eventlet
import os
import sys
import io
import json
import numpy as np
import time, datetime
import time
import datetime

from pyrsistent import T
dnstpye_value = {'1' : "A", '5':"CNAME", '28':"AAAA"}


# 读取dns数据
# dnsdata_path = "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2021-12/pdns_flint_final.8964c40373224b4eb513e1ff965c90ce.1638680400357.log"
# dnsdata_path = "/home/yangc/pdnsdata2/pdns_flint_final.8964c40373224b4eb513e1ff965c90ce.1638680400357.log"
dnsdata_path = "/home/yc/pdns_flint_final.64072b725c544c649c0654a54483cf96.1638770400345.log"
dnsdata_path2 = "/home/yangc/pdnsdata2/pdns_flint_final.8964c40373224b4eb513e1ff965c90ce.1638680400357.log"
# dnsdata_path = "pdns_flint_final.64072b725c544c649c0654a54483cf96.1638770400345.log"
teststr = "RdataUpdate<f:m><btime:1605881538><etime:1638669016><ntime:1638669058><count:1519386><rkey:com.aliyuncs.log.cn-hangzhou.edge2-k8slog+1+b059a7e3e3a6910871e152f678920024><hash:b059a7e3e3a6910871e152f678920024><type:1><name:edge2-k8slog.cn-hangzhou.log.aliyuncs.com><data:114.55.47.138;>"


timeStamp = 1605881538
timeArray = time.localtime(timeStamp)
print(timeArray)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
print(otherStyleTime)   # 2013--10--10 23:40:00\

timeStamp2 = 1638669016
timeArray2 = time.localtime(timeStamp2)
print(timeArray2)
otherStyleTime2 = time.strftime("%Y--%m--%d %H:%M:%S", timeArray2)
print(otherStyleTime2)   # 2013--10--10 23:40:00\

aa = otherStyleTime2 - otherStyleTime
# dnsdata_path = "/home/jiangy2/dnswork/cdnlist/pdns_data"

(timeStamp2-timeStamp).seconds

# import datetime.datetime
def cal_time(stamp1,stamp2):
    t1=time.localtime(stamp1)
    print(t1)
    t2 = time.localtime(stamp2)
    print(t2)
    t1=time.strftime("%Y-%m-%d %H:%M:%S",t1)
    t2 = time.strftime("%Y-%m-%d %H:%M:%S", t2)
    time1=datetime.datetime.strptime(t1,"%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(t2, "%Y-%m-%d %H:%M:%S")
    return (time2-time1).days

cal_time(timeStamp,timeStamp2)




def prasenewdnsdata(data):
    dnsdata = {}
    parts = data.split(">")
    dnsdata['btime'] = parts[1].split(":")[1]
    dnsdata['etime'] = parts[2].split(":")[1]
    dnsdata['ntime'] = parts[3].split(":")[1]
    dnsdata['count'] = parts[4].split(":")[1]
    # tmp = parts[5].split(":")[1]
    # tmp1 = tmp.split("+")
    # dnsdata['rkey'] = tmp1[0]
    dnsdata['Dnstype'] = parts[7].split(":")[1]
    dnsdata['name'] = parts[8].split(":")[1]
    # dnsdata['data'] = parts[9].split(":")[1]
    return dnsdata

a = prasenewdnsdata(teststr)

dnsdata_file = open(dnsdata_path2, 'r', encoding='utf-8')

countdataline = 0
timecount =[]

while True:
    line = dnsdata_file.readline()
    countdataline = countdataline + 1
    if  line:
        try:
            dnsdata = prasenewdnsdata(line)
            # if dnsdata['Dnstype'] in dnstpye_value:
                # print(line)
            timecount.append(dnsdata)
        except:
            continue
    else:
        break


print(len(timecount))  # 100000 十万 五十万

minbtime =1638669019   #1407151101
maxbtime = 0             #1638669118
minetime =1638669019   #1407151101
maxetime = 0             #1638669118

minbtime_data ={}
maxbtime_data ={}
minetime_data ={}
maxetime_data ={}
for tmptime in timecount:
    tmp = int(tmptime['btime'])
    if tmp < minbtime:
        minbtime = tmp
        minbtime_data = tmptime
    if tmp > maxbtime:
        maxbtime = tmp
        maxbtime_data = tmptime
    tmp = int(tmptime['etime'])
    if tmp < minetime:
        minetime = tmp
        minetime_data = tmptime
    if tmp > maxetime:
        maxetime = tmp
        maxetime_data = tmptime

minbtime_data
# {'btime': '1407151101', 'etime': '1638656392', 'ntime': '1638657439', 'count': '41704959'}

maxbtime_data
minetime_data
maxetime_data

# ======================test===========================
timeArray = time.localtime(1638669016)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
print(otherStyleTime)   # 2014--08--04 19:18:21  数据2   2014--08--04 19:18:24 数据1



timeArray = time.localtime(maxtime)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
print(otherStyleTime)   # 2021--12--05 09:51:58   数据2    2021--12--06 11:35:23   数据1


tmptime = 1638665513
timeArray = time.localtime(tmptime)
otherStyleTime = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)

print(otherStyleTime)

tmptime = 1634857621
timeArray = time.localtime(tmptime)
otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)

print(otherStyleTime1)



d1 = datetime.datetime.strptime('2021-12-05 11:35:23', '%Y-%m-%d %H:%M:%S')
d2 = datetime.datetime.strptime('2021-12-05 12', '%Y-%m-%d %H')
tmpd = datetime.timedelta(seconds=1)
delta = d2 - d1 + tmpd

delta.total_seconds()
(d2- d1).total_seconds()

if (d2 - d1).total_seconds()>0:
    print('22222')


(d2 - d1).days
print(delta.days)

now = datetime.datetime.strptime('2021-12-05', '%Y-%m-%d')
delta = datetime.timedelta(days=3)
n_days = d1 + delta
n_days.strftime('%Y-%m-%d')


tmptime = 1634857621
timeArray = time.localtime(tmptime)
dbeg = time.strftime("%Y-%m-%d", timeArray)
dbeg_now = datetime.datetime.strptime(dbeg, '%Y-%m-%d')

for i in range(5):
    delta = datetime.timedelta(days=i)
    n_days = dbeg_now + delta
    a = n_days.strftime('%Y-%m-%d')
    print(a)
    print(type(a))


# =======test end================




#时间间隔统计  ---------------------------------------------------

count_month_delta_dns = {}
for tmptime in timecount:
    tmpcount = int(tmptime['count'])
    #取时间
    tmpfirstday = int(tmptime['btime'])
    timeArray = time.localtime(tmpfirstday)
    dbeg = time.strftime("%Y%m%d", timeArray)
    dbeg_now = datetime.datetime.strptime(dbeg, '%Y%m%d')
    tmpendday = int(tmptime['etime'])
    timeArray = time.localtime(tmpendday)
    dend = time.strftime("%Y%m%d", timeArray)
    dend_now = datetime.datetime.strptime(dend, '%Y%m%d')
    # 日期差
    delta = (dend_now - dbeg_now)
    countday = delta.days + 1
    for i in range(countday):
        delta = datetime.timedelta(days=i)
        n_days = dbeg_now + delta
        tmp_index_day = n_days.strftime('%Y%m')
        if tmp_index_day in count_month_delta_dns:
            count_month_delta_dns[tmp_index_day] = count_month_delta_dns[tmp_index_day] + 1
        else:
            count_month_delta_dns[tmp_index_day] = 1







# win作图
import matplotlib.pyplot as plt
import numpy as np

# win_data_path = "C:/Users/shinelon/Desktop/20211001-62.txt"
# win_data_file = open(win_data_path, 'r', encoding='utf-8')



count_month_delta_dns={'202111': 2403907, '202112': 455718, '202103': 1220168, '202104': 1270490, '202105': 1400877, '202106': 1450978, 
    '202107': 1595815, '202108': 1789373, '202109': 1916585, '202110': 2186901, '201408': 19838, '201409': 31847, '201410': 35993,
    '201411': 36958, '201412': 40491, '201501': 42577, '201502': 39820, '201503': 45678, '201504': 45973, '201505': 49463, 
    '201506': 51395, '201507': 56933, '201508': 59354, '201509': 59211,'201510': 63005, '201511': 62510, '201512': 66844, 
    '201601': 68935, '201602': 66130, '201603': 72836, '201604': 72762, '201605': 78038, '201606': 78034, '201607': 83101, 
    '201608': 85314, '201609': 84235, '201610': 88938,'201611': 88484, '201612': 94207, '201701': 97426, '201702': 90415, 
    '201703': 103942, '201704': 104265, '201705': 111725,
    '201706': 111389, '201707': 118559, '201708': 122438, '201709': 121796, '201710': 130020, '201711': 130801, '201712': 139963,
    '201801': 149790, '201802': 139480, '201803': 157462, '201804': 156166, '201805': 166095, '201806': 166014, '201807': 177603, 
    '201808': 183019, '201809': 184125, '201810': 198610, '201811': 200361, '201812': 215474, '201901': 223021, '201902': 208415, 
    '201903': 240496, '201904': 242431, '201905': 261389, '201906': 263491, '201907': 283329, '201908': 297424, '201909': 301199,
    '201910': 324708, '201911': 331915, '201912': 358884, '202001': 379745, '202002': 368327, '202003': 416009, '202004': 435373,
    '202005': 483728, '202006': 504202, '202007': 570262, '202008': 630075, '202009': 680821, '202010': 769776, '202011': 832787,
    '202012': 923416, '202101': 987196, '202102': 1020623}


list1=[]
x = []
for key in sorted(count_month_delta_dns):
    list1.append(count_month_delta_dns[key])
    x.append(key)

# plt.barh(x,list1)
# plt.yticks(rotation=-15)

plt.bar(x,list1)
plt.xticks(rotation=-90)
plt.show()


len(list1)
list2=[2,3,4,5,8,9,2,1,3,4,5,2,4]

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('显示中文标题')
plt.xlabel("横坐标")
plt.ylabel("纵坐标")
# x=np.arange(0,len(list1))+1
x[0]=1
my_x_ticks = np.arange(1, 89, 1)
plt.xticks(my_x_ticks)
plt.plot(x,list1,label='list1',color='g',linewidth=2,linestyle='--')#添加linestyle设置线条类型
# plt.plot(x,list2,label='list2',color='b',linewidth=5,linestyle='--')
plt.legend()
plt.grid()#添加网格
plt.show()




# -----------------------------------------------------------------------------








# 月访问量统计-----------------------------------------

count_day_dns = {}
for tmptime in timecount:
    tmpcount = int(tmptime['count'])
    #取时间
    tmpfirstday = int(tmptime['btime'])
    timeArray = time.localtime(tmpfirstday)
    dbeg = time.strftime("%Y%m%d", timeArray)
    dbeg_now = datetime.datetime.strptime(dbeg, '%Y%m%d')
    tmpendday = int(tmptime['etime'])
    timeArray = time.localtime(tmpendday)
    dend = time.strftime("%Y%m%d", timeArray)
    dend_now = datetime.datetime.strptime(dend, '%Y%m%d')
    # 日期差
    delta = (dend_now - dbeg_now)
    countday = delta.days + 1
    #计算每日的查询量
    tmp_count_perday = tmpcount/countday
    for i in range(countday):
        delta = datetime.timedelta(days=i)
        n_days = dbeg_now + delta
        tmp_index_day = n_days.strftime('%Y%m%d')
        if tmp_index_day in count_day_dns:
            count_day_dns[tmp_index_day] = count_day_dns[tmp_index_day] + tmp_count_perday
        else:
            count_day_dns[tmp_index_day] = tmp_count_perday

print(len(count_day_dns))

print(sorted(count_day_dns)) 



# 输出 一段时间的流量数据 11月到12月5
find_start_day = '20211101'
find_day_sum = 34
find_start_day = datetime.datetime.strptime(find_start_day, '%Y%m%d')


for i in range(find_day_sum):
    delta = datetime.timedelta(days=i)
    n_days = find_start_day + delta
    tmp_index_day = n_days.strftime('%Y%m%d')
    # print(tmp_index_day)
    print(count_day_dns[tmp_index_day])





# win作图
import matplotlib.pyplot as plt
import numpy as np

win_data_path = "C:/Users/shinelon/Desktop/20211001-62.txt"
win_data_file = open(win_data_path, 'r', encoding='utf-8')


list1=[]

while True:
    line = win_data_file.readline()
    if  line:
        try:
            list1.append(float(line.strip()))
        except:
            continue
    else:
        break

list2=[2,3,4,5,8,9,2,1,3,4,5,2,4]
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.title('显示中文标题')
plt.xlabel("横坐标")
plt.ylabel("纵坐标")
x=np.arange(0,len(list1))+1
x[0]=1
my_x_ticks = np.arange(1, 35, 1)
plt.xticks(my_x_ticks)
plt.plot(x,list1,label='list1',color='g',linewidth=2,linestyle='--')#添加linestyle设置线条类型
# plt.plot(x,list2,label='list2',color='b',linewidth=5,linestyle='--')
plt.legend()
plt.grid()#添加网格
plt.show()
# -------------------------------------------------------------------



# delta = datetime.timedelta(days=3)
# n_days = d1 + delta
# n_days.strftime('%Y-%m-%d')







# 日访问量统计--------------时间粒度 1小时---------------------------

#只统计 20211204 -2021 12 05 9：00的
import math
wanted_time = datetime.datetime.strptime('20211130 03:58:00', '%Y%m%d %H:%M:%S')

count_hour_dns = {}
for tmptime in timecount:
    tmpcount = int(tmptime['count'])
    #取时间
    tmpfirstday = int(tmptime['btime'])
    timeArray = time.localtime(tmpfirstday)
    dbeg = time.strftime("%Y%m%d %H:%M:%S", timeArray)
    dbeg_now = datetime.datetime.strptime(dbeg, '%Y%m%d %H:%M:%S')
    tmpendday = int(tmptime['etime'])
    timeArray = time.localtime(tmpendday)
    dend = time.strftime("%Y%m%d %H:%M:%S", timeArray)
    dend_now = datetime.datetime.strptime(dend, '%Y%m%d %H:%M:%S')
    # 如果不在时间段内不计算
    delta = (wanted_time - dend_now)
    if delta.total_seconds()>0:
        continue
    delta = (dend_now - dbeg_now)
    countseconds = int(delta.total_seconds())+ 1
    #计算每日的查询量
    tmp_count_persecond = tmpcount/countseconds
    # 确定计算开始时间
    if (wanted_time - dbeg_now).total_seconds()>0:
        tmp_wanted_time = wanted_time
    else:
        tmp_wanted_time = dbeg_now
    # print("开始: ",dbeg,"结束: ",dend, "total_seconds: ",delta.total_seconds())
    # print("tmpcount: ", tmpcount, "tmp_count_persecond: ",tmp_count_persecond, "tmp_wanted_time: ",tmp_wanted_time)
    # 加到整小时
    tmphour = datetime.timedelta(hours=1)
    tmp_endt = tmp_wanted_time.strftime('%Y%m%d %H')
    tmp_endt =datetime.datetime.strptime(tmp_endt, '%Y%m%d %H')
    tmp_endt =tmp_endt + tmphour
    while (tmp_wanted_time - tmp_endt).total_seconds()<0 and (tmp_wanted_time - dend_now).total_seconds()<0: #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m%d %H')
        if tmp_index_day in count_hour_dns:
            count_hour_dns[tmp_index_day] = count_hour_dns[tmp_index_day] + tmp_count_persecond 
        else:
            count_hour_dns[tmp_index_day] = tmp_count_persecond
        delta = datetime.timedelta(seconds=1)
        tmp_wanted_time = tmp_wanted_time + delta
    while (tmp_wanted_time + tmphour - dend_now).total_seconds()<0: #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m%d %H')
        if tmp_index_day in count_hour_dns:
            count_hour_dns[tmp_index_day] = count_hour_dns[tmp_index_day] + tmp_count_persecond * 3600
        else:
            count_hour_dns[tmp_index_day] = tmp_count_persecond * 3600
        tmp_wanted_time = tmp_wanted_time + tmphour
    while (tmp_wanted_time - dend_now).total_seconds()<=0: #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m%d %H')
        if tmp_index_day in count_hour_dns:
            count_hour_dns[tmp_index_day] = count_hour_dns[tmp_index_day] + tmp_count_persecond 
        else:
            count_hour_dns[tmp_index_day] = tmp_count_persecond
        delta = datetime.timedelta(seconds=1)
        tmp_wanted_time = tmp_wanted_time + delta



print(len(count_hour_dns))

print(sorted(count_hour_dns)) 


# win作图
import matplotlib.pyplot as plt
import numpy as np
count_hour_dns={'20211130 03': 94144731.49546425, '20211130 04': 2824341945.123711, '20211130 05': 2824341961.889329, '20211130 06': 2824359190.1091537, '20211130 07': 2824368217.6729407, 
    '20211130 08': 2824368227.7055473, '20211130 09': 2824368431.01608, '20211130 10': 2824482107.1268635, '20211130 11': 2824602046.0842004, '20211130 12': 2824604964.629165, 
    '20211130 13': 2824619504.630899, '20211130 14': 2824668739.85583, '20211130 15': 2824728477.6588693, '20211130 16': 2826287978.33078, '20211130 17': 2827119260.703567, 
    '20211130 18': 2828125060.2804036, '20211130 19': 2828201177.8902965, '20211130 20': 2828235933.980317, '20211130 21': 2828260591.3757954, '20211130 22': 2828332915.979931, 
    '20211130 23': 2828347108.332312, '20211201 00': 2830072676.96613, '20211201 01': 2831002297.039649, '20211201 02': 2831005414.4942937, '20211201 03': 2831007957.0692024,
    '20211201 04': 2831022985.2128983, '20211201 05': 2831030833.3906226, '20211201 06': 2831031427.8592286, '20211201 07': 2831106031.22155, '20211201 08': 2831145975.618577, 
    '20211201 09': 2831158941.534171, '20211201 10': 2833702781.5487337, '20211201 11': 2841536667.535101, '20211201 12': 2841595393.796578, '20211201 13': 2841655171.796332, 
    '20211201 14': 2842215749.575602, '20211201 15': 2842914587.864041, '20211201 16': 2843121494.9268813, '20211201 17': 2845829085.1918626, '20211201 18': 2862683044.0683155, 
    '20211201 19': 2867273936.7561197, '20211201 20': 2878912904.680222, '20211201 21': 2881990417.3995085, '20211201 22': 2888874568.4590974, '20211201 23': 2888875125.03317, 
    '20211202 00': 2888876206.519093, '20211202 01': 2888879405.692315, '20211202 02': 2888880491.0578246, '20211202 03': 2888880828.0425897, '20211202 04': 2889318276.159902, 
    '20211202 05': 2889359777.427844, '20211202 06': 2889496739.4483485, '20211202 07': 2889538143.5231843, '20211202 08': 2889546885.7522593, '20211202 09': 2889575736.3120494,
    '20211202 10': 2889580705.578745, '20211202 11': 2889695798.644845, '20211202 12': 2889706625.0513463, '20211202 13': 2889874640.243077, '20211202 14': 2890708620.8117785, 
    '20211202 15': 2893737883.7967443, '20211202 16': 2899586666.8222322, '20211202 17': 2903041945.530811, '20211202 18': 2903661141.4365964, '20211202 19': 2905117294.661305, 
    '20211202 20': 2905475117.730471, '20211202 21': 2905530956.6065164, '20211202 22': 2905538094.669779, '20211202 23': 2905543003.965001, '20211203 00': 2905601944.4284883, 
    '20211203 01': 2905743221.471082, '20211203 02': 2905743698.0953393, '20211203 03': 2905744483.3091273, '20211203 04': 2905745059.697565, '20211203 05': 2905745120.5909767, 
    '20211203 06': 2905746455.3001404, '20211203 07': 2905749157.3603435, '20211203 08': 2905749496.523098, '20211203 09': 2905751589.0148325, '20211203 10': 2905787649.3699646, 
    '20211203 11': 2913738701.0805173, '20211203 12': 2915333691.753634, '20211203 13': 2915358595.097168, '20211203 14': 2915404517.3543353, '20211203 15': 2915466267.4606495, 
    '20211203 16': 2915488061.736579, '20211203 17': 2915613429.630854, '20211203 18': 2915644784.506246, '20211203 19': 2915645167.330964, '20211203 20': 2915645681.9181695, 
    '20211203 21': 2915648184.598221, '20211203 22': 2915650215.674102, '20211203 23': 2915965607.2423244, '20211204 00': 2916260434.1164694, '20211204 01': 2917239462.872923, 
    '20211204 02': 2917402308.0243664, '20211204 03': 2917404830.751266, '20211204 04': 2917405568.3161817, '20211204 05': 2917408299.2421584, '20211204 06': 2918157172.7960353,
    '20211204 07': 2918327494.667078, '20211204 08': 2918812948.460182, '20211204 09': 2919241713.380547, '20211204 10': 2919246960.17252, '20211204 11': 2919249774.477572, 
    '20211204 12': 2919270115.6820984, '20211204 13': 2919280091.8229575, '20211204 14': 2919282338.645334, '20211204 15': 2919299931.1132984, '20211204 16': 2919301602.2278104, 
    '20211204 17': 2919302489.6021743, '20211204 18': 2918853204.435422, '20211204 19': 2903310289.5891304, '20211204 20': 2897936215.597525, '20211204 21': 2897955476.319496, 
    '20211204 22': 2897956197.4500213, '20211204 23': 2897971150.700564, '20211205 00': 2898086518.256091, '20211205 01': 2898182785.5143485, '20211205 02': 2898185965.4374633, 
    '20211205 03': 2898187213.7619357, '20211205 04': 2898194492.504431, '20211205 05': 2890335274.9384556, '20211205 06': 2561632803.051522, '20211205 07': 2106267803.4872677, 
    '20211205 08': 2075733506.7600975, '20211205 09': 710715283.9189745}

list1 = []
sorted(count_hour_dns)
for key in sorted(count_hour_dns):
    list1.append(count_hour_dns[key])

x=np.arange(0,len(count_hour_dns))+1
# x[0]=1
my_x_ticks = np.arange(1, len(count_hour_dns), 1)
plt.xticks(my_x_ticks)
plt.plot(x, list1, label='list1',color='g',linewidth=2,linestyle='--')#添加linestyle设置线条类型
# plt.plot(x,list2,label='list2',color='b',linewidth=5,linestyle='--')
plt.legend()
plt.grid()#添加网格
plt.show()

# ++++++++++++++++++==========================================
















# 统计DNS类型情况====================================================================

# DNS类型统计 
dnstpye_count = {}

for tmptime in timecount:
    tmptype = tmptime['Dnstype']
    if tmptype in dnstpye_count:
        dnstpye_count[tmptype] = dnstpye_count[tmptype] + 1
    else:
        dnstpye_count[tmptype] = 1
    # if tmptype == '15':
    #     print(tmptime)






f = open('/home/yangc/svm_result/spider_result_dic','r')
a = f.read()
dict_class = eval(a)
f.close()





# url = dnsdata['name']
# #取3级域名
# tmpurl = url.split(".")
# if len(tmpurl)>3:
#     tmpurl = tmpurl[-3:]
#     tmpurl = '.'.join(tmpurl)
#     url = tmpurl
# else:
#     tmpurl = url
# tmpurl = url.replace('www.','',1)

dict_class['txtnovels.com'] =dict_class['wwwnovels.com']

# dict_class['wap.txtduo.com'] =dict_class['wapduo.com'] 网站有问题不加入分析


count_url = 0
tmp_dic = []
for tmptime in timecount:
    tmptype = tmptime['Dnstype']
    url = tmptime['name']
    #取3级域名
    tmpurl = url.split(".")
    if len(tmpurl)>3:
        tmpurl = tmpurl[-3:]
        tmpurl = '.'.join(tmpurl)
        url = tmpurl
    else:
        tmpurl = url
    if url in dict_class and url not in tmp_dic:
        count_url = count_url+ 1
        tmp_dic.append(url)
        # add_count
    elif 'www.'+url in dict_class and 'www.'+url not in tmp_dic:
        count_url = count_url+ 1
        tmp_dic.append('www.'+url)
    elif url.replace('www.','',1) in dict_class and url.replace('www.','',1)  not in tmp_dic:
        count_url = count_url+ 1
        tmp_dic.append(url.replace('www.','',1))

count_url


def add_class_count(classindex, classcount_number):
    if classindex in class_count_dic:
        class_count_dic[classindex] =  class_count_dic[classindex] + classcount_number
    else:
        class_count_dic[classindex] = classcount_number


# 统计网站对应类型的数量
count_url = 0
tmp_dic = []

class_count_dic = {}

for tmptime in timecount:
    tmptype = tmptime['Dnstype']
    if tmptype not in dnstpye_value:
        continue
    url = tmptime['name']
    tmpcount = int(tmptime['count'])
    #取3级域名
    tmpurl = url.split(".")
    if len(tmpurl)>3:
        tmpurl = tmpurl[-3:]
        tmpurl = '.'.join(tmpurl)
        url = tmpurl
    else:
        tmpurl = url
    if url in dict_class  :
        count_url = count_url+ 1
        tmp_dic.append(url)
        tmpclass = dict_class[url]
        add_class_count(tmpclass, tmpcount)
    elif 'www.'+url in dict_class  :
        count_url = count_url+ 1
        tmp_dic.append('www.'+url)
        tmpclass = dict_class['www.'+url]
        add_class_count(tmpclass, tmpcount)
    elif url.replace('www.','',1) in dict_class :
        count_url = count_url+ 1
        tmp_dic.append(url.replace('www.','',1))
        tmpclass = dict_class[url.replace('www.','',1)]
        add_class_count(tmpclass, tmpcount)

class_count_dic
count_url

for key in dict_class:
    if key not in tmp_dic:
        print(key)

for tmptime in timecount:
    tmptype = tmptime['Dnstype']
    url = tmptime['name']
    if  '.txt' in url:
        print(tmptime)



# wwwnovels.com
# wapduo.com


dict_class['ftx4.txt99.com']
dict_class['ftx499.com']
dict_class['https.tomav.com']
dict_class['www.httpstomav.com']

dict_class['southboundbride.com']



# 查找所有 类别为 16 的 记录 网站 ===========================================================

count_url = 0

class_16_dic = []

for tmptime in timecount:
    tmptype = tmptime['Dnstype']
    url = tmptime['name']
    if tmptype not in dnstpye_value:
        continue
    tmpcount = int(tmptime['count'])
    #取3级域名
    tmpurl = url.split(".")
    if len(tmpurl)>3:
        tmpurl = tmpurl[-3:]
        tmpurl = '.'.join(tmpurl)
        url = tmpurl
    else:
        tmpurl = url
    if url in dict_class :
        count_url = count_url+ 1
        tmp_dic.append(url)
        tmpclass = dict_class[url]
        # print(tmptime)
        if tmpclass == 16:
            class_16_dic.append(tmptime)
    elif 'www.'+url in dict_class and 'www.'+url:
        count_url = count_url+ 1
        tmp_dic.append('www.'+url)
        tmpclass = dict_class['www.'+url]
        # print(tmptime)
        if tmpclass == 16:
            class_16_dic.append(tmptime)
    elif url.replace('www.','',1) in dict_class :
        count_url = count_url+ 1
        # print(tmptime)
        tmp_dic.append(url.replace('www.','',1))
        tmpclass = dict_class[url.replace('www.','',1)]
        if tmpclass == 16:
            class_16_dic.append(tmptime)




# 查找最早class为16的
minbtime =1638669019   #1407151101
maxbtime = 0             #1638669118
minetime =1638669019   #1407151101
maxetime = 0             #1638669118

minbtime_data ={}
maxbtime_data ={}
minetime_data ={}
maxetime_data ={}
for tmptime in class_16_dic:
    tmp = int(tmptime['btime'])
    if tmp < minbtime:
        minbtime = tmp
        minbtime_data = tmptime
    if tmp > maxbtime:
        maxbtime = tmp
        maxbtime_data = tmptime
    tmp = int(tmptime['etime'])
    if tmp < minetime:
        minetime = tmp
        minetime_data = tmptime
    if tmp > maxetime:
        maxetime = tmp
        maxetime_data = tmptime

minbtime_data
# {'btime': '1407151101', 'etime': '1638656392', 'ntime': '1638657439', 'count': '41704959'}

maxbtime_data
minetime_data
maxetime_data

# minbtime_data
# {'btime': '1407245733', 'etime': '1638667439', 'ntime': '1638667532', 'count': '560805', 'Dnstype': '1', 'name': 'hqgayxxx.com'}
# >>> # {'btime': '1407151101', 'etime': '1638656392', 'ntime': '1638657439', 'count': '41704959'}
# >>> 
# >>> maxbtime_data
# {'btime': '1638665440', 'etime': '1638665440', 'ntime': '1638669078', 'count': '1', 'Dnstype': '1', 'name': '2dx.baby-carriers-downunder.com'}
# >>> minetime_data
# {'btime': '1622816958', 'etime': '1638653680', 'ntime': '1638657369', 'count': '12630', 'Dnstype': '1', 'name': 'v9923.com'}
# >>> maxetime_data
# {'btime': '1634662742', 'etime': '1638669011', 'ntime': '1638669069', 'count': '158719', 'Dnstype': '1', 'name': 'jdav147.xyz'}

timeArray = time.localtime(1407245733)   #最早
otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
timeArray = time.localtime(1638667439)
otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
otherStyleTime1



# 统计每个月 16类的数量
wanted_time = datetime.datetime.strptime('20011130 03:58:00', '%Y%m%d %H:%M:%S')

count_hour_class_16_dic = {}
for tmptime in class_16_dic:
    tmpcount = int(tmptime['count'])
    #取时间
    tmpfirstday = int(tmptime['btime'])
    timeArray = time.localtime(tmpfirstday)
    dbeg = time.strftime("%Y%m%d %H:%M:%S", timeArray)
    dbeg_now = datetime.datetime.strptime(dbeg, '%Y%m%d %H:%M:%S')
    tmpendday = int(tmptime['etime'])
    timeArray = time.localtime(tmpendday)
    dend = time.strftime("%Y%m%d %H:%M:%S", timeArray)
    dend_now = datetime.datetime.strptime(dend, '%Y%m%d %H:%M:%S')
    # 如果不在时间段内不计算
    delta = (wanted_time - dend_now)
    if delta.total_seconds()>0:
        continue
    delta = (dend_now - dbeg_now)
    countseconds = int(delta.total_seconds())+ 1
    #计算每日的查询量
    tmp_count_persecond = tmpcount/countseconds
    # 确定计算开始时间
    if (wanted_time - dbeg_now).total_seconds()>0:
        tmp_wanted_time = wanted_time
    else:
        tmp_wanted_time = dbeg_now
    # print("开始: ",dbeg,"结束: ",dend, "total_seconds: ",delta.total_seconds())
    # print("tmpcount: ", tmpcount, "tmp_count_persecond: ",tmp_count_persecond, "tmp_wanted_time: ",tmp_wanted_time)
    # 加到整小时
    tmpday = datetime.timedelta(days=1)
    tmp_endt = tmp_wanted_time.strftime('%Y%m')
    tmp_endt =datetime.datetime.strptime(tmp_endt, '%Y%m')
    tmp_endt =tmp_endt + tmpday
    while (tmp_wanted_time - tmp_endt).total_seconds()<0 and (tmp_wanted_time - dend_now).total_seconds()<0: #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m%d %H')
        tmp_index_day = tmp_wanted_time.strftime('%Y%m')
        if tmp_index_day in count_hour_class_16_dic:
            count_hour_class_16_dic[tmp_index_day] = count_hour_class_16_dic[tmp_index_day] + tmp_count_persecond 
        else:
            count_hour_class_16_dic[tmp_index_day] = tmp_count_persecond
        delta = datetime.timedelta(seconds=1)
        tmp_wanted_time = tmp_wanted_time + delta
    while (tmp_wanted_time + tmpday - dend_now).total_seconds()<0: #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m%d %H')
        tmp_index_day = tmp_wanted_time.strftime('%Y%m')
        if tmp_index_day in count_hour_class_16_dic:
            count_hour_class_16_dic[tmp_index_day] = count_hour_class_16_dic[tmp_index_day] + tmp_count_persecond * 3600*24
        else:
            count_hour_class_16_dic[tmp_index_day] = tmp_count_persecond * 3600*24
        tmp_wanted_time = tmp_wanted_time + tmpday
    while (tmp_wanted_time - dend_now).total_seconds()<=0: #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m%d %H')
        tmp_index_day = tmp_wanted_time.strftime('%Y%m')
        if tmp_index_day in count_hour_class_16_dic:
            count_hour_class_16_dic[tmp_index_day] = count_hour_class_16_dic[tmp_index_day] + tmp_count_persecond 
        else:
            count_hour_class_16_dic[tmp_index_day] = tmp_count_persecond
        delta = datetime.timedelta(seconds=1)
        tmp_wanted_time = tmp_wanted_time + delta


# 作图
count_hour_class_16_dic={'202101': 272102.6229899167, '202102': 378885.7182513414, '202103': 565991.533065294, '202104': 584199.8256027792, '202105': 754214.9260122243, '202106': 891734.246089814, '202107': 1327239.5862688243, '202108': 1888241.6911781903, '202109': 2064830.2698916423, '202110': 2466784.3101940677, '202111': 3496747.456391092, '202112': 569850.6498268612, '202005': 108889.48771210965, '202006': 117112.58559516008, '202007': 132781.2624229588, '202008': 146450.04948992215, '202009': 144365.83671668364, '202010': 173502.2633568189, '202011': 185616.34516772738, '202012': 206764.45671479078, '201808': 22984.601833365865, '201809': 22292.858045410725, '201810': 23035.95331359107, '201811': 22330.25059237338, '201812': 23088.642811583904, '201901': 23088.642811583904, '201902': 20854.258023366143, '201903': 23119.933363353564, '201904': 39837.59310751059, '201905': 64903.50188445071, '201906': 63803.97557089273, '201907': 65930.77475658912, '201908': 66124.94750948272, '201909': 64114.72457786373, '201910': 66258.60192099593, '201911': 64122.802147216804, '201912': 66260.22888545733, '202001': 66260.22888545733, '202002': 62626.20180997494, '202003': 94539.20791842078, '202004': 103539.78178937768, '201410': 14846.674834291794, '201411': 14389.38320568229, '201412': 14869.029312538361, '201501': 14869.029312538361, '201502': 13430.090991970137, '201503': 14869.029312538361, '201504': 14389.38320568229, '201505': 14908.910056836494, '201506': 14452.3528019425, '201507': 14944.44676516544, '201508': 14946.920413824053, '201509': 14466.023445819857, '201510': 14950.831854393427, '201511': 14468.546955864615, '201512': 14950.831854393427, '201601': 14950.831854393427, '201602': 13986.262057335794, '201603': 14950.831854393427, '201604': 14468.546955864615, '201605': 14953.339290997239, '201606': 14493.621321902729, '201607': 14976.742032632812, '201608': 14976.742032632812, '201609': 14493.621321902729, '201610': 14976.742032632812, '201611': 14493.621321902729, '201612': 14976.742032632812, '201701': 14976.742032632812, '201702': 13527.37990044255, '201703': 14976.742032632812, '201704': 16615.923645636016, '201705': 21532.995410379743, '201706': 20838.382655206227, '201707': 21591.694161635423, '201708': 21760.453071495507, '201709': 21058.502972415026, '201710': 21760.453071495507, '201711': 21871.419082252938, '201712': 22960.472090780044, '201801': 22960.472090780044, '201802': 20738.490920704593, '201803': 22964.900587967208, '201804': 22228.668695129225, '201805': 22969.624318300182, '201806': 22228.668695129225, '201807': 22969.624318300182, '201408': 10454.010502776664, '201409': 14306.416808158085}
list1 = []
sorted(count_hour_class_16_dic)
for key in sorted(count_hour_class_16_dic):
    list1.append(count_hour_class_16_dic[key])

x=np.arange(0,len(count_hour_class_16_dic))+1
# x[0]=1
my_x_ticks = np.arange(1, len(count_hour_class_16_dic), 1)
plt.xticks(my_x_ticks)
plt.plot(x, list1, label='list1',color='g',linewidth=2,linestyle='--')#添加linestyle设置线条类型
# plt.plot(x,list2,label='list2',color='b',linewidth=5,linestyle='--')
plt.legend()
plt.grid()#添加网格
plt.show()


#统计每月的 16 类数量
import dateutil
from dateutil.relativedelta import relativedelta

count_month_class_16_dic = {}
for tmptime in class_16_dic:
    tmpcount = int(tmptime['count'])
    #取时间
    tmpfirstday = int(tmptime['btime'])
    timeArray = time.localtime(tmpfirstday)
    dbeg = time.strftime("%Y%m", timeArray)
    dbeg_now = datetime.datetime.strptime(dbeg, '%Y%m')
    tmpendday = int(tmptime['etime'])
    timeArray = time.localtime(tmpendday)
    dend = time.strftime("%Y%m%d %H:%M:%S", timeArray)
    dend_now = datetime.datetime.strptime(dend, '%Y%m%d %H:%M:%S')
    # 如果不在时间段内不计算
    tmp_wanted_time = dbeg_now
    while (tmp_wanted_time - dend_now).total_seconds()<0 : #当前计算时间小于记录结束时间
        tmp_index_day = tmp_wanted_time.strftime('%Y%m')
        if tmp_index_day in count_month_class_16_dic:
            count_month_class_16_dic[tmp_index_day].append(tmptime['name']) 
        else:
            count_month_class_16_dic[tmp_index_day] = [tmptime['name']]
        # delta = datetime.timedelta(month=1)
        # tmp_wanted_time = tmp_wanted_time + delta
        tmp_wanted_time = tmp_wanted_time + relativedelta(months=1)


count_month_class_16_number_dic={}
for key in sorted(count_month_class_16_dic):
    count_len = len(set(count_month_class_16_dic[key]))
    count_month_class_16_number_dic[key]=count_len


# 统计每个第16 类网站出现的次数
count_16_url = {}
for tmptime in class_16_dic:
    tmpurl = tmptime['name']
    tmpcount = int(tmptime['count'])
    if tmpurl in count_16_url:
        count_16_url[tmpurl] = count_16_url[tmpurl] + tmpcount
    else:
        count_16_url[tmpurl] = tmpcount



max_16_count = 0
max_16_key =""
for key in count_16_url:
    if count_16_url[key] >max_16_count:
        max_16_count = count_16_url[key]
        max_16_key = key

count_16_url[max_16_key]

# max_16_key
# 'www.fanbus.bar'
# >>> count_16_url[max_16_key]
# 1474921

for tmptime in class_16_dic:
    if tmptime['name'] ==max_16_key:
        print(tmptime)

{'btime': '1625864928', 'etime': '1638657475', 'ntime': '1638657541', 'count': '1474921', 'Dnstype': '28', 'name': 'www.fanbus.bar'}

timeArray = time.localtime(1625864928)   #最早
otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
otherStyleTime1

timeArray = time.localtime(1638657475)
otherStyleTime1 = time.strftime("%Y--%m--%d %H:%M:%S", timeArray)
otherStyleTime1





# 统计所有域名的数量

url_count_dic = {}
for tmptime in timecount:
    tmptype = tmptime['Dnstype']
    tmpurl = tmptime['name']
    if tmptype not in dnstpye_value:
        continue
    tmpcount = int(tmptime['count'])
    #取3级域名
    if tmpurl in url_count_dic:
        url_count_dic[tmpurl] = url_count_dic[tmpurl] + tmpcount
    else:
        url_count_dic[tmpurl] =  tmpcount



max_url_count = 0
max_url_key =""
for key in url_count_dic:
    if url_count_dic[key] >max_url_count:
        max_url_count = url_count_dic[key]
        max_url_key = key

url_count_dic[max_url_key]

taobao = "taobao"

for key in url_count_dic:
    if taobao in key:
        print(key)


# fourier.taobao.com

# taobaohuayua.com


# vod-yq-aliyun.taobao.com.w.cdngslb.com
# h5ltao.m.taobao.com.w.alikunlun.com
# live-segment.video.taobao.com.w.alikunlun.com
# 1688live-bfrtc.taobao.com.w.cdngslb.com
# miniapp-package-zcache.taobao.com.w.cdngslb.com
# servicehall.cdn.taobao.com.w.alikunlun.com
# yqrxrc292luslw.taobao945decdn.com
# ald.taobao.com.danuoyi.tbcache.com
# sgcdn.litetao.taobao.com.w.alikunlun.com
# qh-material.taobao.com.w.alikunlun.com
# na61-na62.wagbridge.alibaba.taobao.com.gds.alibabadns.com
# cdn.litetao.taobao.com.w.alikunlun.com
# h5.m.taobao.com.danuoyi.tbcache.com
# fourier.taobao.com
# cdn.npm.taobao.org.w.kunlunar.com
# taobaohuayua.com
# trade-acs.m.taobao.com.host
# na610-na620.acs.m.taobao.com.gds.alibabadns.com
# detail.m.taobao.com.w.alikunlun.com
# yum-taobao.alibaba-inc.com.w.cdngslb.com
# show.chuangyi.taobao.com.w.alikunlun.com
# shop239817955.taobao.com
# shop240540234.taobao.com
# shop150488899.taobao.com
# taobaoboyulecheng.r6.nongji360.com
# push.qintao.taobao.com.gds.alibabadns.com
# shop58614984.m.taobao.com
# urlcheck.browser.taobao.com
# shop241030087.taobao.com
# lu9afjq646.api.taobao.com
# carry.taobao.com.a.lahuashanbx.com
# rhdc-acs.m.taobao.com.gds.alibabadns.com
# liveng-rtclive.taobao.com.w.cdngslb.com
# ltao.seller.taobao.com
# evo.m.taobao.com.w.cdngslb.com
# buy.taobao.com.danuoyi.tbcache.com
# seller.vip.taobao.com
# shop241160841.taobao.com
# shop409769678.m.taobao.com
# zb-center-openjmacs.m.taobao.com.gds.alibabadns.com
# acs-lazada-sg.m.taobao.com.gds.alibabadns.com
# secgw-ipv6-aserver-heyi.m.taobao.com
# h5.taobao.com.w.cdngslb.com
# shop150784207.m.taobao.com
# scene-ossgw.taobao.com.w.alikunlun.com
# shop241528161.taobao.com
# tbmsg.cloud.video.taobao.com.w.cdngslb.com
# outfliggys.m.taobao.com.w.alikunlun.com
# ugcdn.taobao.com.w.alikunlun.com
# buyertrade.taobao.com.danuoyi.tbcache.com
# 3000000004151392.hybrid.miniapp.taobao.com
# openacs4uc.m.taobao.com.domain.name
# item.taobao.com.danuoyi.tbcache.com
# aserver-heyi-wasu.m.taobao.com.gds.alibabadns.com
# e6q0tufyk1.api.taobao.com
# msgacs.m.taobao.com.host
# shop241367998.taobao.com
# 84ywk.taobaodajie.com
# amdc.m.taobao.com.domain.name

dict_class['taobaohuayua.com']
dict_class['shop241367998.taobao.com']
dict_class['shop239817955.taobao.com']
dict_class['fourier.taobao.com']
dict_class['taobaohuayua.com']


sorted_url_count_dic = sorted(url_count_dic.items(), key = lambda kv:(kv[1], kv[0]),reverse=True)


for i in range(15):
    print(sorted_url_count_dic[i])

# ('www.jd.com', 27268566803767)
# ('www.360.com', 7647028817601)
# ('cn.pool.ntp.org', 4814419553576)
# ('www.a.shifen.com', 1266536971320)
# ('img2x-sched.jcloud-cdn.com', 1067215784501)
# ('tracker.sdk.00cdn.com', 884536675320)
# ('m.baidu.com', 771056388736)
# ('hcdnl.pulldyin.gslb.c.cdnhwc2.com', 646866334625)
# ('tracker.dcdn.baidu.com', 503985902649)
# ('pool.ntp.org', 474646254146)
# ('th.pinduoduo.com', 349317344998)
# ('vipxyajs-data.p2cdn.com', 329184738087)
# ('cloud.jdcdn.com', 324372896056)
# ('txdwkmov.a.yximgs.com.cdn.dnsv1.com', 293553980517)
# ('conf-darwin.xycdn.com', 268401915944)


dict_class['m.baidu.com']

url_count_dic['shop241367998.taobao.com']
url_count_dic['shop150784207.m.taobao.com']



