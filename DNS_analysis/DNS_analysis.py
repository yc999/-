

def prasednsdata(data):
    dnsdata = {}
    parts = data.split(" ")
    dnsdata['tnow'] = parts[0].split(":")[1]
    dnsdata['tbeg'] = parts[1].split(":")[1]
    dnsdata['tend'] = parts[2].split(":")[1]
    dnsdata['count'] = parts[3].split(":")[1]
    tmp = parts[4].split(":")[1]
    tmp1 = tmp.split("+")

    dnsdata['rkey'] = tmp1[0]
    dnsdata['Dnstype'] = tmp1[1]
    dnsdata['data'] = parts[5].split(":")[1]
    return dnsdata



dnstpye_value = {1 : "A", 2:"NS",3:"MD",5:"CNAME",6:"SOA",12:"PTR",28:"AAAA"}
# 读取dns数据
dnsdata_path = "E:/wechatfile/WeChat Files/wxid_luhve56t0o4a11/FileStorage/File/2020-11/pdns_data"
dnsdata_file = open(dnsdata_path, 'r', encoding='utf-8')
while True:
    line = dnsdata_file.readline()
    if  line:
        parts = prasednsdata(line)
        print(line)
    else:
        break
# line = dnsdata_file.readline()
print(line)
