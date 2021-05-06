
cctld = ['au',
     'ca' ,
     'cn' ,
     'de' ,
     'eu',
    'uk', #英国
    'fr', # 法国
    'jp', # 日本
    'us', # 美国
    'ru', # 俄罗斯
    'kr', #韩国
    'nl', #荷兰
    'it' , #新加坡
    'sg'  #意大利
    ]

cctld = []

path = "/Users/yangchen/Desktop/TotalGtld.txt"
readfile = open(path,'r', encoding='utf-8')
while True:
    line = readfile.readline()
    if line:
        cctld.append(line.replace('\n',''))
    else:
        break




max_cctld = 5
count = 0
maxcount = max_cctld * len(cctld)

urldic = {}
filename = "/Users/yangchen/Desktop/aleax.txt"
urlfile = open(filename,'r', encoding='utf-8')

while True:
    line = urlfile.readline()
    if  line:
        parts = line.split('.')
        lastname = parts[-1].replace('\n','')
        if  lastname in cctld:
            # print(lastname)
            if lastname not in urldic:
                urldic[lastname] = [line]
            elif len(urldic[lastname]) < max_cctld:
                urldic[lastname].append(line)
                count = count + 1
                if count == maxcount:
                    break
    else:
        break


print(urldic)
for i in urldic:
    print(i)
    print(len(urldic[i]))


                
