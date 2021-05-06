path = "/Users/yangchen/Desktop/ClassDomains.txt"
readfile = open(path,'r', encoding='utf-8')

path = "/Users/yangchen/Desktop/ClassDomains1.txt"
writefile = open(path,'w', encoding='utf-8')

urldic = {}
while True:
    line = readfile.readline()
    if  line:
        parts = line.split(',')
        url = parts[1]
        parts[1] = parts[0]
        parts[0] = url
        tmpline = ','.join(parts[0:3])
        tmpline = tmpline.replace('\n','')
        tmpline = tmpline + '\n'
        if parts[1] not in urldic:
            urldic[parts[1]] = [tmpline]
        else:
            urldic[parts[1]].append(tmpline)
    else:
        break

count = 0 
for urlclass in urldic:
    print(urlclass)
    count = count + 1
    for line in urldic[urlclass]:
        # print(line)
        writefile.write(line)

writefile.close()
print(count)
