gtlddic = {}



filename = "/Users/yangchen/Desktop/aleax.txt"
urlfile = open(filename,'r', encoding='utf-8')

while True:
    line = urlfile.readline()
    if  line:
        parts = line.split('.')
        lastname = parts[-1].replace('\n','')
        if  lastname not in gtlddic:
            gtlddic[lastname] = 1
        else:
            gtlddic[lastname] = gtlddic[lastname] + 1
    else:
        break


for gtld in gtlddic:
    line = ''
    line = line + gtld +': ' + str(gtlddic[gtld])
    print(line)

                
