a=input()

b= input()
if int(a)==0:
    print(0)

ret = []
numlist = b.split(" ")
a = int(a)
maxcount = 0
for k in range(a):
    for i in range(k+1, a):
        yinzi = int(numlist[i])/int(numlist[k])
        count =2
        chengshu = int(numlist[i])
        for j in range(i+1, a):
            if(int(numlist[j]) == yinzi * chengshu):
                chengshu = int(numlist[j])
                count = count +1
        if count> maxcount:
            maxcount = count


print(maxcount)