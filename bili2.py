num = {}
tmpnum = input()
count = 0
while tmpnum != '':
    mylist = tmpnum.split(',')
    count  = count +1
    for i in mylist:
            if i in num.keys():
                num[i] = num[i]+1
            else:
                num[i] = 1
    tmpnum = input()

minnum = 999999

for i in num.keys():
    if num[i]==count and int(i)<minnum:
        minnum = int(i)
if minnum == 999999:
    print(-1)
else:
    print(minnum)
'''   
1,2,3,4,5,9
2,4,5,8,910
2,5,7,9,11
2,3,5,7,9
'''