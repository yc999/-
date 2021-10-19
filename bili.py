num = input()
mylist = num.split(',')
lenth = len(mylist)
half = int(lenth/2)
ret = []
for i in range(half):
    ret.append(mylist[i])
    ret.append(mylist[lenth -1 - i])

if lenth%2!=0:
    ret.append(mylist[half ])
for i in range(lenth-1):
    print(ret[i]+ ',',end='')
print(ret[lenth-1])

