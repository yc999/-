import eventlet
import time
# eventlet.monkey_patch(time=True)
time_limit = 1  #set timeout time 3s
with eventlet.Timeout(time_limit,False):
    # time.sleep(8)
    a=1
    while True:
        a=a+1
print("out")
