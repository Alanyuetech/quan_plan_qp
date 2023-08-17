#写一个函数，每一秒输出一个正整数，遇到质数时，保存到日志中，然后继续输出
import time
import math
import logging
logging.basicConfig(filename='test.log',level=logging.DEBUG)
def is_prime(n):
    if n <= 1:
        return False
    for i in range(2,int(math.sqrt(n))+1):
        if n%i == 0:
            return False
    return True
def print_prime():
    i = 0
    while True:
        if is_prime(i):
            print(i)
            logging.info(i)
        i += 1
        time.sleep(1)
print_prime()






