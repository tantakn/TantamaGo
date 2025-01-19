import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
import psutil
import threading
import torch

import json, subprocess

subprocess.run(["rm", "testpipe1"])
subprocess.run(["rm", "testpipe2"])
subprocess.run(["mkfifo", "testpipe1"])
subprocess.run(["mkfifo", "testpipe2"])



subprocess.Popen(["python3", "zzz.py"])
fifo2 = open('testpipe2', 'r')

s = fifo2.read()
print(s, file=sys.stderr)


time.sleep(2)


# s = "zzpy"

# fifo1 = open('testpipe1', 'w')


# fifo1.write(s)



# s = fifo2.read()
# print(s, file=sys.stderr)




fifo1.close()
fifo2.close()


# time.sleep(2)


# s = "zzpy"

# with open('testpipe1', 'w') as fifo1:
#     fifo1.write(s)
#     fifo1.close()



# with open('testpipe2', 'r') as fifo2:
#     s = fifo2.read()
#     print(s, file=sys.stderr)
#     fifo2.close()


# # print(s, file="./testpipe1")

