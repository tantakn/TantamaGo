import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import time
import psutil
import threading
import torch

import json

print("zzz.py started", file=sys.stderr)

time.sleep(2)

fifo1 = open('testpipe1', 'w')

fifo1.write("zzpy")
fifo1.close()











# s = "zzpy"


# fifo1 = open('testpipe1', 'r')


# s = fifo1.read()
# fifo1.close()

# print(s, file=sys.stderr)
# s += "py"


# fifo2 = open('testpipe2', 'w')

# fifo2.write(s)
# fifo2.close()













# with open('testpipe1', 'r') as fifo1:
#     s = fifo1.read()

# print(s, file=sys.stderr)
# s += "py"

    
# time.sleep(2)

# with open('testpipe2', 'w') as fifo2:
#     fifo2.write(s)
#     fifo2.close()



