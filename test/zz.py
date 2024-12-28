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

s = "zzpy"

with open('testpipe', 'w') as fifo:
    fifo.write(s)
    s = fifo.read()
    print(s, file=sys.stderr)
