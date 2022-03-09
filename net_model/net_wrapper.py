# %%
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import os

os.system('sh compile.sh')
dir_path = os.path.dirname(os.path.realpath(__file__))
T = 500000
size = T*2
f = ctypes.CDLL(dir_path+'/library.so').simulate
f.restype = ctypes.POINTER(ctypes.c_int * size)
spikes = list(f().contents)
spikes = np.reshape(spikes, (2,500000))

bin=int(T/500)
inter_spike_e = [sum(spikes[0][bin*i:bin*i+bin]) for i in range(int(T/bin))]
inter_spike_e = [sum(spikes[1][bin*i:bin*i+bin]) for i in range(int(T/bin))]

# %%
