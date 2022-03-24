# %%
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
os.system('sh compile.sh')
N_E=1000
N_I=1000
N=N_E+N_I
K=100

T = 250*N
size = T*2
c_double_p = ctypes.POINTER(ctypes.c_double)

f = ctypes.CDLL(dir_path+'/library.so').simulate
#f.restype = ctypes.POINTER(ctypes.c_int * size)
f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_double_p, c_double_p]

exp_e=ctypes.c_double()
exp_i=ctypes.c_double()
#spikes = list(f(ctypes.c_int(N_E), ctypes.c_int(N_I), ctypes.c_int(K), ctypes.byref(exp_e), ctypes.byref(exp_i)).contents)
#spikes = np.reshape(spikes, (2,500000))
spikes = f(ctypes.c_int(N_E), ctypes.c_int(N_I), ctypes.c_int(K), ctypes.byref(exp_e), ctypes.byref(exp_i))

me = np.loadtxt('./me(t).txt')
mi = np.loadtxt('./mi(t).txt')

with open('./spikes_e(t).txt', 'r') as f:
        spikes_e = []
        for ele in f:
            line = list(map(int, ele.split('\n')[0].split()))
            spikes_e.append(line)

with open('./spikes_i(t).txt', 'r') as f:
        spikes_i = []
        for ele in f:
            line = list(map(int, ele.split('\n')[0].split()))
            spikes_i.append(line)

fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
fig.patch.set_facecolor('white')

axs[0].set_title('Excitatory cells')
axs[0].plot(me)
axs[0].axhline(y=exp_e.value*N_E, color='r', linestyle='-')
fig.suptitle('#active cells', fontsize=16)
axs[1].set_title('Inhibitory cells')
axs[1].plot(mi)
axs[1].axhline(y=exp_i.value*N_I, color='r', linestyle='-')

plt.show()

fig, axs = plt.subplots(2, constrained_layout=True, figsize=(16,9))
fig.patch.set_facecolor('white')


fig.suptitle('Neuron spikes', fontsize=16)
max_cell = 50
beg = 0

axs[0].set_title('Excitatory cells')
for i in range(100):
    spikes_t = np.array(spikes_e[beg+i])
    spikes_t = spikes_t[spikes_t<max_cell]
    axs[0].scatter(np.repeat(beg+i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')

fig.suptitle('#active cells', fontsize=16)
axs[1].set_title('Inhibitory cells')
for i in range(100):
    spikes_t = np.array(spikes_i[beg+i])
    spikes_t = spikes_t[spikes_t<max_cell]
    axs[1].scatter(np.repeat(beg+i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')
plt.show()



assert(False)

"""
mes = []
mis = []
for K in [10,100,200,500,700,1000,1500]:
    f = ctypes.CDLL(dir_path+'/library.so').simulate
    f.restype = ctypes.POINTER(ctypes.c_int * size)
    f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
    print(K)
    spikes = list(f(ctypes.c_int(N_E), ctypes.c_int(N_I), ctypes.c_int(K)).contents)
    spikes = np.reshape(spikes, (2,500000))
    me = np.loadtxt('./me(t).txt')
    mi = np.loadtxt('./mi(t).txt')
    mes.append(me[-1])
    mis.append(mi[-1])
    del f


bin=int(T/500)
inter_spike_e = [sum(spikes[0][bin*i:bin*i+bin]) for i in range(int(T/bin))]
inter_spike_i = [sum(spikes[1][bin*i:bin*i+bin]) for i in range(int(T/bin))]
fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
fig.patch.set_facecolor('white')

x = np.linspace(0, T, len(inter_spike_e))
axs[0].set_title("Excitatory cells spikes")
axs[0].set_xlabel("time")
axs[0].set_ylabel("spikes count")
axs[0].plot(x, inter_spike_e, '.')
axs[0].legend()

xx = np.linspace(0, T, 1000)
axs[1].set_title("Inhibitory cells spikes")
axs[1].set_xlabel("time")
axs[1].set_ylabel("spikes count")
axs[1].plot(x, inter_spike_i, '.')
axs[1].legend()
plt.show()
"""

# %%
