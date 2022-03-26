# %%
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

me_k = []
mi_k = []
Ks = [100]
for K in Ks:
    print(f'K={K}')
    os.system('sh compile.sh')
    N_E=1000
    N_I=1000
    N=N_E+N_I
    #K=100
    T = 250
    c_double_p = ctypes.POINTER(ctypes.c_double)
    f = ctypes.CDLL(dir_path+'/library.so').simulate
    f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, c_double_p, c_double_p]

    exp_e=ctypes.c_double()
    exp_i=ctypes.c_double()
    spikes = f(ctypes.c_int(N_E), ctypes.c_int(N_I), ctypes.c_int(K), ctypes.byref(exp_e), ctypes.byref(exp_i))

    me = np.loadtxt('./me.txt')
    mi = np.loadtxt('./mi.txt')

    obs_ex = np.loadtxt('./obs_ex.txt')
    obs_in = np.loadtxt('./obs_in.txt')
    obs_spikes = np.loadtxt('./obs_spikes.txt')

    with open('./spikes_e.txt', 'r') as f:
            spikes_e = []
            for ele in f:
                line = list(map(int, ele.split('\n')[0].split()))
                spikes_e.append(line)

    with open('./spikes_i.txt', 'r') as f:
            spikes_i = []
            for ele in f:
                line = list(map(int, ele.split('\n')[0].split()))
                spikes_i.append(line)

    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
    fig.patch.set_facecolor('white')
    fig.suptitle('#active cells', fontsize=16)

    axs[0].set_title('Excitatory cells')
    axs[0].plot(me)
    axs[0].axhline(y=exp_e.value*N_E, color='r', linestyle='-')
    axs[1].set_title('Inhibitory cells')
    axs[1].plot(mi)
    axs[1].axhline(y=exp_i.value*N_I, color='r', linestyle='-')
    fig.savefig(f"./rates_figures/K_{K}.png")
    plt.close()

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
    axs[0].set_xlabel('time')
    axs[1].set_title('Inhibitory cells')
    for i in range(100):
        spikes_t = np.array(spikes_i[beg+i])
        spikes_t = spikes_t[spikes_t<max_cell]
        axs[1].scatter(np.repeat(beg+i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')
    axs[1].set_xlabel('time')
    fig.savefig(f"./spikes_figures/K_{K}.png")
    plt.close()

    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
    fig.patch.set_facecolor('white')
    fig.suptitle('Temporal structure of the input to an excitatory cell.', fontsize=16)

    axs[0].set_ylabel('Input', fontsize=16)
    axs[0].plot(obs_ex, color='black')
    axs[0].plot(obs_in, color='black')
    axs[0].plot(obs_ex+obs_in, color='black')
    axs[0].axhline(y=0, color='black', linestyle='-')
    axs[0].axhline(y=1, color='black', linestyle='--')
    axs[0].spines['bottom'].set_visible(False)
    axs[1].set_ylabel('Spikes', fontsize=16)
    for spike in obs_spikes:
        axs[1].axvline(x=spike, color='black', linestyle='-')
    axs[1].set_yticks([])
    axs[1].set_xticks([0,T])
    axs[0].set_xticks([])
    axs[1].set_xlabel('time')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    fig.align_ylabels(axs)
    fig.savefig(f"./obs_figures/K_{K}.png")
    plt.close()

    sim_e = np.mean(me[-100:])
    sim_i = np.mean(mi[-100:])
    me_k.append(sim_e)
    mi_k.append(sim_i)

if (len(Ks)>1):
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
    fig.patch.set_facecolor('white')
    axs[0].set_title('Excitatory cells')
    axs[0].plot(Ks, me_k)
    axs[0].axhline(y=exp_e.value*N_E, color='r', linestyle='-')
    axs[0].set_xlabel('connectivity index K')
    fig.suptitle('#active cells', fontsize=16)
    axs[1].set_title('Inhibitory cells')
    axs[1].plot(Ks, mi_k)
    axs[1].axhline(y=exp_i.value*N_I, color='r', linestyle='-')
    axs[1].set_xlabel('connectivity index K')
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
