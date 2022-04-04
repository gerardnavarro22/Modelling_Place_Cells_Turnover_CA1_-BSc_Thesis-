# %%
import ctypes
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

me_k = []
mi_k = []
Ks = [100,200,400,600,800,1000,2000,4000,6000]
#Ks = [100]
for K in Ks:
    print(f'K={K}')
    os.system('sh compile.sh')
    N_E=20000
    N_I=8000
    N=N_E+N_I
    T = 20
    c_double_p = ctypes.POINTER(ctypes.c_double)
    f = ctypes.CDLL(dir_path+'/library.so').simulate
    f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, c_double_p, c_double_p]

    exp_e=ctypes.c_double()
    exp_i=ctypes.c_double()
    spikes = f(ctypes.c_int(N_E), ctypes.c_int(N_I), ctypes.c_int(K), ctypes.c_int(T), ctypes.byref(exp_e), ctypes.byref(exp_i))

    me = np.loadtxt('./me.txt', ndmin=1)
    mi = np.loadtxt('./mi.txt', ndmin=1)

    obs_ex = np.loadtxt('./obs_ex.txt', ndmin=1)
    obs_in = np.loadtxt('./obs_in.txt', ndmin=1)
    obs_spikes = np.loadtxt('./obs_spikes.txt', ndmin=1)

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

    #PLOTTING RATES 
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
    fig.patch.set_facecolor('white')
    fig.suptitle('#active cells', fontsize=16)

    axs[0].set_title('Excitatory cells')
    axs[0].plot(me/N_E)
    axs[0].axhline(y=exp_e.value, color='r', linestyle='-')
    axs[0].set_xticks([0,T])
    axs[0].set_ylim([0,1])
    axs[1].set_title('Inhibitory cells')
    axs[1].plot(mi/N_I)
    axs[1].axhline(y=exp_i.value, color='r', linestyle='-')
    axs[1].set_xticks([0,T])
    axs[1].set_ylim([0,1])
    fig.savefig(f"./rates_figures/K_{K}.png")
    plt.close()

    #PLOTTING SPIKES
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(16,9))
    fig.patch.set_facecolor('white')
    fig.suptitle('Neuron spikes', fontsize=16)
    
    max_cell = 50
    axs[0].set_title('Excitatory cells')
    for i in range(T):
        spikes_t = np.array(spikes_e[i])
        spikes_t = spikes_t[spikes_t<max_cell]
        axs[0].scatter(np.repeat(i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')
    axs[0].set_xlabel('time')
    axs[0].set_xticks([0,T])
    axs[1].set_title('Inhibitory cells')
    for i in range(T):
        spikes_t = np.array(spikes_i[i])
        spikes_t = spikes_t[spikes_t<max_cell]
        axs[1].scatter(np.repeat(i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')
    axs[1].set_xlabel('time')
    axs[1].set_xticks([0,T])
    fig.savefig(f"./spikes_figures/K_{K}.png")
    plt.close()

    #PLOTTING INPUT OF RANDOM EXCITATORY CELL
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
    min_max_y = int(np.max(np.absolute(np.concatenate([obs_ex,obs_in])))+1)
    axs[0].set_ylim([-min_max_y,min_max_y])
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

    sim_e = np.mean((me/N_E)[-int(T*0.8):])
    sim_i = np.mean((mi/N_I)[-int(T*0.8):])
    me_k.append(sim_e)
    mi_k.append(sim_i)

#PLOTTING RATES VS K
if (len(Ks)>1):
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
    fig.patch.set_facecolor('white')
    axs[0].set_title('Excitatory cells')
    axs[0].plot(Ks, me_k)
    axs[0].axhline(y=exp_e.value, color='r', linestyle='-')
    axs[0].set_xlabel('connectivity index K')
    axs[0].set_xticks([Ks[0],Ks[-1]])
    axs[0].set_ylim([0,1])
    fig.suptitle('#active cells', fontsize=16)
    axs[1].set_title('Inhibitory cells')
    axs[1].plot(Ks, mi_k)
    axs[1].axhline(y=exp_i.value, color='r', linestyle='-')
    axs[1].set_xlabel('connectivity index K')
    axs[1].set_xticks([Ks[0],Ks[-1]])
    axs[1].set_ylim([0,1])
    fig.savefig(f"./K_rates_figures/K_{Ks[0]}_{Ks[-1]}.png")
    plt.close()


# %%
