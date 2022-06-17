# %%
import ctypes
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
os.system('sh compile.sh')

def simulate(K):
    print(f'K={K}')
    N_E=K*16
    N_I=K*4
    N=N_E+N_I
    T = 20
    c_double_p = ctypes.POINTER(ctypes.c_double)
    f = ctypes.CDLL(dir_path+'/library.so').simulate
    f.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, c_double_p, c_double_p]

    exp_e=ctypes.c_double()
    exp_i=ctypes.c_double()
    flag = f(ctypes.c_int(N_E), ctypes.c_int(N_I), ctypes.c_int(K), ctypes.c_int(T), ctypes.byref(exp_e), ctypes.byref(exp_i))

    me = np.loadtxt(f'./me_{K}.txt', ndmin=1)
    mi = np.loadtxt(f'./mi_{K}.txt', ndmin=1)

    obs_ex = np.loadtxt(f'./obs_ex_{K}.txt', ndmin=1)
    obs_in = np.loadtxt(f'./obs_in_{K}.txt', ndmin=1)
    obs_spikes = np.loadtxt(f'./obs_spikes_{K}.txt', ndmin=1)

    with open(f'./spikes_e_{K}.txt', 'r') as f:
            spikes_e = []
            for ele in f:
                line = list(map(int, ele.split('\n')[0].split()))
                spikes_e.append(line)

    with open(f'./spikes_i_{K}.txt', 'r') as f:
            spikes_i = []
            for ele in f:
                line = list(map(int, ele.split('\n')[0].split()))
                spikes_i.append(line)

    #PLOTTING RATES 
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(5,5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'#active cells K={K}', fontsize=14)

    axs[0].set_title('Excitatory cells')
    axs[0].plot(me/N_E)
    axs[0].axhline(y=exp_e.value, color='r', linestyle='-', label=f'expected $m_e$={exp_e.value}')
    axs[0].set_xticks([0,T])
    axs[0].set_ylim([0,1])
    axs[0].set_xlabel('time')
    axs[0].legend()
    axs[1].set_title('Inhibitory cells')
    axs[1].plot(mi/N_I)
    axs[1].axhline(y=exp_i.value, color='r', linestyle='-', label=f'expected $m_i$={exp_i.value}')
    axs[1].set_xticks([0,T])
    axs[1].set_xlabel('time')
    axs[1].set_ylim([0,1])
    axs[1].legend()
    fig.savefig(f"./rates_figures/K_{K}.png")
    plt.close()

    #PLOTTING SPIKES
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(5,5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Neuron spikes K={K}', fontsize=14)
    
    max_cell_e = 100
    max_cell_i = 100
    axs[0].set_title('Excitatory cells')
    for i in range(T):
        spikes_t = np.array(spikes_e[i])
        spikes_t = spikes_t[spikes_t<max_cell_e]
        axs[0].scatter(np.repeat(i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')
    axs[0].set_xlabel('time')
    axs[0].set_xticks([0,T])
    axs[1].set_title('Inhibitory cells')
    for i in range(T):
        spikes_t = np.array(spikes_i[i])
        spikes_t = spikes_t[spikes_t<max_cell_i]
        axs[1].scatter(np.repeat(i,spikes_t.shape[0]), spikes_t, s=3, marker='s', color='black')
    axs[1].set_xlabel('time')
    axs[1].set_xticks([0,T])
    fig.savefig(f"./spikes_figures/K_{K}.png")
    plt.close()




    #PLOTTING SPIKES2
    max_cell_e = 100
    max_cell_i = 100
    spikes_ef = np.array([x for xs in spikes_e for x in xs])
    spikes_ef = spikes_ef[spikes_ef<max_cell_e]
    spikes_if = np.array([x for xs in spikes_i for x in xs])
    spikes_if = spikes_if[spikes_if<max_cell_i]
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(5,5))
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Neuron spikes K={K}', fontsize=14)
    
    axs[0].set_title('Excitatory cells')
    x = np.arange(0,len(spikes_ef))
    axs[0].scatter(x, spikes_ef, s=3, marker='s', color='blue')
    axs[0].set_xlabel('time')
    axs[0].set_xticks([])
    axs[1].set_title('Inhibitory cells')
    x = np.arange(0,len(spikes_if))
    axs[1].scatter(x, spikes_if, s=3, marker='s', color='blue')
    axs[1].set_xlabel('time')
    axs[1].set_xticks([])
    fig.savefig(f"./spikes_figures/2K_{K}.png")
    plt.close()




    #PLOTTING INPUT OF RANDOM EXCITATORY CELL
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(5,5), gridspec_kw={'height_ratios': [6, 1]}, sharex=True)
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Temporal structure of the input K={K}.', fontsize=13)

    axs[0].set_ylabel('Input', fontsize=14)
    axs[0].plot(obs_ex, color='red', label = 'ex. input')
    axs[0].plot(obs_in, color='blue', label = 'in. input')
    axs[0].plot(obs_ex+obs_in, color='green', label = 'net input')
    axs[0].axhline(y=0, color='black', linestyle='-')
    axs[0].axhline(y=1, color='black', linestyle='--')
    axs[0].spines['bottom'].set_visible(False)
    min_max_y = int(np.max(np.absolute(np.concatenate([obs_ex,obs_in])))+1)
    axs[0].set_ylim([-min_max_y*1.3,min_max_y*1.3])
    axs[1].set_ylabel('Spikes', fontsize=14)
    for spike in obs_spikes:
        axs[1].axvline(x=spike, color='black', linestyle='-')
    axs[1].set_yticks([])
    axs[1].set_xticks([0,T])
    axs[0].set_xticks([])
    axs[0].legend()
    axs[1].set_xlabel('time')
    axs[1].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    fig.align_ylabels(axs)
    fig.savefig(f"./obs_figures/K_{K}.png")
    plt.close()

    sim_e = np.mean((me/N_E)[-int(T*0.5):])
    sim_i = np.mean((mi/N_I)[-int(T*0.5):])
    me_k.append(sim_e)
    mi_k.append(sim_i)
    return sim_e, sim_i, exp_e.value, exp_i.value

me_k = []
mi_k = []
Ks = [100,200,400,600,800,1000,2000,4000,6000]
#Ks = [10000]

me_k, mi_k, exp_e, exp_i = zip(*Parallel(n_jobs=-1)(delayed(simulate)(K) for K in Ks))

os.system('rm *.so *.txt')

#PLOTTING RATES VS K
if (len(Ks)>1):
    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(5,5))
    fig.patch.set_facecolor('white')
    axs[0].set_title('Excitatory cells')
    axs[0].plot(Ks, me_k)
    axs[0].axhline(y=exp_e[0], color='r', linestyle='-')
    axs[0].set_xlabel('connectivity index K')
    axs[0].set_xticks([Ks[0],Ks[-1]])
    axs[0].set_ylim([0,1])
    fig.suptitle('#active cells', fontsize=14)
    axs[1].set_title('Inhibitory cells')
    axs[1].plot(Ks, mi_k)
    axs[1].axhline(y=exp_i[0], color='r', linestyle='-')
    axs[1].set_xlabel('connectivity index K')
    axs[1].set_xticks([Ks[0],Ks[-1]])
    axs[1].set_ylim([0,1])
    fig.savefig(f"./K_rates_figures/K_{Ks[0]}_{Ks[-1]}.png")
    plt.close()



# %%
