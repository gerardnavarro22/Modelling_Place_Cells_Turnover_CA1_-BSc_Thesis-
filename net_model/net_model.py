# %%
from cell import cell
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def asynchronous_update(T, plot=True):
    spikes = np.zeros((N,T))
    for t in range(T):
        idx = np.random.randint(N)
        before = population[idx].active
        population[idx].update(J, population, theta, ext)
        after = population[idx].active
        if (after-before == 1):
            spikes[idx,t]=1

    if plot:
        fig, ax = plt.subplots(figsize=(15,15))
        fig.patch.set_facecolor('white')
        ax.set_title('spikes of asynchronous update')
        ax.set_xlabel("time")
        ax.set_ylabel("neuron")
        #ax.set_xticks()
        ax.imshow(spikes[:,:], origin='lower')
        plt.show()

    return spikes

def all_update(T, plot=True):
    spikes = np.zeros((N,T))
    for t in range(T):
        pre_population = population.copy()
        for i in range(N):
            before = population[i].active
            population[i].update(J, pre_population, theta, ext)
            after = population[i].active
            if (after-before == 1):
                spikes[i,t]=1

    if plot:
        fig, ax = plt.subplots(figsize=(15,15))
        fig.patch.set_facecolor('white')
        ax.set_title('spikes of synchronous update')
        ax.set_xlabel("time")
        ax.set_ylabel("neuron")
        #ax.set_xticks()
        ax.imshow(spikes[:,:], origin='lower')
        plt.show()

    return spikes

def plot_spikes(spikes, T):
    spikes_e = np.sum(spikes[:N_I,:], axis=0)
    spikes_i = np.sum(spikes[N_I:,:], axis=0)
    
    bin=int(T/500)
    x = np.arange(0, T, bin)
    inter_spike_e = [sum(spikes_e[bin*i:bin*i+bin]) for i in range(int(T/bin))]
    inter_spike_i = [sum(spikes_i[bin*i:bin*i+bin]) for i in range(int(T/bin))]

    def func(x, a, c, d):
        return a*np.exp(-c*x)+d

    fig, axs = plt.subplots(2, constrained_layout=True, figsize=(8,6))
    fig.patch.set_facecolor('white')

    popt, pcov = curve_fit(func,  x,  inter_spike_e, p0 = (1, 1e-6, 1))
    xx = np.linspace(0, 50000, 1000)
    yy = func(xx, *popt)
    axs[0].set_title("Excitatory cells spikes")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("spikes count")
    axs[0].plot(x, inter_spike_e, '.')
    axs[0].plot(xx, yy, label='fitted negative exponential')
    axs[0].legend()

    popt, pcov = curve_fit(func,  x,  inter_spike_i, p0 = (1, 1e-6, 1))
    xx = np.linspace(0, 50000, 1000)
    yy = func(xx, *popt)
    axs[1].set_title("Inhibitory cells spikes")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("spikes count")
    axs[1].plot(x, inter_spike_i, '.')
    axs[1].plot(xx, yy, label='fitted negative exponential')
    axs[1].legend()
    plt.show()

if __name__ == "__main__":
    N_K=1           #total excitatory populations
    N_I=10000        #total excitatory neurons
    N_L=1           #total inhibitory populations
    N_J=10000        #total inhibitory neurons
    N=N_I+N_J       #total neurons
    con_idx=10      #connectivity index K
    m_0=0.3         #mean activity of external neurons

    J = np.zeros((N_K+1, N_L+1))    #values of connections J(postsynaptic,presynaptic)
    #when presynaptic cell is inhibitory negative connection, otherwise positive
    J[0,0] = 1/np.sqrt(con_idx)
    J[0,1] = -1.1/np.sqrt(con_idx)
    J[1,0] = 1/np.sqrt(con_idx)
    J[1,1] = -1.2/np.sqrt(con_idx)

    ext = np.array([1.15*m_0*np.sqrt(con_idx), 0.92*m_0*np.sqrt(con_idx)])  #external inputs
    theta = np.array([0.87, 0.87])                                          #thresholds for each population
    probs = np.array([con_idx/N_I, con_idx/N_J])                            #probability of a connection happening

    #creating population of cells
    population = []

    for i in range(N_I):
        population.append(cell(np.random.choice(2, p=(0.8, 0.2)), i, 0))

    for i in range(N_J):
        population.append(cell(np.random.choice(2, p=(0.8, 0.2)), i, 1))

    #creating connections between cells
    for i in range(N):
        for j in range(N):
            if (np.random.rand()<probs[population[i].k]):
                population[i].pre.append(j)

    n_active = sum(c.active for c in population)
    n_active_e = sum(c.active for c in population if c.k == 0)
    n_active_i = sum(c.active for c in population if c.k == 1)

    print(f'initial total active cells = {n_active} ({round(n_active/N*100,2)}%)')
    print(f'initial excitatory active cells = {n_active_e} ({round(n_active_e/N_I*100,2)}%)')
    print(f'initial inhibitory active cells = {n_active_i} ({round(n_active_i/N_J*100,2)}%)')

    #simulating cells
    T=50000
    spikes = asynchronous_update(T, False)

    n_active = sum(c.active for c in population)
    n_active_e = sum(c.active for c in population if c.k == 0)
    n_active_i = sum(c.active for c in population if c.k == 1)

    print(f'final total active cells = {n_active} ({round(n_active/N*100,2)}%)')
    print(f'final excitatory active cells = {n_active_e} ({round(n_active_e/N_I*100,2)}%)')
    print(f'final inhibitory active cells = {n_active_i} ({round(n_active_i/N_J*100,2)}%)')

    #plotting spikes
    plot_spikes(spikes, T)

    #actualizar todas a la vez
    #mirar si spike canvio o si activada

# %%
