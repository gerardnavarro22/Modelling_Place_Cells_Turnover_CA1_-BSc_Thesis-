# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

global J,N_K,N_I,N_L,N_J,N,ext,theta,con_idx,probs,m_0
N_K=1           #total excitatory populations 
N_I=1000        #total excitatory neurons
N_L=1           #total inhibitory populations
N_J=1000        #total inhibitory neurons
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


def heaviside(x):
    if x<=0: return False
    else: return True

class cell:
    def __init__(self, active, i, k, pre=[]):
        self.active = bool(active)
        self.pre = list(pre)
        #self.post = post
        self.i = i
        self.k = k
    
    def update(self):
        u = 0
        for j in range(len(self.pre)):
            u += J[self.k, population[self.pre[j]].k]*population[self.pre[j]].active
        u = u + ext[self.k] - theta[self.k]
        self.active = heaviside(u)

    def __repr__(self):
        return f'active={self.active}\nindex={self.i}\npopulation={self.k}\n{len(self.pre)} presynaptic cells'

#creating population of cells   
global population 
population = []

for i in range(N_I):
    population.append(cell(np.random.choice(2, p=(0.5, 0.5)), i, 0))
    
for i in range(N_J):
    population.append(cell(np.random.choice(2, p=(0.5, 0.5)), i, 1))

#creating connections between cells    
for i in range(N):
    for j in range(N):
        if (np.random.rand()<probs[population[i].k]):
            population[i].pre.append(j)

n_active = sum(c.active for c in population)

print(f'initial active cells = {n_active}')

T = 50000
spikes = np.zeros(T)
for t in range(T):
    idx = np.random.randint(N)
    before = population[idx].active
    population[idx].update()
    after = population[idx].active
    if (after-before == 1):
        spikes[t]=1
        
n_active = sum(c.active for c in population)

print(f'final active cells = {n_active}')

bin=100
x = np.arange(0, T, bin)
inter_spike = [sum(spikes[bin*i:bin*i+bin]) for i in range(int(T/bin))]

def func(x, a, c, d):
    return a*np.exp(-c*x)+d

popt, pcov = curve_fit(func,  x,  inter_spike, p0 = (1, 1e-6, 1))
xx = np.linspace(0, 50000, 1000)
yy = func(xx, *popt)

fig, ax = plt.subplots()
fig.patch.set_facecolor('white')
ax.set_xlabel("time")
ax.set_ylabel("spikes count")
#ax.set_xticks()
ax.plot(np.arange(0,T, bin), inter_spike, '.')
ax.plot(xx, yy)
plt.show()