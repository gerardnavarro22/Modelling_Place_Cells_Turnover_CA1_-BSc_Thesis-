import numpy as np

global J,N_K,N_I,N_L,N_J,N,ext,theta,con_idx,probs,m_0
N_K=1        #total excitatory populations
N_I=1000      #total excitatory neurons
N_L=1        #total inhibitory populations
N_J=1000      #total inhibitory neurons
N=N_I+N_J    #total neurons
con_idx=10   #connectivity index K
m_0=0.3      #mean activity of external neurons

J = np.zeros((N_K+1, N_L+1))   #values of connections
J[0,0] = 1/np.sqrt(con_idx)
J[0,1] = 1.1/np.sqrt(con_idx)
J[1,0] = 1/np.sqrt(con_idx)
J[1,1] = 1.2/np.sqrt(con_idx)

ext = np.array([1.15*m_0*np.sqrt(con_idx), 0.92*m_0*np.sqrt(con_idx)])  #external inputs
theta = np.array([0.87, 0.87])                                          #thresholds for each population
probs = np.array([con_idx/N_I, con_idx/N_J])                            #probability of a connection happening


def heaviside(x):
    if x<=0: return False
    else: return True

class cell:
    def __init__(self, active, i, k, pre=[]):
        self.active = active
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

#creating population of cells   
global population 
population = []

for i in range(N_I):
    population.append(cell(np.random.randint(2), i, 0))
    
for i in range(N_J):
    population.append(cell(np.random.randint(2), i, 1))

#creating connections between cells    
for i in range(N):
    for j in range(N):
        if (np.random.rand()<probs[population[i].k]):
            population[i].pre.append(j)

n_active = 0
for i in range(N):
    n_active += population[i].active

print(f'initial active cells = {n_active}')

for t in range(50000):
    idx = np.random.randint(N)
    before = population[idx].active
    population[idx].update()
    after = population[idx].active
    #if (after-before == 1):
    #    print(f'spike! t={t}')
        
n_active = 0
for i in range(N):
    n_active += population[i].active

print(f'final active cells = {n_active}')