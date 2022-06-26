# %%
from data_init import data_init, add_noise
import numpy as np
from sklearn.linear_model import LogisticRegression

N = 2000
n_session = 8
n_environments = 1


a = [np.squeeze(data_init(N, n_session, n_environments)).T for _ in range(100)]

#To epresent the data we take out the neurons that are never active
a_active = []
for i in range(100):
  idx_active = np.where(np.sum(a[i], axis=1)!=0)[0]
  a_active.append(a[i][idx_active])

all_frac = []
for j in range(100):
  N_active = a_active[j].shape[0]
  frac_cells = []
  for i in range(1,n_session+1):
    frac_cells.append(np.sum(np.sum(a_active[j], axis=1)==i)/N_active)
  all_frac.append(np.array(frac_cells))
all_frac = np.array(all_frac)
mean_frac = np.mean(all_frac, axis=0)

all_overlap = []
for j in range(100):
  N_active = a_active[j].shape[0]
  overlap = []
  for i in range(0, n_session):
    overlap.append(np.sum(a_active[j][:,0] == a_active[j][:,i])/N_active)
  all_overlap.append(np.array(overlap))
all_overlap = np.array(all_overlap)
mean_overlap = np.mean(all_overlap, axis=0)


# %%
