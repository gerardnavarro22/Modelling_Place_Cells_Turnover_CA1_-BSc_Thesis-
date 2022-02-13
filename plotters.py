import decoders
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
import time

def environment_plotter():
  N = 400
  n_session = 50
  n_environments = 7
  n_trials = 10
  probs = [0,0.05,0.1,0.15,0.2]
  n_probs = len(probs)
  n_father_matrcies = 50
  
  accuracies = np.array(Parallel(n_jobs=-1)(delayed(decoders.pred_environment)(N, n_session, n_environments, n_trials, probs, n_probs) for i in range(n_father_matrcies)))
  
  means = np.zeros((n_probs, n_session))
  stds = np.zeros((n_probs, n_session))
  ci = np.zeros((n_probs, n_session))
  for j in range(n_probs):
    means[j] = np.array([np.mean(accuracies[:,j,i]) for i in range(n_session)])
    stds[j] = np.array([np.std(accuracies[:,j,i]) for i in range(n_session)])
    ci[j] = 1.96 * stds[j]/np.sqrt(n_father_matrcies)
    
  delta_t = np.arange(0,n_session)
  fig, ax = plt.subplots()
  fig.patch.set_facecolor('white')
  ax.axhline(y = 1/n_environments, color = 'black', linestyle = 'dashed') 
  ax.set_xlabel("$\Delta$ sessions")
  ax.set_ylabel("Accuracy")
  ax.set_yticks([1/n_environments,1])
  ax.set_xticks(delta_t[::6])
  for i in range(n_probs):
    ax.plot(delta_t, means[i], label=f'p={probs[i]}')
    #ax.errorbar(delta_t, means[i], yerr=ci[i], fmt='.k')
    ax.fill_between(delta_t, (means[i]-ci[i]), (means[i]+ci[i]), alpha=.2)
    ax.legend(loc='best')
  date = time.strftime("%Y-%m-%d %H%M%S")
  plt.savefig(f'./figures/env_{date}.png', dpi=fig.dpi)

def sessions_plotter():
  N = 400
  n_session = 50
  n_environments = 7
  n_trials = 10
  probs = [0,0.05,0.1,0.15,0.2,0.3,0.4,0.5]
  n_probs = len(probs)
  n_father_matrcies = 50
  
  accuracies = np.array(Parallel(n_jobs=-1)(delayed(decoders.pred_session)(N, n_session, n_environments, n_trials, probs, n_probs) for i in range(n_father_matrcies)))
  
  means = np.zeros((n_probs, n_session))
  stds = np.zeros((n_probs, n_session))
  ci = np.zeros((n_probs, n_session))
  for j in range(n_probs):
    means[j] = np.array([np.mean(accuracies[:,j,i]) for i in range(n_session)])
    stds[j] = np.array([np.std(accuracies[:,j,i]) for i in range(n_session)])
    ci[j] = 1.96 * stds[j]/np.sqrt(n_father_matrcies)
  
  delta_t = np.arange(0,n_session)
  fig, ax = plt.subplots()
  fig.patch.set_facecolor('white')
  ax.axhline(y = 1/n_session, color = 'black', linestyle = 'dashed') 
  ax.set_xlabel("$\Delta$ sessions")
  ax.set_ylabel("Accuracy")
  ax.set_yticks([1/n_session,1])
  ax.set_xticks(delta_t[::6])
  for i in range(n_probs):
    ax.plot(delta_t, means[i], label=f'p={probs[i]}')
    #ax.errorbar(delta_t, means[i], yerr=ci[i], fmt='.k')
    ax.fill_between(delta_t, (means[i]-ci[i]), (means[i]+ci[i]), alpha=.2)
    ax.legend(loc='right')
  date = time.strftime("%Y-%m-%d %H%M%S")
  plt.savefig(f'./figures/sess_{date}.png', dpi=fig.dpi)
    
def environment_param_heatmap_plotter(p=0.3, ret=False):
  N = 400
  n_session = 30
  n_environments = 5
  n_trials = 10
  probs = [p]
  n_probs = len(probs)
  n_father_matrcies = 12
  acc_heatmap = np.zeros((11,11))
  for i in tqdm(range(11)):
    for j in range(11):
      params = {'rho_s': i*0.1,
                'rho_s_AB': j*0.1}
      
      accuracies = np.array(Parallel(n_jobs=-1)(delayed(decoders.pred_environment2)(N, n_session, n_environments, n_trials, probs, n_probs, params) for i in range(n_father_matrcies)))
      acc_heatmap[i,j] = np.mean(np.array([np.mean(accuracies[:,:,k]) for k in range(n_session)]))
    
  fig, ax = plt.subplots()
  fig.patch.set_facecolor('white')
  im = ax.imshow(acc_heatmap, cmap='Reds_r', origin='lower')
  ax.set_title("Environment decoder performance")
  ax.set_xticks([0,10], labels=['0', '1'])
  ax.set_yticks([0,10], labels=['0', '1'])
  ax.set_xlabel('corr. env.')
  ax.set_ylabel('corr. sessions')
  fig.colorbar(im)
  date = time.strftime("%Y-%m-%d %H%M%S")
  plt.savefig(f'./figures/environment_param_heatmap_{date}.png', dpi=fig.dpi)
  if ret: return acc_heatmap
    
def session_param_heatmap_plotter(p=0.3, ret=False):
  N = 400
  n_session = 30
  n_environments = 5
  n_trials = 10
  probs = [p]
  n_probs = len(probs)
  n_father_matrcies = 50
  acc_heatmap = np.zeros((11,11))
  for i in tqdm(range(11)):
    for j in range(11):
      params = {'rho_s': i*0.1,
                'rho_s_AB': j*0.1}
      
      accuracies = np.array(Parallel(n_jobs=-1)(delayed(decoders.pred_session)(N, n_session, n_environments, n_trials, probs, n_probs, params) for i in range(n_father_matrcies)))
      acc_heatmap[i,j] = np.mean(np.array([np.mean(accuracies[:,:,k]) for k in range(n_session)]))
    
  fig, ax = plt.subplots()
  fig.patch.set_facecolor('white')
  im = ax.imshow(acc_heatmap, cmap='Reds_r', origin='lower')
  ax.set_title("Session decoder performance")
  ax.set_xticks([0,10], labels=['0', '1'])
  ax.set_yticks([0,10], labels=['0', '1'])
  ax.set_xlabel('corr. env.')
  ax.set_ylabel('corr. sessions')
  fig.colorbar(im)
  date = time.strftime("%Y-%m-%d %H%M%S")
  plt.savefig(f'./figures/session_param_heatmap_plotter_{date}.png', dpi=fig.dpi)
  if ret: return acc_heatmap


def combined_heatmap(p):
  heatmap1 = environment_param_heatmap_plotter(p=p, ret=True)
  heatmap2 = session_param_heatmap_plotter(ret=True)
  comb_heatmap = np.multiply(heatmap1, heatmap2)
  fig, ax = plt.subplots()
  fig.patch.set_facecolor('white')
  im = ax.imshow(comb_heatmap, cmap='Reds_r', origin='lower')
  ax.set_title("Combined decoder performance")
  ax.set_xticks([0,10], labels=['0', '1'])
  ax.set_yticks([0,10], labels=['0', '1'])
  ax.set_xlabel('corr. env.')
  ax.set_ylabel('corr. sessions')
  fig.colorbar(im)
  date = time.strftime("%Y-%m-%d %H%M%S")
  plt.savefig(f'./figures/combined_param_heatmap_plotter_{date}_{p}.png', dpi=fig.dpi)
