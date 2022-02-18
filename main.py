import plotters
import decoders
import numpy as np
from data_init import data_init, add_noise

    
if __name__ == "__main__":
    N = 400
    n_session = 30
    n_environments = 5
    n_trials = 10
    probs = [0.3]
    n_probs = len(probs)
    n_father_matrcies = 50
    acc_heatmap = np.zeros((11,11))
    params = {'rho_s': 1, 'rho_s_AB': 0.8}

    #print(decoders.pred_session(N, n_session, n_environments, n_trials, probs, n_probs, params))
    a = data_init(N, n_session, n_environments, params)
    print(a[0,0:2,:])