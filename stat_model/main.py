import plotters
import decoders
import numpy as np
from data_init import data_init, add_noise

    
if __name__ == "__main__":
    N = 400
    n_session = 80
    n_environments = 7
    n_trials = 20
    probs = [0, 0.05, 0.1, 0.2, 0.3, 0.5]
    n_father_matrcies = 100

    plotters.environment_plotter(N, n_session, n_environments, n_trials, probs, n_father_matrcies)
    plotters.sessions_plotter(N, n_session, n_environments, n_trials, probs, n_father_matrcies)