from data_init import data_init, add_noise
import numpy as np
from sklearn.linear_model import LogisticRegression

def pred_environment(N, n_session, n_environments, n_trials, probs, n_probs, params={}):
  a = data_init(N, n_session, n_environments, params)
  a_noisy = np.zeros((n_trials, n_environments, n_session, N))
  accuracies = np.zeros((n_probs, n_session))
  for i_prob, prob in enumerate(probs):
    for i in range(n_trials):
      a_noisy[i] = add_noise(a, prob)

    e_decoder = LogisticRegression(max_iter=2000)

    X_train = np.vstack(a_noisy[:,:,0,:])
    y_train = np.tile(np.arange(n_environments), n_trials)

    e_decoder.fit(X_train, y_train)

    for j in range(n_session):
      X_test = np.vstack(a_noisy[:,:,j,:])
      y_test = np.tile(np.arange(n_environments), n_trials)

      pred = e_decoder.predict(X_test)
      accuracies[i_prob,j] = (y_test == pred).sum() / y_test.shape[0]
  return accuracies


def pred_session(N, n_session, n_environments, n_trials, probs, n_probs, params={}):
  a = data_init(N, n_session, n_environments, params)
  a_noisy = np.zeros((n_trials, n_environments, n_session, N))
  accuracies = np.zeros((n_probs, n_session))
  for i_prob, prob in enumerate(probs):
    for i in range(n_trials):
      a_noisy[i] = add_noise(a, prob)

    e_decoder = LogisticRegression(max_iter=2000)

    X_train = np.reshape(a_noisy[:int(n_trials*0.8),:,:,:], (int(n_trials*0.8)*n_environments*n_session, N))
    y_train = np.tile(np.arange(n_session), int(n_trials*0.8)*n_environments)

    e_decoder.fit(X_train, y_train)

    for j in range(n_session):
      X_test = np.reshape(a_noisy[int(n_trials*0.8):,:,j,:], (int(n_trials*0.2)*n_environments, N))
      y_test = np.tile(j, int(n_trials*0.2)*n_environments) 

      pred = e_decoder.predict(X_test)
      accuracies[i_prob,j] = (y_test == pred).sum() / y_test.shape[0]
      
  return accuracies


def pred_environment2(N, n_session, n_environments, n_trials, probs, n_probs, params={}):
  a = data_init(N, n_session, n_environments, params)
  a_noisy = np.zeros((n_trials, n_environments, n_session, N))
  accuracies = np.zeros((n_probs, n_session))
  for i_prob, prob in enumerate(probs):
    for i in range(n_trials):
      a_noisy[i] = add_noise(a, prob)

    e_decoder = LogisticRegression(max_iter=2000)

    X_train = np.reshape(a_noisy[:int(n_trials*0.8),:,:,:], (int(n_trials*0.8)*n_environments*n_session, N))
    y_train = np.tile(np.repeat(np.arange(n_environments), n_session), int(n_trials*0.8))

    e_decoder.fit(X_train, y_train)

    for j in range(n_session):
      X_test = np.reshape(a_noisy[int(n_trials*0.8):,:,:,:], (int(n_trials*0.2)*n_environments*n_session, N))
      y_test = np.tile(np.repeat(np.arange(n_environments), n_session), int(n_trials*0.2))

      pred = e_decoder.predict(X_test)
      accuracies[i_prob,j] = (y_test == pred).sum() / y_test.shape[0]
  return accuracies