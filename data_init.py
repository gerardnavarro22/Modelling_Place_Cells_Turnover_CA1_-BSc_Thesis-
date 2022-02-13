import numpy as np

def data_init(N, n_session, n_environments, params={}):
  rho_s = params.get('rho_s', 0.93)
  rho_ns = params.get('rho_ns', 0)
  sigma = params.get('sigma', 1.49)
  theta = params.get('theta', 0.65)
  rho_s_AB = params.get('rho_s_AB', 0.53)
  rho_ns_AB = params.get('rho_ns_AB', 0.52)
  cov_x = rho_s_AB*sigma**2
  cov_y = rho_ns_AB

  cov_mat_x = np.full((n_environments, n_environments),cov_x)
  for i in range(n_environments):
    cov_mat_x[i,i] = sigma**2

  cov_mat_y = np.full((n_environments, n_environments),cov_y)
  for i in range(n_environments):
    cov_mat_y[i,i] = 1

  x = np.zeros((n_environments, n_session, N))
  y = np.zeros((n_environments, n_session, N))

  x[:,0,:] = np.random.multivariate_normal(np.repeat(0, n_environments), cov_mat_x, N).T
  y[:,0,:] = np.random.multivariate_normal(np.repeat(0, n_environments), cov_mat_y, N).T

  for i in range(1, n_session):
    norm = np.random.multivariate_normal(np.repeat(0, n_environments), cov_mat_x, N).T
    x[:,i,:] = rho_s * x[:,i-1,:] + np.sqrt(1-rho_s**2) * norm

  for i in range(1, n_session):
    norm = np.random.multivariate_normal(np.repeat(0, n_environments), cov_mat_y, N).T
    y[:,i,:] = rho_ns * y[:,i-1,:] + np.sqrt(1-rho_ns**2) * norm

  h = x + y

  theta = np.quantile(h[0,0,:],0.5)
  a = np.heaviside(h-theta, 1).astype(int)

  return a

def add_noise(a, p):
  return abs(a - np.random.choice([0,1], p=[1-p, p], size=a.shape))