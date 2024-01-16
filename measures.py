import numpy as np
import geom
from scipy import optimize

# Compute CA, according to some metric
def collective_alignment(u_nu, u_m, metric):

    mu = sum([u_m[i] * u_nu[i] for i in range(len(u_nu))]) / sum(u_m)
    d = sum([u_m[i] * metric.m(mu - u_nu[i]) for i in range(len(u_nu))])
    z = sum(u_m)
    
    return 1 - (d / z)

# Compute IA, according to some metric
def individual_alignment(u_i_nu, u_hat_i_nu, metric):
    
    return 1 - (metric.m(u_i_nu - u_hat_i_nu) / metric.z)

# Compute the bounds defined in the paper (see paper for explanation of notation)
def get_bounds(G, metric, CC, NEs, eps_NEs, s_hat_star, u_avg):

    IA = G.ia(metric)
    CA = G.ca(metric)
    K = metric.K
    w_0 = min([G.w(s) for s in NEs])
    w_eps = min([G.w(s) for s in eps_NEs])
    w_star = max([G.w(s) for s in G.S])
    m = np.array([u_i.m for u_i in G.u])
    m_hat = np.array([u_hat_i.m for u_hat_i in G.u_hat])

    # Slightly different R term used for bounds in plots (this can be made simpler and more coarse by taking a max over players i, as described in the paper)
    R = sum([((m_hat[i]/m[i]) - 1) * (G.u[i](s_hat_star) - u_avg[i]) for i in range(G.n)]) / G.n
    
    max_regret = ((1 - CC) * (w_star - w_0)) + (w_0 - w_eps) + ((4 * K * np.dot(m_hat, np.ones(G.n) - IA)) / G.n) + R

    # ideal_regret = K * (1 - CA) * (1 / sum(m_hat))
    ideal_regret = K * (1 - CA) * (sum(m_hat) / G.n)

    return max_regret, ideal_regret

# Any metric, e.g. EPIC, must have the following elements
class Metric:
    
    def __init__(self, name, m, c, dist, adjust, constants):
        self.name = name
        self.m = m
        self.c = c
        self.dist = dist
        self.adjust_ia = adjust["ia"]
        self.adjust_ca = adjust["ca"]
        self.z = constants["z"]
        self.K = constants["K"]

    def normalise(self, v):

        centred = v - self.c(v)
        return centred / self.m(centred)
    
    def sample(self, dim):

        u_nu = self.normalise(self.dist(size=dim))
        return u_nu

# Create a set of normalised utility functions that have a pre-specified measure of CA
def adjust_ca_epic(u_nu, u_m, ca, metric, tol=0.01, max_attempts=100):

    if ca == 1.0:
        return [u_nu[0] for _ in u_nu]
    if ca == 0.0:
        tol = 0.025

    mu = sum([u_m[i] * u_nu[i] for i in range(len(u_nu))]) / sum(u_m)
    m_mu = metric.m(mu)

    bases = [geom.orthogonal_project_planar(mu, v) for v in u_nu]
    x_0 = np.array([np.arccos(np.dot(mu,v) / m_mu) for v in u_nu])

    def ca_loss(thetas):

        new_u_nu = [bases[i][0] * np.cos(thetas[i]) + bases[i][1] * np.sin(thetas[i]) for i in range(len(u_nu))]
        current = collective_alignment(new_u_nu, u_m, metric) 

        return (current - ca) ** 2

    res = optimize.minimize(ca_loss, x_0)
    u_nu = [bases[i][0] * np.cos(res.x[i]) + bases[i][1] * np.sin(res.x[i]) for i in range(len(u_nu))]

    return None if np.sqrt(res.fun) > tol else u_nu

    # current = collective_alignment(u_nu, u_m, metric) 
    # j = 0

    # while np.abs(current-ca) > tol and j < max_attempts:
        
    #     thetas = [(current + 10*u_m[i])/(ca + 10*u_m[i]) * thetas[i] for i in range(len(thetas))]
    #     u_nu = [bases[i][0] * np.cos(thetas[i]) + bases[i][1] * np.sin(thetas[i]) for i in range(len(u_nu))]
    #     current = collective_alignment(u_nu, u_m, metric)
    #     # print(j)
    #     j += 1

    # return None if j == max_attempts else u_nu

    # return None if j == max_attempts else u_nu

# Create a new utility function that has a pre-specified measure of IA to another
def adjust_ia_epic(u_i_nu, u_hat_i_nu, ia):

    if ia == 0.0:
        return -u_i_nu
    if ia == 1.0:
        return u_i_nu

    basis = geom.orthogonal_project_planar(u_i_nu, u_hat_i_nu)
    theta = 2 * np.arcsin(1 - ia)
    return basis[0] * np.cos(theta) + basis[1] * np.sin(theta)

epic_adjustment = {"ia": adjust_ia_epic, "ca": adjust_ca_epic}

epic_constants = {"z": 2.0, "K":1.0}

# Define the EPIC measurement (see "Quantifying Differences in Reward Functions" by Gleave et al.)
epic = lambda rng : Metric(
    name="EPIC", 
    m=np.linalg.norm, 
    c=np.mean,
    dist=rng.normal,
    adjust=epic_adjustment,
    constants=epic_constants)