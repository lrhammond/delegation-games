import nashpy as nash
import numpy as np
import itertools
import measures

def average_utilitarian(u):

    return sum(u) / len(u)

class UtilityFunction:

    def __init__(self, u_nu, m, c, decimals=5):
        self.nu = np.around(u_nu, decimals=decimals)
        self.m = np.around(m, decimals=decimals)
        self.c = np.around(c, decimals=decimals)
    
    def __call__(self, s):
        return self.m * self.nu[*s] + self.c
    
    def flatten(self, normalised=False):
        flat = self.nu.flatten()
        return flat if normalised else self.m * flat + self.c
    
class DelegationGame:

    def __init__(self, u, u_hat):
        self.u = u
        self.u_hat = u_hat
        self.dims = u[0].nu.shape
        self.S = tuple(itertools.product(*[range(d) for d in self.dims]))
        self.size = np.prod(self.dims)
        self.n = len(self.dims)

    def ia(self, metric):

        return np.array([measures.individual_alignment(self.u[i].nu, self.u_hat[i].nu, metric) for i in range(self.n)])

    def ca(self, metric, agents=False):
        
        utilities = self.u if agents else self.u_hat
        u_nu = [u_i.nu for u_i in utilities]
        u_m = [u_i.m for u_i in utilities]
        
        return measures.collective_alignment(u_nu, u_m, metric)
    
    def get_pure_eps_NEs(self, eps=[], tol=1e-10):
        
        if len(eps) == 0:
            eps = [0.0 for _ in self.dims]
        NEs = []
        for s in self.S:
            for i in range(self.n):
                s_i = self.u[i].m * self.u[i].nu[*s[:i],:,*s[i+1:]] + self.u[i].c
                u_min = np.min(s_i)
                u_max = np.max(s_i)
                if self.u[i](s) + tol < u_min + (1 - eps[i]) * (u_max - u_min):
                    break
                if i == self.n - 1:
                    NEs += [s]
                    
        return NEs
    
    def w(self, s, welfare=average_utilitarian):
        
        return welfare([u_i(s) for u_i in self.u])
    
    def w_hat(self, s, welfare=average_utilitarian):
        
        return welfare([u_hat_i(s) for u_hat_i in self.u_hat])
    
    def w_bounds(self, welfare=average_utilitarian):

        w_minus = welfare([self.u[i].m * np.min(self.u[i].nu) + self.u[i].c for i in range(self.n)])
        w_plus = welfare([self.u[i].m * np.max(self.u[i].nu) + self.u[i].c for i in range(self.n)])

        return w_minus, w_plus
    
    def w_hat_bounds(self, welfare=average_utilitarian):

        w_hat_minus = welfare([self.u_hat[i].m * np.min(self.u_hat[i].nu) + self.u_hat[i].c for i in range(self.n)])
        w_hat_plus = welfare([self.u_hat[i].m * np.max(self.u_hat[i].nu) + self.u_hat[i].c for i in range(self.n)])

        return w_hat_minus, w_hat_plus
    
    def get_played_strategies(self, NEs, eps_NEs, cc, tol=1e-6):
        
        w_0 = min([self.w(s) for s in NEs])
        w_eps = min([self.w(s) for s in eps_NEs])
        w_star = max([self.w(s) for s in self.S])

        w_min = w_eps + cc * (w_star - w_0)
        w_max = w_eps + (w_star - w_0)
        w_mean = 0.5 * (w_max + w_min)

        played = []
        closest_dist = 10e6 

        for s in self.S:
            w_s = self.w(s)
            if w_s + tol > w_min and w_s - tol < w_max:
                played += [s]
            elif abs(w_mean - w_s) < closest_dist:
                closest = s
                closest_dist = abs(w_mean - w_s)
        
        return [closest] if played == [] else played
    
    def print(self):

        print("\n--- Agents ---\n")
        for i in range(self.n):
            print("Agent {}:".format(i))
            print(self.u[i].m * self.u[i].nu + self.u[i].c)
        # print("\n--- Principals ---\n")
        # for i in range(self.n):
        #     print("Principal {}:".format(i))
        #     print(self.u_hat[i].m * self.u_hat[i].nu + self.u_hat[i].c)
    
def generate_delegation_game(dims, ia, ca, metric, rng, agents_first=False, m_range=(0.5,1.5), c_range=(-1.0,1.0)):

    size = np.prod(dims)
    n = len(dims)

    adjusted_u_nu = None

    k =  0

    while adjusted_u_nu == None:

        # print(k)
        u_nu = [metric.sample(size) for _ in dims]
        u_m = [rng.uniform(*m_range) for _ in dims]
        adjusted_u_nu = metric.adjust_ca(u_nu, u_m, ca, metric)
        k += 1
    
    u_nu = adjusted_u_nu
    u_c = [rng.uniform(*c_range) for _ in dims]

    u_hat_nu = []
    for i in range(n):

        u_hat_i_nu = metric.sample(size)
        u_hat_nu += [metric.adjust_ia(u_nu[i], u_hat_i_nu, ia[i])]

    u_hat_m = [rng.uniform(*m_range) for _ in dims]
    u_hat_c = [rng.uniform(*c_range) for _ in dims]

    u = [UtilityFunction(u_nu[i].reshape(dims), u_m[i], u_c[i]) for i in range(n)]
    u_hat = [UtilityFunction(u_hat_nu[i].reshape(dims), u_hat_m[i], u_hat_c[i]) for i in range(n)]

    return DelegationGame(u, u_hat) if agents_first else DelegationGame(u_hat, u)