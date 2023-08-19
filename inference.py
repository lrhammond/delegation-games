import numpy as np
import measures

def alignment_estimate(u, u_hat, metric):
    
    n = len(u)
    u_array = [np.array(list(u_i.values())) for u_i in u]
    u_nu = [metric.normalise(u_a) if len(u_a) > 1 else u_a for u_a in u_array]
    u_m = [metric.m(u_a) for u_a in u_array]

    u_hat_array = [np.array(list(u_hat_i.values())) for u_hat_i in u_hat]
    u_hat_nu = [metric.normalise(u_hat_a) if len(u_hat_a) > 1 else u_hat_a for u_hat_a in u_hat_array]

    ia = [measures.individual_alignment(u_nu[i], u_hat_nu[i], metric) for i in range(n)]
    ca = measures.collective_alignment(u_nu, u_m, metric)

    return ia, ca

def capabilities_estimate(w, br, d):

    n = len(br)

    if d == "played":

        ic = np.ones(n)
        cc = w["min"] / w["max"]

    elif d == "eps_NEs":

        ic = []
        for i in range(n):
            ic += [min([br[i][k]["min"]/br[i][k]["max"] for k in br[i]])]
        ic = np.array(ic)
        cc = 1.0

    elif d == "NEs":

        ic = np.ones(n)
        cc = 1.0

    elif d == "all":

        ic = np.ones(n)
        cc = 1.0

    return ic, cc
            
def get_approx_NEs(S, u, eps, tol=1e-10):

    n = len(u)
    for s in S:
        NE = True
        for i in range(n):
            k = tuple(s[:i] + s[i+1:])
            u_min = min(u[i][k])
            u_max = max(u[i][k])
            if u[i][k][s[i]] + tol < u_min + (1 - eps[i]) * (u_max - u_min):
                break
            if u[i][k][s[i]] + tol < u_max:
                NE = False
            if i == n - 1:
                eps_NEs += [s]
                if NE:
                    NEs += [s]
    
    return NEs, eps_NEs
    