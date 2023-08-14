import numpy as np
import measures

def inference(u, u_hat, w, eps, metric):
    
    n = len(u)
    u_array = [np.array(u_i.values()) for u_i in u]
    u_nu = [metric.normalise(u_a) for u_a in u_array]
    u_m = [metric.m(u_a) for u_a in u_array]
    u_hat_nu = [metric.normalise(np.array(u_hat_i.values())) for u_hat_i in u_hat]

    ia = [measures.individual_alignment(u_nu[i], u_hat_nu[i], metric) for i in range(n)]
    ca = measures.collective_alignment(u_nu, u_m, metric)

    NEs, eps_NEs = get_approx_NEs(w.keys(), u, eps)

    w["zero"] = min([w(s) for s in NEs])
    w["eps"] = min([w(s) for s in eps_NEs])

    

    return ia, ic, ca, cc, w
    
            
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
    