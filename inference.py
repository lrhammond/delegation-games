import numpy as np
import measures

# Estimate IA and CA given a set of payoff samples (see paper for more details)
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

# Estimate IC and CC given samples from set d (e.g. approximate pure NEs), the pure best response sets for the players, and recorded min and max welfares
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