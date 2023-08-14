import numpy as np
from games import generate_delegation_game
import measures
import pandas as pd
from tqdm import tqdm
import itertools
import os
import pickle
import utils
from inference import infer_measures

# Experiment 1

SIZES =[ (3,3),
         (5,5,4),
         (6,7,4,6),
         (7,3,8,5,12),
         (15,7,3,7,9,5) ]
REPETITIONS = 10
INCREMENTS = 20
VARIABLES = ["ia", "ic", "ca", "cc"]
OTHERS = [0.0, 0.25, 0.5, 0.75, 1.0]
SEED = 1282

def exp_1(dims, repetitions, increments, variable, rng, v_others=1.0, measure="EPIC", force_m=False,force_c=False):

    # samples = len(game_sizes) * repetitions * increments
    # i = 0
    if measure == "EPIC":
        metric = measures.epic(rng)

    values = list(np.linspace(0.00, 1.0, num=increments+1))
    entries = []
    n = len(dims)
    x ={"ic":np.array([v_others for _ in dims]),
        "ia":np.array([v_others for _ in dims]),
        "cc":v_others,
        "ca":v_others}

    for v in values:

        # nwr = []
    
        for _ in range(repetitions):

            if variable == "ic" or variable == "ia":
                x[variable] = v * np.ones(n)
            else:
                x[variable] = v

            eps_tol = np.zeros(n)
            NEs = []
            attempts = 0
            m_range = (1.0,1.0) if force_m else (0.5,1.5)
            c_range = (0.0,0.0) if force_c else (-1.0,1.0)
            while NEs == []:
                G = generate_delegation_game(dims, x["ia"], x["ca"], metric, rng, m_range=m_range,c_range=c_range)
                attempts += 1
                NEs = G.get_pure_eps_NEs(eps=eps_tol)
                if attempts == 10:
                    attempts = 0
                    eps_tol += (0.01 * np.ones(n))
                    if eps_tol[0] > 1 - x["ic"][0]:
                        x["ic"] = np.ones(n) - eps_tol

            if np.allclose(x["ic"],np.ones(n)-eps_tol): 
                eps_NEs = NEs
            elif np.allclose(x["ic"],np.zeros(n)):
                eps_NEs = G.S
            else:
                eps_NEs = G.get_pure_eps_NEs(eps=np.ones(n)-x["ic"])
            
            played = G.get_played_strategies(NEs, eps_NEs, x["cc"])

            w_hat = [G.w_hat(s) for s in G.S]
            w_hat_max = max(w_hat)
            w_hat_min = min(w_hat)
            w_hat_minus, w_hat_plus = G.w_hat_bounds()
            w_hat_avg = sum([G.w_hat(s) for s in played]) / len(played)
            
            u_avg = [sum([G.u[i](s) for s in played])/len(played) for i in range(G.n)]
            s_hat_star = G.S[w_hat.index(max(w_hat))]

            max_regret, ideal_regret = measures.get_bounds(G, metric, x["cc"], NEs, eps_NEs, s_hat_star, u_avg)

            if variable == "ia":
                v_actual = np.mean(G.ia(metric))
            elif variable == "ca":
                v_actual = G.ca(metric)
            elif variable == "ic":
                v_actual = np.mean(x["ic"])
            else:
                v_actual = v

            entries.append((v_actual,w_hat_plus,w_hat_max,w_hat_avg,w_hat_min,w_hat_minus,max_regret,ideal_regret))
            
            # print(r)

            # i += 1
            # wa = sum([G.w_hat(s) for s in G.S]) / len(G.S)
            # wr = (w_hat_max - w_hat_avg) / (w_hat_max - w_hat_min)
            # wl = (w_hat_plus - w_hat_max) / (w_hat_plus - w_hat_min)
            # nwr += [wr]
            # print(round(i/samples,4))

        # print(v, sum(nwr)/repetitions)
    
    return pd.DataFrame(data=entries,
                        columns=["v",
                                "w_hat_plus",
                                "w_hat_max",
                                "w_hat_avg",
                                "w_hat_min",
                                "w_hat_minus",
                                "max_regret",
                                "ideal_regret"])

def run_exp_1(sizes=SIZES[:2], variables=VARIABLES, others=OTHERS, repetitions=10, increments=20, seed=SEED, progress_bar=False,force_m=False, force_c=False, name=""):

    rng = np.random.RandomState(seed)

    exp_1_combinations = list(itertools.product(sizes, variables, others))
    if progress_bar:
       exp_1_combinations = tqdm(exp_1_combinations, total=len(exp_1_combinations)) 

    for (dims, variable, v_others) in exp_1_combinations:
        
        code = "{}-{}-{}-{}".format(variable, "x".join(map(str,dims)), v_others, name)
        fname = "exp1/data/{}.csv".format(code)

        # if not os.path.exists(fname):

        sname = "exp1/random_states/{}.pickle".format(code)
        rs = rng.get_state()
        with open(sname, 'wb') as handle:
            pickle.dump(rs, handle)
        rng.set_state(rs)

        data = exp_1(dims, repetitions, increments, variable, rng, v_others=v_others, force_m=force_m, force_c=force_c)
        data.to_csv(fname)

        utils.plot_exp_1(dims, variable, v_others, name, bounds=True)

def re_run_exp_1(dims, variable, v_others, seed, repetitions=10, increments=20, name=""):

    rng = np.random.RandomState(seed)
    code = "{}-{}-{}-{}".format(variable, "x".join(map(str,dims)), v_others, name)
    fname = "exp1/data/{}.csv".format(code)
    sname = "exp1/random_states/{}.pickle".format(code)
    with open(sname, 'rb') as handle:
        rs = pickle.load(handle)
    rng.set_state(rs)
    data = exp_1(dims, repetitions, increments, variable, rng, v_others=v_others)
    data.to_csv(fname)

def exp_2(dims, repetitions, rng, samples=1000, increments=100, measure="EPIC", force_m=False, force_c=False, name=""):

    # samples = len(game_sizes) * repetitions * increments
    # i = 0
    if measure == "EPIC":
        metric = measures.epic(rng)
    
    entries = []
    n = len(dims)

    for r in range(repetitions):

        code = "{}-{}-{}".format("x".join(map(str,dims)), r, name)
        gname = "exp2/games/{}.pickle".format(code)

        if not os.path.exists(gname):

            x ={"ic":rng.uniform(n),
                "ia":rng.uniform(n),
                "cc":rng.uniform(1),
                "ca":rng.uniform(1)}

            eps_tol = np.zeros(n)
            NEs = []
            attempts = 0
            m_range = (1.0,1.0) if force_m else (0.5,1.5)
            c_range = (0.0,0.0) if force_c else (-1.0,1.0)

            while NEs == []:
                G = generate_delegation_game(dims, x["ia"], x["ca"], metric, rng, m_range=m_range,c_range=c_range)
                attempts += 1
                NEs = G.get_pure_eps_NEs(eps=eps_tol)
                if attempts == 10:
                    attempts = 0
                    eps_tol += (0.01 * np.ones(n))
                    if eps_tol[0] > 1 - x["ic"][0]:
                        x["ic"] = np.ones(n) - eps_tol
        
            with open(gname, 'wb') as handle:
                pickle.dump({"measures": x, "game": G}, handle)

        else:
            
            with open(gname, 'rb') as handle:
                saved = pickle.load(handle)
            G = saved["game"]
            x = saved["measures"]

        if np.allclose(x["ic"],np.ones(n)-eps_tol): 
            eps_NEs = NEs
        elif np.allclose(x["ic"],np.zeros(n)):
            eps_NEs = G.S
        else:
            eps_NEs = G.get_pure_eps_NEs(eps=np.ones(n)-x["ic"])
        
        played = G.get_played_strategies(NEs, eps_NEs, x["cc"])

        sname = "exp2/random_states/{}.pickle".format(code)
        rs = rng.get_state()
        with open(sname, 'wb') as handle:
            pickle.dump(rs, handle)
        rng.set_state(rs)
        
        u = [{} for _ in dims]
        u_hat = [{} for _ in dims]
        w = {}
        w_hat = {}
        eps = [0 for _ in dims]
        delta = 1
        step = np.floor_divide(samples, increments)

        for i in range(samples):

            s = rng.choice(played)
            w[s] = G.w(s)
            w["max"] = max(w.get("max", -10e20), w[s])
            w_hat[s] = G.w_hat(s)

            if "eps" in w and "zero" in w:
                if w["max"] != w["zero"]:
                    delta = min(delta, (w[s] - w["eps"])/(w["max"] - w["zero"]))
                    # w["limit"] = w["eps"] + delta*(w["max"] - w["eps"])
                w["limit"] = min(w.get("limit", 10e20), w["eps"] + (delta*(w["max"] - w["zero"])))
            else:
                w["limit"] = w["max"]

            for i in range(n):

                k = tuple(s[:i] + s[i+1:])
                
                u[i][k][s[i]] = G.u[i](s)
                u[i][k]["u_min"] = min(u[i][k].get("u_min",10e20), u[i][k][s[i]])
                u[i][k]["u_max"] = max(u[i][k].get("u_max",-10e20), u[i][k][s[i]])
                u[i][k]["w_max"] = max(u[i][k].get("w", -10e20), w[s])
                
                if w[s] < u[i][k]["w_max"] and u[i][k]["w_max"] <= w["limit"]:
                    eps[i] = max(eps[i], 1 - ((u[i][k][s[i]] - u[i][k]["u_min"])/(u[i][k]["u_max"] - u[i][k]["u_min"])))

            if i % step == 0 or i == samples-1:

                ia, ca, ic, cc = infer_measures(u, u_hat, w, eps, metric, n)

                ia_loss = np.mean(np.abs(x["ia"] - ia))
                ic_loss = np.mean(np.abs(x["ic"] - ic))
                ca_loss = abs(x["ca"] - ca)
                cc_loss = abs(x["cc"] - cc)

            entries.append((i,ia_loss,ic_loss,ca_loss,cc_loss))

    return pd.DataFrame(data=entries,
                        columns=["samples",
                                "ia_loss",
                                "ic_loss",
                                "ca_loss",
                                "cc_loss",])

# run_exp_1(variables=["ca"],sizes=[(3,3)],others=[0.9],increments=25,repetitions=25,name="aaai")