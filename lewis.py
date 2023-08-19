import numpy as np
from games import generate_delegation_game
import measures
import pandas as pd
from tqdm import tqdm
import itertools
import os
import pickle
import utils
import inference

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
    x ={"ic":v_others*np.ones(n),
        "ia":v_others*np.ones(n),
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

def exp_2(dims, repetitions, rng, dists=["eps_NEs","all","played","NEs"], samples=1000, increments=100, measure="EPIC", force_m=False, force_c=False, name=""):

    # samples = len(game_sizes) * repetitions * increments
    # i = 0
    if measure == "EPIC":
        metric = measures.epic(rng)
    
    entries = dict([(d,[]) for d in dists])
    n = len(dims)

    for r in range(repetitions):

        code = "{}-{}-{}".format("x".join(map(str,dims)), r, name)
        gname = "exp2/games/{}.pickle".format(code)

        if not os.path.exists(gname):

            x ={"ic":rng.uniform(size=n),
                "ia":rng.uniform(size=n),
                "cc":rng.uniform(),
                "ca":rng.uniform()}

            strategies = {"NEs":[], "eps_NEs":[], "played":[]}

            eps_tol = np.zeros(n)
            attempts = 0
            m_range = (1.0,1.0) if force_m else (0.5,1.5)
            c_range = (0.0,0.0) if force_c else (-1.0,1.0)

            while strategies["NEs"] == []:
                G = generate_delegation_game(dims, x["ia"], x["ca"], metric, rng, agents_first=True,m_range=m_range,c_range=c_range,positive=True)
                attempts += 1
                strategies["NEs"] = G.get_pure_eps_NEs(eps=eps_tol)
                if attempts == 10:
                    attempts = 0
                    eps_tol += (0.01 * np.ones(n))
                    if eps_tol[0] > 1 - x["ic"][0]:
                        x["ic"] = np.ones(n) - eps_tol

            if np.allclose(x["ic"],np.ones(n)-eps_tol): 
                strategies["eps_NEs"] = strategies["NEs"]
            elif np.allclose(x["ic"],np.zeros(n)):
                strategies["eps_NEs"] = G.S
            else:
                strategies["eps_NEs"] = G.get_pure_eps_NEs(eps=np.ones(n)-x["ic"])
        
            strategies["played"] = G.get_played_strategies(strategies["NEs"], strategies["eps_NEs"], x["cc"])
            strategies["all"] = G.S

            # G, NEs, eps_NEs, played, x = get_new_game()
        
            with open(gname, 'wb') as handle:
                pickle.dump({"measures": x, "game": G, "strategies": strategies}, handle)

        else:
            
            with open(gname, 'rb') as handle:
                saved = pickle.load(handle)
            G = saved["game"]
            x = saved["measures"]
            strategies = saved["strategies"]

        sname = "exp2/random_states/{}.pickle".format(code)
        rs = rng.get_state()
        with open(sname, 'wb') as handle:
            pickle.dump(rs, handle)

        for d in dists:

            rng.set_state(rs)

            u = [{} for _ in dims]
            u_hat = [{} for _ in dims]
            br = [{} for _ in dims]
            w = {}
            m = len(strategies[d])
            step = np.floor_divide(samples, increments)

            for j in range(samples):
                
                index = rng.randint(low=0, high=m)
                s = strategies[d][index]
                w[s] = G.w(s)
                w["max"] = max(w.get("max", -10e20), w[s])
                w["min"] = min(w.get("min", 10e20), w[s])

                for i in range(n):

                    u[i][s] = G.u[i](s)
                    u_hat[i][s] = G.u_hat[i](s)

                    k = tuple(s[:i] + s[i+1:])
                    if k not in br[i]:
                        br[i][k] = {"max": u[i][s], "min": u[i][s]}
                    else:
                        br[i][k]["min"] = min(br[i][k]["min"], u[i][s])
                        br[i][k]["max"] = max(br[i][k]["max"], u[i][s])
                        # if u[i][s] >= br[i][k].get("max",-10e20):
                        #     br[i][k]["max"] = (s[i], u[i][s])
                        # if u[i][s] <= br[i][k].get("min",10e20):
                        #     br[i][k]["min"] = (s[i], u[i][s])

                if (j % step == 0 and j > 0) or j == samples-1:
                    
                    ia, ca = inference.alignment_estimate(u, u_hat, metric)
                    ic, cc = inference.capabilities_estimate(w, br, d)

                    ia_loss = np.mean(np.abs(x["ia"] - ia))
                    ic_loss = np.mean(np.abs(x["ic"] - ic))
                    ca_loss = abs(x["ca"] - ca)
                    cc_loss = abs(x["cc"] - cc)

                    entries[d].append((j,ia_loss,ic_loss,ca_loss,cc_loss))

    # index_combinations = list(itertools.product(["ia", "ic", "ca", "cc"],dists))
    # index = pd.MultiIndex.from_tuples(index_combinations, names=["var","dist"])

    # df = pd.DataFrame(np.random.randn(3, 8), columns=index

    return dict([(d,pd.DataFrame(data=entries[d],
                                columns=["samples",
                                        "ia_loss",
                                        "ic_loss",
                                        "ca_loss",
                                        "cc_loss"])) for d in dists])

def run_exp_2(sizes=SIZES[:2], dists=["eps_NEs","all","played","NEs"], repetitions=100, samples=1000, increments=100, seed=SEED, progress_bar=False,force_m=False, force_c=False, name=""):

    rng = np.random.RandomState(seed)

    exp_2_combinations = tqdm(sizes, total=len(sizes)) if progress_bar else sizes

    for dims in exp_2_combinations:

        # if not os.path.exists(fname):

        data = exp_2(dims, repetitions, rng, dists=dists, samples=samples, increments=increments, measure="EPIC", force_m=force_m, force_c=force_c, name=name)

        for d in dists:

            code = "{}-{}-{}".format("x".join(map(str,dims)), d, name)
            fname = "exp2/data/{}.csv".format(code)
            data[d].to_csv(fname)

    utils.plot_exp_2(sizes, dists, name)

run_exp_2(sizes=SIZES[:4], repetitions=25, samples=1000, increments=100, force_m=False, force_c=False, name="aaai")

# run_exp_1(variables=["ca"],sizes=[(3,3)],others=[0.9],increments=25,repetitions=25,name="aaai")