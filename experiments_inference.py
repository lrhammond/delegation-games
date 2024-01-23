import numpy as np
from games import generate_delegation_game
import measures
import pandas as pd
from tqdm import tqdm
import os
import pickle
import experiments_plotting_utils
import inference
from experiments_common_params import SIZES, SEED

def run_one(dims, repetitions, rng, dists=["eps_NEs","all","played","NEs"], samples=1000, increments=100, measure="EPIC", force_m=False, force_c=False, name=""):

    # samples = len(game_sizes) * repetitions * increments
    # i = 0
    if measure == "EPIC":
        metric = measures.epic(rng)
    
    entries = dict([(d,[]) for d in dists])
    n = len(dims)

    for r in range(repetitions):

        code = "{}-{}-{}".format("x".join(map(str,dims)), r, name)
        gname = "exp2/games/{}.pickle".format(code)

        # If we've already generated the game, skip this step; see exp_1 for explanation of the game generation process
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
            os.makedirs(os.path.dirname(gname), exist_ok=True)
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
        os.makedirs(os.path.dirname(sname), exist_ok=True)
        with open(sname, 'wb') as handle:
            pickle.dump(rs, handle)

        # We can measure the various metrics over different distributions of strategies
        for d in dists:

            rng.set_state(rs)

            # Set up dicts for key quantities (see paper for definitions)
            u = [{} for _ in dims]
            u_hat = [{} for _ in dims]
            br = [{} for _ in dims]
            w = {}
            m = len(strategies[d])
            step = np.floor_divide(samples, increments)

            for j in range(samples):
                
                # Sample a strategy uniformly from the support of the given distribution (i.e. the set of strategies under consideration)
                index = rng.randint(low=0, high=m)
                s = strategies[d][index]
                w[s] = G.w(s)
                w["max"] = max(w.get("max", -10e20), w[s])
                w["min"] = min(w.get("min", 10e20), w[s])

                # Figure out what the (pure) best response sets are for each player
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
                    
                    # Estimate the four measures using the inference procedures defined in the paper
                    ia, ca = inference.alignment_estimate(u, u_hat, metric)
                    ic, cc = inference.capabilities_estimate(w, br, d)

                    # Calculate the losses based on the true measures
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

def run_experiments_inference(sizes=SIZES[:2], dists=["eps_NEs","all","played","NEs"], repetitions=100, samples=1000, increments=100, seed=SEED, progress_bar=False,force_m=False, force_c=False, name=""):

    rng = np.random.RandomState(seed)

    exp_2_combinations = tqdm(sizes, total=len(sizes)) if progress_bar else sizes

    for dims in exp_2_combinations:
        
        data = run_one(dims, repetitions, rng, dists=dists, samples=samples, increments=increments, measure="EPIC", force_m=force_m, force_c=force_c, name=name)

        for d in dists:

            code = "{}-{}-{}".format("x".join(map(str,dims)), d, name)
            fname = "exp2/data/{}.csv".format(code)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            data[d].to_csv(fname)

    experiments_plotting_utils.plot_exp_2(sizes, dists, name)
