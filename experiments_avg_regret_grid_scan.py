import numpy as np
from games import generate_delegation_game
import measures
import pandas as pd
from tqdm import tqdm
import itertools
import os
import pickle
import experiments_plotting_utils
from experiments_common_params import SIZES, SEED

VARIABLES = ["ia", "ic", "ca", "cc"]
OTHERS = [0.0, 0.25, 0.5, 0.75, 1.0]

def run_one(dims, repetitions, increments, variable, rng, v_others=1.0, measure="EPIC", force_m=False,force_c=False, name=""):

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
    
        for r in range(repetitions):

            # IC and IA are vectors (one value for each player)
            if variable == "ic" or variable == "ia":
                x[variable] = v * np.ones(n)
            else:
                x[variable] = v

            code = "{}-{}-{}-{}-{}-{}".format(variable, v, "x".join(map(str,dims)), v_others, r, name)
            gname = "exp1/games/{}.pickle".format(code)

            # Avoids creating a new game every time
            if not os.path.exists(gname):
            # if True:

                # Generates a game with the desired characteristics where the set of pure approximate NEs is non-empty (taking into account some additional tolerance) 
                eps_tol = np.zeros(n)
                NEs = []
                attempts = 0
                # When generating utility functions, we can force their norm and how off centre they are to lie within particular ranges
                m_range = (1.0,1.0) if force_m else (0.5,1.5)
                c_range = (0.0,0.0) if force_c else (-1.0,1.0)
                while NEs == []:
                    G = generate_delegation_game(dims, x["ia"], x["ca"], metric, rng, m_range=m_range,c_range=c_range)
                    attempts += 1
                    NEs = G.get_pure_eps_NEs(eps=eps_tol)
                    # If we generate 5 random games and none of them have pure approximate NEs, increase the tolerance slightly for what counts as a pure approximate NE
                    if attempts == 5:
                        attempts = 0
                        eps_tol += (0.01 * np.ones(n))
                        if eps_tol[0] > 1 - x["ic"][0]:
                            x["ic"] = np.ones(n) - eps_tol

                # If IC is roughly 1, then the eps_NEs are the NEs
                if np.allclose(x["ic"],np.ones(n)-eps_tol): 
                    eps_NEs = NEs
                # If IC is roughly 0, then the eps_NEs are are all strategies
                elif np.allclose(x["ic"],np.zeros(n)):
                    eps_NEs = G.S
                else:
                    eps_NEs = G.get_pure_eps_NEs(eps=np.ones(n)-x["ic"])
                
                # Given the degree of cooperative capabilities, only some strategies will actually be played
                played = G.get_played_strategies(NEs, eps_NEs, x["cc"])

                strategies = {"NEs":NEs, "eps_NEs":eps_NEs, "played":played}

                # Save all this info for later
                os.makedirs(os.path.dirname(gname), exist_ok=True)
                with open(gname, 'wb') as handle:
                    pickle.dump({"measures": x, "game": G, "strategies": strategies}, handle)

            else:
            
                with open(gname, 'rb') as handle:
                    saved = pickle.load(handle)
                    G = saved["game"]
                    x = saved["measures"]
                    played = saved["strategies"]["played"]
                    NEs = saved["strategies"]["NEs"]
                    eps_NEs = saved["strategies"]["eps_NEs"]

            # Compute the various welfare metrics (see paper for definitions)
            w_hat = [G.w_hat(s) for s in G.S]
            w_hat_max = max(w_hat)
            w_hat_min = min(w_hat)
            w_hat_minus, w_hat_plus = G.w_hat_bounds()
            w_hat_avg = sum([G.w_hat(s) for s in played]) / len(played)
            
            # As above
            u_avg = [sum([G.u[i](s) for s in played])/len(played) for i in range(G.n)]
            s_hat_star = G.S[w_hat.index(max(w_hat))]

            # As above
            max_regret, ideal_regret = measures.get_bounds(G, metric, x["cc"], NEs, eps_NEs, s_hat_star, u_avg)

            # The game's precise metrics may differ very slightly from the chosen metrics, so we calculate them here to be sure
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

def run_experiments_avg_regret(sizes=SIZES[:2], variables=VARIABLES, others=OTHERS, repetitions=10, increments=20, seed=SEED, progress_bar=False,force_m=False, force_c=False, name=""):

    # For each run we initialise a random number generator using the seed, and then continually make use of this and save its state for each trial, to enable replicability
    rng = np.random.RandomState(seed)

    exp_1_combinations = list(itertools.product(sizes, variables, others))
    if progress_bar:
       exp_1_combinations = tqdm(exp_1_combinations, total=len(exp_1_combinations)) 

    for (dims, variable, v_others) in exp_1_combinations:
        
        code = "{}-{}-{}-{}".format(variable, "x".join(map(str,dims)), v_others, name)
        fname = "exp1/data/{}.csv".format(code)

        # If we've already generated the data and plotted it, skip this step
        if not os.path.exists(fname):
        # if True:

            sname = "exp1/random_states/{}.pickle".format(code)
            rs = rng.get_state()
            os.makedirs(os.path.dirname(sname), exist_ok=True)
            with open(sname, 'wb') as handle:
                pickle.dump(rs, handle)
            rng.set_state(rs)

            data = run_one(dims, repetitions, increments, variable, rng, v_others=v_others, force_m=force_m, force_c=force_c, name=name)
            os.makedirs(os.path.dirname(fname), exist_ok=True)
            data.to_csv(fname)

            experiments_plotting_utils.plot_exp_1(dims, variable, v_others, name, bounds=True)
