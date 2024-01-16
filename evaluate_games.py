import numpy as np
from typing import List, NamedTuple, Tuple

from generate_games import flatgame_to_game, DelegationGame, FlatGame
from alignment import distance as alignment_distance, AlignmentMetric
from welfare import welf_regret, princ_welf_regret, welfare_regret_general, Welfare
from geom import sample_simplex

SimpleStat = NamedTuple('Stat', total_misalignment=float, welfare_regret=float, principals_welfare_regret=float)
'''
Simple stat over a delegation game, descriptive in fully welfare-calibrated games
'''

def get_stat_nash(dg:DelegationGame, am:AlignmentMetric) -> SimpleStat:
    '''
    Use 'worst agent Nash' solution concept (i.e. 'anarchy')

    NB requires 2-agent in current setup
    '''
    principals, agents = dg
    epics = [alignment_distance(am, p, a) for p, a in zip(principals.payoffs, agents.payoffs)]
    principals_game = flatgame_to_game(principals)
    agents_game = flatgame_to_game(agents)
    welfare_regret = welf_regret(agents_game)
    principals_welfare_regret = princ_welf_regret(principals_game, agents_game)
    return SimpleStat(sum(epics), welfare_regret, principals_welfare_regret)

def get_stat_general(dg:DelegationGame, max_welfare_regret:float, am:AlignmentMetric, rng, min_welfare_regret=0., use_agents=True) -> SimpleStat:
    '''
    Produce a randomly-selected strategy with full support and welfare regret in the given range, if feasible

    :param dg: delegation game to get stats for
    :param max_welfare_regret: upper target welfare regret bound
    :param min_welfare_regret: lower target welfare regret bound
    :param use_agent: if True, target agents' welfare regret, otherwise target principals' welfare regret
    '''
    principals, agents = dg
    epics = [alignment_distance(am, p, a) for p, a in zip(principals.payoffs, agents.payoffs)]
    target_payoffs = agents if use_agents else principals
    strat = get_strategy_with_welfare_regret(target_payoffs, target_welfare_regret=rng.uniform(min_welfare_regret, max_welfare_regret))
    welfare_regret = welfare_regret_general(agents.payoffs, strat)
    principals_welfare_regret = welfare_regret_general(principals.payoffs, strat)
    return SimpleStat(sum(epics), welfare_regret, principals_welfare_regret)

def optimal_pessimal_welfare_strategies(payoffs:List[np.ndarray], welfare:Welfare=sum) -> Tuple[np.ndarray, np.ndarray]:
    '''
    For given welfare aggregation and payoffs, return the best and worst strategies available
    '''
    pure_welfares = welfare(payoffs)
    max_welfare = pure_welfares.max()
    min_welfare = pure_welfares.min()
    ismax = pure_welfares == max_welfare
    ismin = pure_welfares == min_welfare
    return ismax / ismax.sum(), ismin / ismin.sum()

def sample_strategy(n_strats:int):
    '''
    Randomly uniformly sample a mixed strategy over the given number of options
    '''
    return sample_simplex(n_strats)

def get_strategy_with_welfare_regret(game:FlatGame, target_welfare_regret:float):
    '''
    For given game, generate a strategy with specified welfare regret.

    If target is unfeasibly high, return pessimal strategy with worst possible welfare regret.

    TODO this is where the strategy is fully correlated, rather than marginal per player
    TODO to change that, the FlatGame would need to contain more structure, and the search for a strategy would be more complicated to respect that structure
    NB the strategy does not take into account game structure, so it may be correlated/entirely mixed over outcomes.

    :param game: the game with payoffs
    :param target_welfare_regret: the welfare regret to target
    '''
    d_u = len(game.payoffs[0])

    # convex combinations of strategies produce the same combination of expected welfare
    # optimal strategy produces zero regret; pessimal strategy produces some upper bound
    # we can achieve a target welfare regret by finding the appropriate convex combination
    # to produce more 'off-centre' strategies, we can instead interpolate with a randomly sample intermediate strategy
    optimal_strat, pessimal_strat = optimal_pessimal_welfare_strategies(game.payoffs)
    pessimal_regret = welfare_regret_general(game.payoffs, pessimal_strat)
    if target_welfare_regret > pessimal_regret:
        strat = pessimal_strat
    else:
        sample_strat = sample_strategy(d_u)
        sample_regret = welfare_regret_general(game.payoffs, sample_strat)
        if target_welfare_regret < sample_regret:
            # mix with optimal strategy
            r = target_welfare_regret / sample_regret
            strat = r * sample_strat + (1-r) * optimal_strat
        else:
            # mix with pessimal strategy
            r = (target_welfare_regret-sample_regret) / (pessimal_regret-sample_regret)
            strat = r * pessimal_strat + (1-r) * sample_strat
    return strat
