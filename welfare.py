from typing import Callable, List, Sequence, Union
import nashpy as nash
import numpy as np

Welfare = Union[Callable[[Sequence[np.ndarray]], np.ndarray], Callable[[Sequence[float]], float]]

# we assume games are 2x2 for now (TODO: update this)
pure_strategies_2b2 = [[0, 1], [1, 0]]

def price_of_anarchy(game:nash.Game):
    '''
    DEPRECATED
    '''
    # nash equilibria
    eqs = list(game.support_enumeration())

    # list of expected utilities for both players for pure strategies
    pure_outcomes = [game[s_row, s_col]
                     for s_row in pure_strategies_2b2 for s_col in pure_strategies_2b2]

    # list of expected utilities for both players in nash equilibria
    equil_outcomes = [game[s_row, s_col]
                      for (s_row, s_col) in eqs]

    # we only need to check pure strategies because the maximum is
    # necessarily among these
    # maximum expected payoff strategy
    max_strat = np.argmax(list(map(sum, pure_outcomes)))

    # maximum expected payoff nash equil
    min_in_equil = np.argmin(list(map(sum, equil_outcomes)))

    return sum(pure_outcomes[max_strat]) / sum(equil_outcomes[min_in_equil])


def welf_regret(game:nash.Game, welfare:Welfare=sum) -> float:
    '''
    DEPRECATED (subsumed by general)
    '''
    # nash equilibria
    eqs = list(game.support_enumeration())

    # list of expected utilities for both players for pure strategies
    pure_welfares = [welfare(game[s_row, s_col])
                     for s_row in pure_strategies_2b2 for s_col in pure_strategies_2b2]

    # list of expected utilities for both players in nash equilibria
    equil_welfares = [welfare(game[s_row, s_col])
                      for (s_row, s_col) in eqs]

    # we only need to check pure strategies because the maximum is
    # necessarily among these
    max_strat = np.max(pure_welfares)

    min_in_equil = np.min(equil_welfares)

    return max_strat - min_in_equil


def princ_welf_regret(principal_game:nash.Game, agents_game:nash.Game, welfare:Welfare=sum) -> float:
    '''
    DEPRECATED (subsumed by general)
    '''
    # nash equilibria
    eqs = list(agents_game.support_enumeration())

    # list of expected utilities for both PRINCIPALS for pure strategies
    pure_welfares = [welfare(principal_game[s_row, s_col])
                     for s_row in pure_strategies_2b2 for s_col in pure_strategies_2b2]

    # list of expected utilities for both PRINCIPALS in nash equilibria of AGENT GAME
    equil_welfares = [welfare(principal_game[s_row, s_col])
                      for (s_row, s_col) in eqs]

    # we only need to check pure strategies because the maximum is
    # necessarily among these
    max_strat = np.max(pure_welfares)

    min_in_equil = np.min(equil_welfares)

    return max_strat - min_in_equil

def welfare_regret_general(payoffs:List[np.ndarray], strategy:np.ndarray, welfare:Welfare=sum) -> float:
    '''
    Given payoffs of some players, and a (mixed) strategy,
    compute the welfare regret of the strategy.

    Payoffs should be flat representation,
    and strategy should be a flat representation summing to 1.
    '''
    pure_welfares = welfare(payoffs)
    max_welfare = pure_welfares.max()
    actual_welfare = np.sum(pure_welfares * strategy)
    return max_welfare - actual_welfare
