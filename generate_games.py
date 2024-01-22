from nashpy import Game
import numpy as np
from typing import List, NamedTuple
import geom
from alignment import EPIC


class FlatGame(NamedTuple):
    payoffs:List[np.ndarray]

class DelegationGame(NamedTuple):
    principals:FlatGame
    agents:FlatGame

def game_to_flatgame(g:Game) -> FlatGame:
    return FlatGame([a.flatten() for a in g.payoff_matrices])

def flatgame_to_game(g:FlatGame) -> Game:
    '''
    Convert a 4d FlatGame into a 2x2 Game representation

    NB ONLY WORKS for 4d
    '''
    return Game(*(a.reshape((2, 2)) for a in g.payoffs))

def generate_delegation_games(rng, n_players=2, n_outcomes=4, m=None, am=EPIC):
    '''
    Yield infinite stream of uniformly-generated standardised delegation games
    '''
    if m is None:
        m = [1 for _ in range(n_players)]
    def payoff():
        return am.standardise(rng.uniform(size=n_outcomes))
    while True:
        principals = FlatGame([payoff() for _ in range(n_players)])
        agents = FlatGame([m[i]*payoff() for i in range(n_players)])
        yield DelegationGame(principals, agents)

def generate_delegation_games_with_alignment_bounds(rng, n_players=2, n_outcomes=4, m=None, max_epic=.2, min_epic=.0, am=EPIC):
    '''
    Yield infinite stream of delegations games where
    - principals are uniformly-generated
    - agents are precisely randomly misaligned within given range
    '''
    if m is None:
        m = [1 for _ in range(n_players)]
    def payoff():
        return am.standardise(rng.uniform(size=n_outcomes))
    while True:
        total_misalignment = rng.uniform(min_epic, max_epic) * n_players
        epics_unadjusted = total_misalignment * geom.sample_simplex(rng, n_players)
        # simplex sample can produce impossibly-large misalignment for larger shares (i.e. > 1)
        # geom currently just truncates to max possible misalignment
        # mixing in a flat 1/n share of the total misalignment resolves this
        # for d+ the max unadjusted, the coefficient must be less than (n-1)/(nd+-1)
        greatest_epic_unadjusted = np.max(epics_unadjusted)
        if greatest_epic_unadjusted > 1:
            coeff_max = (n_players - 1) / (n_players*greatest_epic_unadjusted - 1)
            coeff = rng.uniform(0, coeff_max)
            epics = coeff * epics_unadjusted + (1-coeff) / n_players
        else:
            epics = epics_unadjusted
        principals = FlatGame([payoff() for _ in range(n_players)])
        agents = FlatGame(
            [m[i]*am.standardise(geom.random_epic_distance_step(principals.payoffs[i], epics[i], rng)) for i in range(n_players)]
        )
        yield DelegationGame(principals, agents)
