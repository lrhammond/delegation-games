from nashpy import Game
import numpy as np
from typing import List, NamedTuple
from utils import rs

import games
import geom
from alignment import AlignmentMetric, EPIC
from statistical import sample_simplex


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
        # TODO simplex sample can produce impossibly-large misalignment for larger shares
        # geom currently just truncates to max possible misalignment
        # could instead mix the simplex sample with a uniform share of misalignment depending on how large total is
        epics = total_misalignment * sample_simplex(n_players)
        principals = FlatGame([payoff() for _ in range(n_players)])
        agents = FlatGame(
            [m[i]*am.standardise(geom.random_epic_distance_step(principals.payoffs[i], epics[i])) for i in range(n_players)]
        )
        yield DelegationGame(principals, agents)
