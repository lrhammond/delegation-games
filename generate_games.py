from nashpy import Game
import numpy as np
from typing import List, NamedTuple

import games
import geom
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

def generate_delegation_games(n_players=2, n_outcomes=4, m=None):
    '''
    Yield infinite stream of uniformly-generated standardised delegation games
    '''
    if m is None:
        m = [1 for _ in range(n_players)]
    def payoff():
        return geom.standardise(np.random.uniform(size=n_outcomes))
    while True:
        principals = FlatGame([payoff() for _ in range(n_players)])
        agents = FlatGame([m[i]*payoff() for i in range(n_players)])
        yield DelegationGame(principals, agents)

def generate_delegation_games_with_alignment_bounds(n_players=2, n_outcomes=4, m=None, max_epic=.2, min_epic=.0):
    '''
    Yield infinite stream of delegations games where
    - principals are uniformly-generated
    - agents are precisely randomly misaligned within given range
    '''
    if m is None:
        m = [1 for _ in range(n_players)]
    def payoff():
        return geom.standardise(np.random.uniform(size=n_outcomes))
    while True:
        epics = np.random.uniform(min_epic, max_epic) * sample_simplex(n_players) * n_players
        principals = FlatGame([payoff() for _ in range(n_players)])
        agents = FlatGame(
            [m[i]*geom.random_epic_distance_step(principals.payoffs[i], epics[i]) for i in range(n_players)]
        )
        yield DelegationGame(principals, agents)
