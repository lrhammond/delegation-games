from nashpy import Game
import numpy as np
from typing import NamedTuple

import games
import geom


class FlatGame(NamedTuple):
    '''
    Flat representation of 2-player game; arrays intended to be 1d payoff functions
    '''
    first:np.ndarray
    second:np.ndarray

class DelegationGame(NamedTuple):
    principals:FlatGame
    agents:FlatGame

def game_to_flatgame(g:Game) -> FlatGame:
    return FlatGame(*(a.flatten() for a in g.payoff_matrices))

def flatgame_to_game(g:FlatGame) -> Game:
    return Game(*(a.reshape((2, 2)) for a in g))

def generate_delegation_games():
    '''
    Yield infinite stream of uniformly-generated standardised delegation games
    '''
    def payoff():
        return geom.standardise(games.uniform_2x2_payoff().flatten())
    while True:
        yield DelegationGame(FlatGame(payoff(), payoff()), FlatGame(payoff(), payoff()))

def generate_delegation_games_with_alignment_bounds(max_epic=.2, min_epic=.0):
    '''
    Yield infinite stream of delegations games where
    - principals are uniformly-generated
    - agents are precisely randomly misaligned within given range
    '''
    def payoff():
        return geom.standardise(games.uniform_2x2_payoff().flatten())
    while True:
        epics = np.random.uniform(min_epic, max_epic, 2)
        principals = FlatGame(payoff(), payoff())
        agents = FlatGame(
            geom.random_epic_distance_step(principals[0], epics[0]),
            geom.random_epic_distance_step(principals[1], epics[1]))
        yield DelegationGame(principals, agents)
