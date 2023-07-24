import numpy as np
from typing import NamedTuple

from generate_games import flatgame_to_game, DelegationGame
import metrics

SimpleStat = NamedTuple('Stat', total_misalignment=float, welfare_regret=float, principals_welfare_regret=float)
'''
Simple stat over a delegation game, descriptive in fully welfare-calibrated games
'''

def get_stat(dg:DelegationGame) -> SimpleStat:
    principals, agents = dg
    epic_left = metrics.epic(principals[0], agents[0])
    epic_right = metrics.epic(principals[1], agents[1])
    principals_game = flatgame_to_game(principals)
    agents_game = flatgame_to_game(agents)
    welfare_regret = metrics.welf_regret(agents_game)
    principals_welfare_regret = metrics.princ_welf_regret(principals_game, agents_game)
    return SimpleStat(epic_left+epic_right, welfare_regret, principals_welfare_regret)
