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
    epics = [metrics.epic(p, a) for p, a in zip(principals.payoffs, agents.payoffs)]
    principals_game = flatgame_to_game(principals)
    agents_game = flatgame_to_game(agents)
    welfare_regret = metrics.welf_regret(agents_game)
    principals_welfare_regret = metrics.princ_welf_regret(principals_game, agents_game)
    return SimpleStat(sum(epics), welfare_regret, principals_welfare_regret)
