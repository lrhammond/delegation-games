import numpy as np
import pandas as pd
from typing import Callable, Union

SimpleBound = Union[Callable[[float, float], float], Callable[[pd.Series, pd.Series], pd.Series]]
'''
Simple regret bound taking agents' welfare regret and total misalignment as arguments, descriptive in fully welfare-calibrated games
'''

def bound_principals_welfare_regret_simple(mul=4, cap=2*np.sqrt(2)) -> SimpleBound:
    '''
    Simplest case (perfect welfare calibration)

    Principals' welfare regret bounded by linear combo of
    - agent welfare regret (cooperation)
    - agent misalignment (from principals)
    '''
    def bound(welfare_regret, total_misalignment):
        return np.minimum(welfare_regret + mul*total_misalignment, cap)
    return bound
