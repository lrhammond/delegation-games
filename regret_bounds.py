import numpy as np
import pandas as pd
from typing import Callable, List, TypeVar, Union

SimpleBound = Union[Callable[[float, float], float], Callable[[pd.Series, pd.Series], pd.Series]]
'''
Simple regret bound taking agents' welfare regret and total misalignment as arguments,
determined for a particular welfare calibration, but otherwise underdetermined (a different bound applies each time)

Assume total misalignment has been appropriately weighted according to principal weights.
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

def bound_principals_welfare_regret_miscalibrated(
        ms:np.ndarray,
        K_m:float,
        rs:np.ndarray=None,
        K_d=2,
        align_mul=2,
        cap=2*np.sqrt(2)) -> SimpleBound:
    '''
    Miscalibrated case (welfare calibration ratios provided)

    Principals' welfare regret bounded by combined
    - agent welfare regret (cooperation) weighted by miscalibration
    - agent misalignment (from principals) weighted by calibration ratios
    - welfare miscalibration adjustment

    If not provided, ASSUME principal weights are all 1

    :param ms: list of agent weights
    :param K_m: maximum difference a standardised utility assigns to two strategies, constant determined by standardisation
    :param rs: list of principal weights
    :param K_d: maximum ratio of L^\infty to alignment distance, constant determined by standardisation
    :param align_mul: multiplier for total alignment, default to 2
    :param cap: trivial upper bound based on range of principals' utilities
    '''
    ms = np.array(ms)
    if rs is None:
        rs = 1/ms # assume principal weights are all 1

    def bound(welfare_regret, total_misalignment, R=None, return_all_bounds=False):
        '''
        :param welfare_regret: welfare loss vs perfect agent welfare
        :param total_misalignment: aggregate misalignment distance of agents from principals
        :param R: the gross welfare ratio to use; if not provided, an optimal ratio is computed
        '''
        welfare_regret = np.array(welfare_regret)
        total_misalignment = np.array(total_misalignment)

        if R is None:
            R = np.concatenate([[0], rs])
        else:
            R = np.reshape(R, (-1,))

        C = get_miscalibration_adjustment(ms, R, rs)

        uncapped_bounds = np.multiply.outer(R, welfare_regret) + align_mul*K_d*total_misalignment + K_m * C.reshape((-1,)+(1,)*welfare_regret.ndim)
        all_bounds = np.concatenate([
            uncapped_bounds,
            np.broadcast_to(cap, (1,) + uncapped_bounds.shape[1:])
        ])

        if return_all_bounds:
            return all_bounds
        else:
            return np.min(all_bounds, axis=0)
    return bound

T = TypeVar('T', float, np.ndarray)

def get_miscalibration_adjustment(ms:np.ndarray, R:T, rs:np.ndarray=None) -> T:
    '''
    Get the aggregate miscalibration adjustment factor C for given agent weights, assuming unit principal weights for comparison.
    '''
    if rs is None:
        rs = 1/ms
    C = sum(np.abs(R-r)*m for r, m in zip(rs, ms))
    return C
