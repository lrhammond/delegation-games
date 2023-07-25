import numpy as np

from typing import Callable, NamedTuple

class StandardisationConstants(NamedTuple):
    '''
    Constants for a choice of standardisation
    '''
    K_m:float
    K_d:float

Standardisation = Callable[[np.ndarray], np.ndarray]
'''Passed a utility function array, returns standardised equivalent'''

StandardisedDistance = Callable[[np.ndarray, np.ndarray], float]

class AlignmentMetric(NamedTuple):
    standardise:Standardisation
    standardised_distance:StandardisedDistance
    constants:StandardisationConstants
    name:str

def distance(am:AlignmentMetric, u:np.ndarray, v:np.ndarray) -> float:
    u_s = am.standardise(u)
    v_s = am.standardise(v)
    return am.standardised_distance(u_s, v_s)

def epic_standardisation(u:np.ndarray) -> np.ndarray:
    u_flat = u.flatten()
    u_centred = u_flat - u_flat.mean()
    return u_centred / np.linalg.norm(u_centred)

def epic_distance(u:np.ndarray, v:np.ndarray) -> float:
    return np.linalg.norm(u - v) / 2

EPIC_CONSTANTS = StandardisationConstants(K_m=np.sqrt(2), K_d=2)
'''
Measuring alignment using EPIC (the version with uniform distribution, 2-norm, scaled to range 0-1).

K_m is the largest absolute utility difference assignable by a 2-standardised utility function.
- all points apart from 2 are at mean
- those two are at opposites: distance sqrt(1/2) from mean

K_d is the largest ratio L^\infty can have to the (0-1 scaled) EPIC 2-norm
- EPIC distance is half of the 2-norm distance
- 2-norm is a tight lower-bound for \infty-norm
'''

EPIC = AlignmentMetric(standardise=epic_standardisation, standardised_distance=epic_distance, constants=EPIC_CONSTANTS, name='EPIC')

def max_standardisation(u:np.ndarray) -> np.ndarray:
    u_flat = u.flatten()
    u_min = u_flat.min()
    u_max = u_flat.max()
    u_centred = u - (u_max+u_min)/2
    return 2 * u_centred / (u_max-u_min)

def max_distance(u:np.ndarray, v:np.ndarray) -> float:
    return np.abs(u-v).max()

MAX_CONSTANTS = StandardisationConstants(K_m=2, K_d=1)
'''
Measuring alignment using min-max scaled standard utilities and L^\infty norm for distance.

K_m is the largest absolute utility difference assignable by a \infty-standardised utility function.
- min at -1, max at 1 - this is true for all standardised utility functions
'''

MAX = AlignmentMetric(standardise=max_standardisation, standardised_distance=max_distance, constants=MAX_CONSTANTS, name='MAX')
