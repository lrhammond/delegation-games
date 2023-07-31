import numpy as np

from typing import Callable, NamedTuple

class StandardisationConstants(NamedTuple):
    '''
    Constants for a choice of standardisation
    '''
    K_m:float
    K_d:float

_Shift = Callable[[np.ndarray], float]
'''Retrieve the off-centre affine-equivariant shift of a utility function according to some rule'''

_Scale = Callable[[np.ndarray], float]
'''Retrieve the scale of a centred utility function according to some rule'''

StandardisedDistance = Callable[[np.ndarray, np.ndarray], float]

class AlignmentMetric(NamedTuple):
    shift:_Shift
    scale:_Scale
    standardised_distance:StandardisedDistance
    constants:StandardisationConstants
    name:str

    def standardise(self, u:np.ndarray) -> np.ndarray:
        '''
        Given a utility vector, produce the scaled and shifted standardised representative equivalent to it.
        '''
        centred = u - self.shift(u)
        scale = self.scale(centred)
        return np.where(scale > 0, centred / scale, np.zeros_like(centred))

    def distance(self, u:np.ndarray, v:np.ndarray) -> float:
        '''
        Given two utility vectors, produce the standardised preference distance between them.
        '''
        u_s = self.standardise(u)
        v_s = self.standardise(v)
        return self.standardised_distance(u_s, v_s)

def epic_shift(u:np.ndarray) -> float:
    return u.mean(axis=-1, keepdims=True)

def epic_scale(u:np.ndarray) -> float:
    return np.linalg.norm(u, axis=-1, keepdims=True)

def epic_distance(u:np.ndarray, v:np.ndarray) -> float:
    return np.linalg.norm(u - v, axis=-1) / 2

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

EPIC = AlignmentMetric(shift=epic_shift, scale=epic_scale, standardised_distance=epic_distance, constants=EPIC_CONSTANTS, name='EPIC')

def max_shift(u:np.ndarray) -> float:
    return (u.min(axis=-1, keepdims=True)+u.max(axis=-1, keepdims=True))/2

def max_scale(u:np.ndarray) -> float:
    return u.max(axis=-1, keepdims=True)

def max_distance(u:np.ndarray, v:np.ndarray) -> float:
    return np.abs(u-v).max(axis=-1)

MAX_CONSTANTS = StandardisationConstants(K_m=2, K_d=1)
'''
Measuring alignment using min-max scaled standard utilities and L^\infty norm for distance.

K_m is the largest absolute utility difference assignable by a \infty-standardised utility function.
- min at -1, max at 1 - this is true for all standardised utility functions
'''

MAX = AlignmentMetric(shift=max_shift, scale=max_scale, standardised_distance=max_distance, constants=MAX_CONSTANTS, name='MAX')
