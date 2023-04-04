from typing import Tuple
import numpy as np


def standardise(v:np.ndarray) -> np.ndarray:
    '''
    Zero-centre and normalise vector(s).

    Applied along final axis, preserving batch dims if vectorised.
    '''
    v_centred = v - v.mean(axis=-1, keepdims=True)
    return v_centred / np.linalg.norm(v_centred, axis=-1, keepdims=True)

def get_angle(v1:np.ndarray, v2:np.ndarray) -> float:
    '''
    Get angle between vector(s).

    Applied along final axis, preserving batch dims if vectorised.
    '''
    u1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    u2 = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    return np.arccos(np.einsum('...i,...i', u1, u2))

def orthogonal_project_planar(v1:np.ndarray, v2:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Produce orthonormalised vectors from given vectors using Gram-Schmidt projection procedure, restricted to two vectors.

    Input vectors must be linearly independent.
    '''
    u1 = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    print(np.einsum('...i,...i', u1, v2)) # inner product...? How to deal with v1 _not_ vector?
    print(u1 * np.einsum('...i,...i', u1, v2))
    o2 = v2 - np.einsum('...i,...i', u1, v2) * u1 # orthogonal projection of v2 onto v1
    print(o2)
    u2 = o2 / np.linalg.norm(o2, axis=-1, keepdims=True)
    return u1, u2

def get_rotation_matrix(a:np.ndarray, b:np.ndarray, theta:float) -> np.ndarray:
    '''
    Get the rotation of the given angle parallel to the plane containing the given points and the origin.

    Returns rotation matrix.
    '''
    dim = a.shape[0]
    # get orthogonal vectors spanning the plane of rotation between v_centred and the random heading
    u1, u2 = orthogonal_project_planar(a, b)
    # n-dimensional generalisation of Rodrigues' formula for efficient rotation matrix given plane and angle
    R = np.eye(dim) + np.outer(u2, u1) * np.sin(theta) + (np.outer(u1, u1) + np.outer(u2, u2)) * (np.cos(theta) - 1)
    return R

def rotate_in_plane(a:np.ndarray, b:np.ndarray, theta:float) -> np.ndarray:
    '''
    The normalised result of rotating point a by theta about the origin through the plane spanned by a and b
    '''
    u1, u2 = orthogonal_project_planar(a, b)
    return u1 * np.cos(theta) + u2 * np.sin(theta)

def random_heading_in_zero_mean_subspace(dim:int) -> np.ndarray:
    '''
    Generate a random vector direction in the subspace defined by entries summing to zero.
    '''
    # normal variates are spherically symmetric
    # but we also need to satisfy zero mean
    # TODO is this still usefully 'uniform' after the projection? Possible to sample directly from dim-1 subspace?
    return standardise(np.random.randn(dim))

def random_epic_distance_step(v:np.ndarray, d_epic:float) -> np.ndarray:
    '''
    Generate a random utility which has given epic distance from the given utility.

    Result is flattened, centred, and normalised.
    '''
    u = standardise(v)
    random_heading = random_heading_in_zero_mean_subspace(v.shape[0])
    theta = 2 * np.arcsin(np.minimum(1, d_epic))
    rotated = rotate_in_plane(u, random_heading, theta)
    return rotated
