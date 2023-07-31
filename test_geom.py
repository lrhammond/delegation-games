import numpy as np

from alignment import EPIC
from geom import orthogonal_project_planar, standardise, get_angle, random_heading_in_zero_mean_subspace, rotate_in_plane, random_epic_distance_step

seed = 1282
rng = np.random.RandomState(seed)

def test_random_standardise():
    a = rng.randn(5)
    b = standardise(a)
    m = b.mean().round(3)
    s = np.linalg.norm(b).round(3)
    print(m, s)
    assert m == 0.0
    assert s == 1.0

def test_orthogonal_project_axis_aligned():
    a = [0, 0.001]
    b = rng.randn(2)
    # should project to (0, 1) and (+-1, 0)
    a_orth, b_orth = orthogonal_project_planar(a, b)
    print(a_orth, b_orth)
    assert np.array_equiv([0, 1], a_orth)
    assert np.array_equiv([1, 0], b_orth) or np.array_equiv([-1, 0], b_orth)

def test_orthogonal_project_random():
    a, b = rng.randn(2, 5)
    print(a, b)
    a_orth, b_orth = orthogonal_project_planar(a, b)
    print(a_orth, b_orth)
    a_m = np.linalg.norm(a_orth).round(3)
    b_m = np.linalg.norm(b_orth).round(3)
    turns = (get_angle(a_orth, b_orth) / (2*np.pi)).round(3)
    assert a_m == 1.0
    assert b_m == 1.0
    assert turns == 0.25

def random_zero_mean_rotation(v:np.ndarray, theta:float) -> np.ndarray:
    '''
    Randomly rotate given vector by given angle
    '''
    dim = v.shape[0]
    random_heading = random_heading_in_zero_mean_subspace(dim, rng)
    return rotate_in_plane(v, random_heading, theta)

def test_rotate_standard_vector():
    # rotating a standardised vector by any amount should yield a standardised vector
    a = standardise(rng.randn(5))
    u = random_zero_mean_rotation(a, rng.uniform(0, np.pi))
    u_m = u.mean().round(3)
    u_s = np.linalg.norm(u).round(3)
    assert u_m == 0.0
    assert u_s == 1.0

def test_rotate_known_axis_aligned():
    # specific 2d rotations of known point
    a = np.array([0, 1])
    assert np.allclose(random_zero_mean_rotation(a, 0), a)
    assert np.allclose(random_zero_mean_rotation(a, np.pi), -a)
    assert np.allclose(np.abs(random_zero_mean_rotation(a, np.pi/2)), [1, 0])
    assert np.allclose(np.abs(random_zero_mean_rotation(a, np.pi/3)), [np.sqrt(3/4), 0.5])
    assert np.allclose(np.abs(random_zero_mean_rotation(a, np.pi/4)), [np.sqrt(1/2), np.sqrt(1/2)])
    assert np.allclose(np.abs(random_zero_mean_rotation(a, np.pi/6)), [0.5, np.sqrt(3/4)])

def test_rotate_random():
    # random rotation should be by specified angle
    a = standardise(rng.randn(10))
    theta = rng.uniform(0, np.pi)
    a_rotated = random_zero_mean_rotation(a, theta)
    theta_recovered = get_angle(a, a_rotated)
    assert np.allclose(theta_recovered, theta)

def test_random_epic_step():
    for _ in range(10):
        a = standardise(rng.randn(16))
        d_epic = rng.uniform(0, 1)
        a_stepped = random_epic_distance_step(a, d_epic, rng)
        d_epic_recovered = EPIC.distance(a, a_stepped)
        assert np.allclose(d_epic, d_epic_recovered)
