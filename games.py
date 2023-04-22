import nashpy as nash
import numpy as np

# Prisoner's Dilemma
pd = nash.Game(np.array([[3, 0], [5, 1]]), np.array([[3, 5], [0, 1]]))

# Negative Prisoner's Dilemma
pd_negative = nash.Game(np.array([[-3, -10], [0, -5]]), np.array([[-3, 0], [-10, -5]]))

# Game of Chicken
chicken = nash.Game(np.array([[0, -1], [1, -10]]), np.array([[0, 1], [-1, -10]]))

# Stug Hunt
hunt = nash.Game(np.array([[5, 0], [2, 1]]), np.array([[5, 2], [0, 1]]))

# Stug Hunt
coop_hunt = nash.Game(np.array([[5, 1], [1, 1]]), np.array([[5, 1], [1, 1]]))

# Matching Pennies
mp = nash.Game(np.array([[1, -1], [-1, 1]]), np.array([[-1, 1], [1, -1]]))

# Battle of the Sexes
bos = nash.Game(np.array([[1, 0], [0, 2]]), np.array([[2, 0], [0, 1]]))

# Test Game
test = nash.Game(np.array([[1.0, 0.9999], [1.0, 1.00001]]), np.array([[1.0, 1.00001], [1.0, 1.0]]))

games = {
    "Prisoner's dilemma": pd,
    "Prisoner's dilemma (negative pay-offs)": pd_negative,
    "Game of Chicken": chicken,
    "Stug Hut": hunt,
    "Matching Pennies": mp,
    "Battle of the Sexes": bos,
    "Test game": test}

def uniform_2x2_payoff() -> np.ndarray:
    return np.random.uniform(size=(2, 2))

def uniform_2x2() -> nash.Game:
    return nash.Game(uniform_2x2_payoff(), uniform_2x2_payoff())

def uniform_2x2_aligned() -> nash.Game:
    payoff = uniform_2x2_payoff()
    return nash.Game(payoff, payoff)

def swap_2x2_0(g:nash.Game, new_payoff:np.ndarray) -> nash.Game:
    return nash.Game(new_payoff, g.payoff_matrices[1])

def swap_2x2_1(g:nash.Game, new_payoff:np.ndarray) -> nash.Game:
    return nash.Game(g.payoff_matrices[0], new_payoff)
