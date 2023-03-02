import nashpy as nash
import numpy as np
import random

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
test = nash.Game(np.array([[1, 0.9999], [1, 1.00001]]), np.array([[1, 1.00001], [1, 1]]))

games = {
    "Prisoner's dilemma": pd,
    "Prisoner's dilemma (negative pay-offs)": pd_negative,
    "Game of Chicken": chicken,
    "Stug Hut": hunt,
    "Matching Pennies": mp,
    "Battle of the Sexes": bos,
    "Test game": test}

# we assume games are 2x2 for now
pure_strategies_2b2 = [[0, 1], [1, 0]]


def price_of_anarchy(game):
    # nash equilibria
    eqs = list(game.support_enumeration())

    # list of expected utilities for both players for pure strategies
    pure_outcomes = [game[s_row, s_col]
                     for s_row in pure_strategies_2b2 for s_col in pure_strategies_2b2]

    # list of expected utilities for both players in nash equilibria
    equil_outcomes = [game[s_row, s_col]
                      for (s_row, s_col) in eqs]

    # we only need to check pure strategies because the maximum is
    # necessarily among these
    # maximum expected payoff strategy
    max_strat = np.argmax(list(map(sum, pure_outcomes)))

    # maximum expected payoff nash equil
    min_in_equil = np.argmin(list(map(sum, equil_outcomes)))

    return sum(pure_outcomes[max_strat]) / sum(equil_outcomes[min_in_equil])


def welf_regret(game):
    # nash equilibria
    eqs = list(game.support_enumeration())

    # list of expected utilities for both players for pure strategies
    pure_outcomes = [game[s_row, s_col]
                     for s_row in pure_strategies_2b2 for s_col in pure_strategies_2b2]

    # list of expected utilities for both players in nash equilibria
    equil_outcomes = [game[s_row, s_col]
                      for (s_row, s_col) in eqs]

    # we only need to check pure strategies because the maximum is
    # necessarily among these
    max_strat = np.argmax(list(map(sum, pure_outcomes)))

    min_in_equil = np.argmin(list(map(sum, equil_outcomes)))

    return sum(pure_outcomes[max_strat]) - sum(equil_outcomes[min_in_equil])


def princ_welf_regret(principal_game, agents_game):
    # nash equilibria
    eqs = list(agents_game.support_enumeration())

    # list of expected utilities for both PRINCIPALS for pure strategies
    pure_outcomes = [principal_game[s_row, s_col]
                     for s_row in pure_strategies_2b2 for s_col in pure_strategies_2b2]

    # list of expected utilities for both PRINCIPALS in nash equilibria of AGENT GAME
    equil_outcomes = [principal_game[s_row, s_col]
                      for (s_row, s_col) in eqs]

    # we only need to check pure strategies because the maximum is
    # necessarily among these
    max_strat = np.argmax(list(map(sum, pure_outcomes)))

    min_in_equil = np.argmin(list(map(sum, equil_outcomes)))

    return sum(pure_outcomes[max_strat]) - sum(equil_outcomes[min_in_equil])


def epic_horiz(game):
    # for measuring horizontal alignment
    row_payoffs = game.payoff_matrices[0]
    col_payoffs = game.payoff_matrices[1]

    # note: uniform distribution assumed
    matrix_coeffs = np.corrcoef([row_payoffs.flatten(), col_payoffs.flatten()])
    return matrix_coeffs[0][1]


def epic_vertic(principal_game, agents_game):
    # for measuring vertical alignment

    row_payoffs = principal_game.payoff_matrices[0]
    col_payoffs = principal_game.payoff_matrices[1]

    agent_row_payoffs = agents_game.payoff_matrices[0]
    agent_col_payoffs = agents_game.payoff_matrices[1]

    # note: uniform distribution assumed
    row_matrix_coeffs = np.corrcoef(
        [row_payoffs.flatten(), agent_row_payoffs.flatten()])
    col_matrix_coeffs = np.corrcoef(
        [col_payoffs.flatten(), agent_col_payoffs.flatten()])
    return (row_matrix_coeffs[0][1], col_matrix_coeffs[0][1])


def perturb(principal_game):
    row_payoffs = principal_game.payoff_matrices[0]
    col_payoffs = principal_game.payoff_matrices[1]

    pert_row_payoffs = np.array([[0, 0], [0, 0]], dtype=float)
    pert_col_payoffs = np.array([[0, 0], [0, 0]], dtype=float)

    for i in range(2):
        for j in range(2):
            payoff = row_payoffs[i][j]
            x = random.uniform(-1, 2)
            pert_row_payoffs[i][j] = x * payoff

    for i in range(2):
        for j in range(2):
            payoff = col_payoffs[i][j]
            x = random.uniform(-1, 2)
            pert_col_payoffs[i][j] = x * payoff

    agent_game = nash.Game(pert_row_payoffs, pert_col_payoffs)

    return agent_game


# TODO: add vertical capabilities (epsilon best response)
def calculate_measures_and_print(principal_game, agents_game):
    print("====== PRINCIPALS ======")
    print(principal_game)
    print("\n","====== AGENTS ======")
    print(agents_game)

    print("\n","====== MEASURES ======")
    print("Horizontal Alignment (EPIC): " + str(epic_horiz(agents_game)))
    row_align, col_align = epic_vertic(principal_game, agents_game)
    print("Vertical Alignment (Principal(ROW)-Agent): " + str(row_align))
    print("Vertical Alignment (Principal(COL)-Agnet): " + str(col_align))
    print("Welfare regret: " + str(welf_regret(agents_game)))
    print("Cross-game regret: " + str(princ_welf_regret(principal_game, agents_game)))

    # The Grand Equasion
    # the parentheses are for clarity only, we assume VC to be 1.
    # normalze from [-1, 1] to [0, 1]
    var = (row_align +1) / 2
    vac = (col_align +1) / 2
    ha = (epic_horiz(agents_game) + 1) /2

    print("\n","====== The Grand Equasion ======")
    print(f"C = ((VA-R * VA-C) * VC) * (HA * HC)")
    print(f"C = ((VA-R[{var}] * VA-C[{vac}]) * VC[{1}]) * (HA[{ha}] * HC[{1}])")
    print("")
    print(f"C = (VA * VC) * H")
    print(f"C = ((VA[{var * vac}]) * VC[{1}]) * H[{ha}])")
    print(f"C = {var * vac * ha}")


calculate_measures_and_print(test, test)
