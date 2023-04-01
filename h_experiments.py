import math
import numpy as np
import nashpy as nash
import matplotlib.pyplot as plt


# NUMBER OF STRATEGIES AND GAME SIZE IS HARDCODED TO FOR NOW (TO 2x2)
pure_strategies_2b2 = [[0, 1], [1, 0]]

# prisoners' dilemma
pd = nash.Game(np.array([[3, 0], [5, 1]]), np.array([[3, 5], [0, 1]]))


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
    max_strat = np.max(list(map(sum, pure_outcomes)))

    min_in_equil = np.min(list(map(sum, equil_outcomes)))

    return max_strat - min_in_equil


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
    max_strat = np.max(list(map(sum, pure_outcomes)))

    min_in_equil = np.min(list(map(sum, equil_outcomes)))

    return max_strat - min_in_equil


def dist_scale(rho):
    return math.sqrt(1-rho) * math.sqrt(1/2)


def epic_horiz(game):
    # for measuring horizontal alignment
    row_payoffs = game.payoff_matrices[0]
    col_payoffs = game.payoff_matrices[1]

    # note: uniform distribution assumed
    matrix_coeffs = np.corrcoef([row_payoffs.flatten(), col_payoffs.flatten()])

    return dist_scale(matrix_coeffs[0][1])


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
    return (dist_scale(row_matrix_coeffs[0][1]), dist_scale(col_matrix_coeffs[0][1]))


def max_min_normalise(data: np.array) -> np.array:
    return (data-np.min(data))/(np.max(data)-np.min(data))


"""
WELF REGRET vs HORIZONTAL ALIGNMENT 

everything else constant
"""


def welf_regr_vs_horiz_align():
    # vertical alignment set to 1 by making principal game = agent game
    # epsilon = 1, so we consider simple NEs

    # sample row & col payoffs from uniform distribution [0,1]
    # min-max normalise these between [0,1]
    normalised_utilities_row = max_min_normalise(
        np.random.uniform(0, 1, size=(2, 2)))
    normalised_utilities_col = max_min_normalise(
        np.random.uniform(0, 1, size=(2, 2)))

    # for our purposes G = G'
    game = nash.Game(normalised_utilities_row, normalised_utilities_col)

    # note: in this case, principals_welf_regret = welf_regret
    return (welf_regret(game), epic_horiz(game), game)


def plot_iterations_ha(num_iterations):
    # plots num_iterations of welfare regret vs. horizontal alignment
    x_axis = []
    y_axis = []
    # first_in_q1, first_in_q2, first_in_q3, first_in_q4 = True, True, True, True

    for i in range(num_iterations):
        welf_regret, horiz_align, game = welf_regr_vs_horiz_align()
        x_axis.append(horiz_align)
        y_axis.append(welf_regret)
    #     if horiz_align <= 0.5 and welf_regret >= 1 and first_in_q1:
    #         print("QUADRANT 1")
    #         print(game)
    #         first_in_q1 = False
    #     if horiz_align >= 0.5 and welf_regret >= 1 and first_in_q2:
    #         print("QUADRANT 2")
    #         print(game)
    #         first_in_q2 = False
    #     if horiz_align <= 0.5 and welf_regret <= 1 and first_in_q3:
    #         print("QUADRANT 3")
    #         print(game)
    #         first_in_q3 = False
    #     if horiz_align >= 0.5 and welf_regret <= 1 and first_in_q4:
    #         print("QUADRANT 4")
    #         print(game)
    #         first_in_q4 = False

    plt.scatter(x_axis, y_axis)
    plt.xlabel('Distance from Horizontal Alignment')
    plt.ylabel('Principals\' Welfare Regret')
    plt.title('Random sample of games (uniform between [0,1])')
    plt.show()


"""
WELF REGRET vs VERTICAL ALIGNMENT 

everything else constant
"""


def welf_regr_vs_vertic_align():
    # horizontal alignment set to 1 as row utilities = col utilities

    # sample payoffs from uniform distribution [0,1]
    # min-max normalise these between [0,1]
    normalised_utilities_principals = max_min_normalise(
        np.random.uniform(0, 1, size=(2, 2)))

    normalised_utilities_agents = max_min_normalise(
        np.random.uniform(0, 1, size=(2, 2)))

    principal_game = nash.Game(
        normalised_utilities_principals, normalised_utilities_principals)
    agent_game = nash.Game(normalised_utilities_agents,
                           normalised_utilities_agents)

    return (princ_welf_regret(principal_game, agent_game), epic_vertic(principal_game, agent_game), welf_regret(agent_game) == 0)


def welf_regr_vs_single_vertic_align():
    # horizontal alignment kept at 1 by making row_agent = col_agent = col_principal

    # sample payoffs from uniform distribution [0,1]
    # min-max normalise these between [0,1]
    normalised_utilities_principals = max_min_normalise(
        np.random.uniform(0, 1, size=(2, 2)))

    normalised_utilities_agent = max_min_normalise(
        np.random.uniform(0, 1, size=(2, 2)))

    principal_game = nash.Game(
        normalised_utilities_principals, normalised_utilities_agent)

    # only row agent is misaligned
    agent_game = nash.Game(normalised_utilities_agent,
                           normalised_utilities_agent)
    row, col = epic_vertic(principal_game, agent_game)

    return (princ_welf_regret(principal_game, agent_game), row, welf_regret(agent_game) == 0)


def plot_iterations_va(num_iterations, double: bool):
    # plots num_iterations of welfare regret vs. vertical alignment of both players
    x_axis = []
    y_axis = []

    for i in range(num_iterations):
        if double:
            welf_regret, vertic_align, hcc = welf_regr_vs_vertic_align()

            row_align, col_align = vertic_align
            x_axis.append(row_align)
        else:
            welf_regret, vertic_align, hcc = welf_regr_vs_single_vertic_align()
            x_axis.append(vertic_align)
        y_axis.append(welf_regret)

    plt.scatter(x_axis, y_axis)
    if double:
        plt.xlabel(
            'Distance from Vertical Alignment of both principal-agent pairs')

    else:
        plt.xlabel('Distance from Vertical Alignment')
    plt.ylabel('Principals\' Welfare Regret')
    plt.title('Random sample of games (uniform between [0,1])')
    plt.show()


def plot_iterations_va_hcc(num_iterations, double: bool):
    # plots num_iterations of welfare regret vs. vertical alignment of both players
    # only plots games where horizontal welfare regret = 0
    x_axis = []
    y_axis = []

    for i in range(num_iterations):
        if double:
            welf_regret, vertic_align, horiz_cap = welf_regr_vs_vertic_align()

            row_align, col_align = vertic_align
            if horiz_cap:
                x_axis.append(row_align)
        else:
            welf_regret, vertic_align, horiz_cap = welf_regr_vs_single_vertic_align()
            if horiz_cap:
                x_axis.append(vertic_align)
        if horiz_cap:
            y_axis.append(welf_regret)

    plt.scatter(x_axis, y_axis)
    if double:
        plt.xlabel(
            'Distance from Vertical Alignment of both principal-agent pairs')

    else:
        plt.xlabel('Distance from Vertical Alignment')
    plt.ylabel('Principals\' Welfare Regret')
    plt.title(
        'HC controlled sandom sample of games')
    plt.show()


# plot_iterations_ha(1000)
# plot_iterations_va(1000, True)
#plot_iterations_va(1000, False)
plot_iterations_va_hcc(1000, True)
# plot_iterations_va_hcc(1000, False)
