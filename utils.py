import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

x_labels = {"ca": "Collective Alignment",
            "ia": "Individual Alignment", 
            "ic": "Individual Capabilities",
            "cc": "Collective Capabilities"}

teal = "#2b8a67"
orange = "#c96532"
pink = "#a1427b"
blue = "#4244a1"

# Experiment 1
def re_plot_all(sizes, variables, others, name, bounds=False):

    exp_1_combinations = list(itertools.product(sizes, variables, others))
    for (dims, variable, v_others) in exp_1_combinations:
         plot_exp_1(dims, variable, v_others, name, bounds=bounds)

def plot_exp_1(dims, variable, v_others, name, bounds=False):

    code = "{}-{}-{}-{}".format(variable, "x".join(map(str,dims)), v_others, name)
    fname = "exp1/data/{}.csv".format(code)

    data = pd.read_csv(fname)

    if bounds:
        # if variable == "ca":
        #     data["ca"] = data["v"]
        # else:
        #     data["ca"] = v_others
        # data["correct * incorrect"] = (2 * (1 - data["ca"])) ** 2
        # data["actual_ideal_regret"] = data["correct * incorrect"].div(data["ideal_regret"], axis="index")
        data["bound_1"] = data["w_hat_max"] - data["max_regret"]
        data["bound_2"] = data["w_hat_plus"] - data["ideal_regret"]

    normalised_data = data.sub(data["w_hat_minus"], axis="index")
    normalised_data = normalised_data.div(data["w_hat_plus"] - data["w_hat_minus"], axis="index")
    normalised_data["v"] = data["v"]
    normalised_data = normalised_data.drop(columns=["Unnamed: 0", "max_regret", "ideal_regret"])
    normalised_data = normalised_data.round(3)

    if bounds:
        plot_data = pd.melt(normalised_data[['v','w_hat_min','w_hat_avg','w_hat_max','bound_1','bound_2']], ['v'])
        palette = [teal, orange, teal, pink, blue]
    else:
        plot_data = pd.melt(normalised_data[['v','w_hat_min','w_hat_avg','w_hat_max']], ['v'])
        palette = [teal, orange, teal]

    plt.figure()
    # plt.subplots_adjust(bottom=0.15)
    ax = sns.lineplot(x='v', 
                    y='value', 
                    hue='variable',
                    # palette=sns.color_palette(palette='Set2',n_colors=2),
                    palette=palette,
                    legend=False,
                    data=plot_data, 
                    errorbar=('ci', 90))
    ax.set_xticks([0.0,1.0],["0","1"])
    ax.set_yticks([0.0,1.0],["Min","Max"])
    # ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
    # ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=15)
    ax.set_xlabel(x_labels[variable],fontsize=15)
    ax.set_ylabel("Principal Welfare",fontsize=15)
    plt.tight_layout()

    pname = "exp1/plots/{}.png".format(code)
    plt.savefig(pname, dpi=300)

    return