import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

x_labels = {"ca": "Collective Alignment",
            "ia": "Individual Alignment", 
            "ic": "Individual Capabilities",
            "cc": "Collective Capabilities"}

to_plot = { "ca": "all",
            "ia": "all", 
            "ic": "eps_NEs",
            "cc": "played"}

# teal = "#2b8a67"
# pink = "#a1427b"
# orange = "#c96532"
# blue = "#4244a1"

teal = (47.0/255.0, 95.0/255.0, 82.0/255.0)
red = (161.0/255.0, 41.0/255.0, 47.0/255.0)
orange = (224.0/255.0, 105.0/255.0, 63.0/255.0)
blue = (88.0/255.0, 143.0/255.0, 188.0/255.0)

# Experiment 1
def re_plot_all_exp_1(sizes, variables, others, name, bounds=False):

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
        plot_data = pd.melt(normalised_data[['v','bound_2','bound_1','w_hat_min','w_hat_max','w_hat_avg']], ['v'])
        palette = [blue, orange, teal, teal, red]
    else:
        plot_data = pd.melt(normalised_data[['v','w_hat_min','w_hat_max', 'w_hat_avg']], ['v'])
        palette = [teal, teal, red]

    plt.figure()
    # plt.subplots_adjust(bottom=0.15)
    ax = sns.lineplot(x='v', 
                    y='value', 
                    hue='variable',
                    # palette=sns.color_palette(palette='Set2',n_colors=2),
                    palette=palette,
                    legend=False,
                    data=plot_data, 
                    errorbar=('ci', 90),
                    linewidth=2)
    ax.set_xticks([0.0,1.0],["0","1"])
    ax.set_yticks([0.0,1.0],["Min","Max"])
    ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0], ["0","","","","","1"])
    ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0], ["Min","","","","","Max"])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=20,length=8, width=2)
    ax.set_xlabel(x_labels[variable],fontsize=20)
    ax.set_ylabel("Principal Welfare",fontsize=20)
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -25
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
    plt.tight_layout()

    pname = "exp1/plots/{}.png".format(code)
    plt.savefig(pname, dpi=300)

    return

def plot_exp_2(sizes, dists, name):

    data = {}
    for dims in sizes:
        data[dims] = {}
        for d in dists:
            code = "{}-{}-{}".format("x".join(map(str,dims)), d, name)
            fname = "exp2/data/{}.csv".format(code)
            data[dims][d] = pd.read_csv(fname)

    for k in x_labels:

        loss = k + "_loss"
        columns = ["samples"] + ["x".join(map(str,dims)) for dims in data]
        plt.figure()
        # plt.subplots_adjust(bottom=0.15)

        y_lim = 0.1 if k == "ia" or k == "ca" else 0.6
        
        for d in dists:

            if to_plot[k] != d:
                continue

            combined_data = pd.concat([data[sizes[0]][d]["samples"]] + [data[dims][d][loss] for dims in data], axis=1)
            combined_data.columns = columns
            plot_data = pd.melt(combined_data, ["samples"])

            # if d == "all":
            #     colour = red
            # elif d == "played":
            #     colour = orange
            # elif d == "eps_NEs":
            #     colour = teal
            # elif d == "NEs":
            #     colour = blue
            palette = [red, orange, blue, teal]
            # for i in range(len(dims)):
            #     new_colour = tuple([min(c + (i*0.2), 1.0) for c in colour])
            #     palette += [new_colour]

            ax = sns.lineplot(x='samples', 
                            y='value', 
                            hue='variable',
                            # palette=sns.color_palette(palette='Set2'),
                            palette=palette,
                            legend=False,
                            data=plot_data, 
                            errorbar=('ci', 90))

        # ax.set_xticks([0.0,1.0],["0","1"])
        # ax.set_yticks([0.0,1.0],["Min","Max"])
        # ax.set_xticks([0.0,0.2,0.4,0.6,0.8,1.0])
        # ax.set_yticks([0.0,0.2,0.4,0.6,0.8,1.0])
        ax.set_xlim(10, 1000)
        ax.set_ylim(0, y_lim)
        plt.xscale('log')
        ax.tick_params(labelsize=20,length=8,width=2,which='both')
        ax.set_xlabel("Samples",fontsize=20)
        ax.set_ylabel("{}".format(x_labels[k]),fontsize=20)
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 10
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)
        plt.tight_layout()
        pname = "exp2/plots/{}-{}.png".format(k, name)
        plt.savefig(pname, dpi=300)

    # normalised_data = data.sub(data["w_hat_minus"], axis="index")
    # normalised_data = normalised_data.div(data["w_hat_plus"] - data["w_hat_minus"], axis="index")
    # normalised_data["v"] = data["v"]
    # normalised_data = normalised_data.drop(columns=["Unnamed: 0", "max_regret", "ideal_regret"])
    # normalised_data = normalised_data.round(3)

    # if bounds:
    #     plot_data = pd.melt(normalised_data[['v','w_hat_min','w_hat_avg','w_hat_max','bound_1','bound_2']], ['v'])
    #     palette = [teal, orange, teal, pink, blue]
    # else:
    #     plot_data = pd.melt(normalised_data[['v','w_hat_min','w_hat_avg','w_hat_max']], ['v'])
    #     palette = [teal, orange, teal]

    # return
        
def print_latex_figure(dims, variables, others, name):

    print("\\begin{figure}")
    print("\t\centering")
    for o in others:
        for v in variables:
            code = "{}-{}-{}-{}".format(v, "x".join(map(str,dims)), o, name)
            print('\t\\begin{subfigure}{0.24\\textwidth}')
            print("\t\t\centering")
            print("\t\t\includegraphics[width=0.95\\textwidth]{figures/inference/" + code + ".png}")
            print("\t\t% \caption\{\}")
            print("\t\end{subfigure}")
        print("\n")
    print("\end{figure}")