import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import matplotlib.patches as mpatches

def get_perturbation(text):
    tokens = text.split("_")
    if len(tokens) == 3:
        return tokens[1]
    else:
        return tokens[-1]

def get_percentage(text):
    tokens = text.split("_")
    if len(tokens) == 3:
        return tokens[-1]
    else:
        return np.nan
    
def subcategory_bar(
    categories: list,
    values: list,
    labels: list,
    alphas: list,
    colors: list,
    ax: None or plt.Axes = None,
    type: str = "character"):
    
    if ax is None:
        fig, ax = plt.subplots()

    n_categories = len(categories)
    n_bars = len(values)

    aux_x = np.arange(n_categories)

    bars1 = ax.bar(x = aux_x - 0.3, height = values[0], width = 0.3, color = colors[0], label = labels[0])
    bars2 = ax.bar(x = aux_x, height = values[1], width = 0.3, color = colors[1], label = labels[1])
    bars3 = ax.bar(x = aux_x + 0.3, height = values[2], width = 0.3, color = colors[2], label = labels[2])
    for i in range(n_categories):
        bars1[i].set_alpha(alphas[0][i])
        bars2[i].set_alpha(alphas[1][i])
        bars3[i].set_alpha(alphas[2][i])

        for j in range(3):
            ax.annotate(
                text = f"{(alphas[j][i] * 100):.0f} %",
                xy = (aux_x[i] + 0.3*(j-1), values[j][i] + 0.035),
                ha = "center",
                va = "center",
                bbox = {
                    "facecolor" : "white",
                    "edgecolor" : "black",
                    "boxstyle" : "round,pad=0.3"
                },
                size = 9 if type == "word" else 7
            )

    ax.set_xticks(aux_x, categories)
    ax.set_ylabel("Cohen's kappa")
    if np.min(values) < 0:
        ax.set_ylim(-1, 1)
    else:
        ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(mticks.MultipleLocator(0.1))
    ax.grid(which = "major", axis = "x", visible = False)

    ax.legend(
        labels = labels,
        handles = [
            mpatches.Rectangle((0,0), 0, 0, color = colors[0]),
            mpatches.Rectangle((0,0), 0, 0, color = colors[1]),
            mpatches.Rectangle((0,0), 0, 0, color = colors[2])
        ],
        ncols = 3,
        loc = "upper center",
        bbox_to_anchor = (0.5, 1.20),
        title = "% of characters perturbed" if type == "character" else "% of words perturbed"
    )

    return ax

datasets = ["cola_perturbed", "hsol_perturbed", "mnli_perturbed", "paws_perturbed", "sst2_perturbed"]
tasks = ["GC", "HSOL", "NLI", "SS", "SA"]
long_names = ["GC (COLA)", "HSOL (HSO)", "NLI (MNLI)", "SS (PAWS)", "SA (SST2)"]
models = ["DistilBERT", "ELECTRA", "Funnel Transformer", "XLNet"]

for i in range(len(tasks)):
    ds = datasets[i]
    t = tasks[i]
    ln = long_names[i]
    for j in range(len(models)):
        m = models[j]

        dataset = pd.read_csv(ds + ".csv", sep = ";")

        if m == "Funnel Transformer":
            path = rf"./{m}/{ln}/FunTrf-{t}_character_level_evaluation"
        else:
            path = rf"./{m}/{ln}/{m}-{t}_character_level_evaluation"

        model_task = pd.read_csv(path + ".csv", sep = ";")

        dataset["type_lower"] = dataset.dataset_name.apply(get_perturbation)
        dataset["percentage_perturbation"] = dataset.dataset_name.apply(get_percentage)
        dataset = dataset.drop(columns = ["dataset_name"])
        dataset.percentage_perturbation = dataset.percentage_perturbation.astype(float)

        model_task["type_lower"] = model_task.type.str.lower()

        model_task_scores = pd.merge(dataset, model_task, on = ["type_lower", "percentage_perturbation"], how = "inner")
        model_task_scores = model_task_scores.drop(columns = ["type_lower"])

        categories = model_task_scores.loc[model_task_scores.percentage_perturbation == 0.01, "type"].values

        values = [
            model_task_scores.loc[model_task_scores.percentage_perturbation == 0.01, "kappa"].values,
            model_task_scores.loc[model_task_scores.percentage_perturbation == 0.05, "kappa"].values,
            model_task_scores.loc[model_task_scores.percentage_perturbation == 0.1, "kappa"].values
        ]

        alphas = [
            model_task_scores.loc[model_task_scores.percentage_perturbation == 0.01, "percentage_altered"].values,
            model_task_scores.loc[model_task_scores.percentage_perturbation == 0.05, "percentage_altered"].values,
            model_task_scores.loc[model_task_scores.percentage_perturbation == 0.1, "percentage_altered"].values 
        ]

        ax = subcategory_bar(
            categories = categories,
            values = values,
            labels = ["1%", "5%", "10%"],
            alphas = alphas,
            colors = ["lightblue", "deepskyblue", "royalblue"],
            type = "character"
        )

        plt.savefig(rf"Plots/{m}-{t}-character_evaluation.png", dpi = 150, bbox_inches = "tight")