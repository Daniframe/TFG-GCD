import pandas as pd
import numpy as np

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
        model_task_scores = model_task_scores.rename(
            columns = {
                "percentage_altered" : "% of samples perturbed",
                "percentage_perturbation" : "% of perturbed characters per sample",
                "kappa" : "Cohen's kappa",
                "type" : "Perturbation"
            }
        ).reindex(columns = ["Perturbation", "% of perturbed characters per sample",
                             "Cohen's kappa", "% of samples perturbed"])
        
        model_task_scores.to_csv(rf"Tables/Character/{m}-{t}.csv", index = False, sep = ";", decimal = ",")

for i in range(len(tasks)):
    ds = datasets[i]
    t = tasks[i]
    ln = long_names[i]
    for j in range(len(models)):
        m = models[j]

        dataset = pd.read_csv(ds + ".csv", sep = ";")

        if m == "Funnel Transformer":
            path = rf"./{m}/{ln}/FunTrf-{t}_word_level_evaluation"
        else:
            path = rf"./{m}/{ln}/{m}-{t}_word_level_evaluation"

        model_task = pd.read_csv(path + ".csv", sep = ";")

        dataset["type_lower"] = dataset.dataset_name.apply(get_perturbation)
        dataset["percentage_perturbation"] = dataset.dataset_name.apply(get_percentage)
        dataset = dataset.drop(columns = ["dataset_name"])
        dataset.percentage_perturbation = dataset.percentage_perturbation.astype(float)

        model_task["type_lower"] = model_task.type.str.lower()

        model_task_scores = pd.merge(dataset, model_task, on = ["type_lower", "percentage_perturbation"], how = "inner")
        model_task_scores = model_task_scores.drop(columns = ["type_lower"])
        model_task_scores = model_task_scores.rename(
            columns = {
                "percentage_altered" : "% of samples perturbed",
                "percentage_perturbation" : "% of perturbed words per sample",
                "kappa" : "Cohen's kappa",
                "type" : "Perturbation"
            }
        ).reindex(columns = ["Perturbation", "% of perturbed words per sample",
                             "Cohen's kappa", "% of samples perturbed"])
        
        model_task_scores.to_csv(rf"Tables/Word/{m}-{t}.csv", index = False, sep = ";", decimal = ",")

for i in range(len(tasks)):
    ds = datasets[i]
    t = tasks[i]
    ln = long_names[i]
    for j in range(len(models)):
        m = models[j]

        dataset = pd.read_csv(ds + ".csv", sep = ";")

        if m == "Funnel Transformer":
            path = rf"./{m}/{ln}/FunTrf-{t}_other_evaluation"
        else:
            path = rf"./{m}/{ln}/{m}-{t}_other_evaluation"

        model_task = pd.read_csv(path + ".csv", sep = ";")

        dataset["type_lower"] = dataset.dataset_name.apply(get_perturbation)
        dataset = dataset.drop(columns = ["dataset_name"])

        model_task["type_lower"] = model_task.type.str.lower()

        model_task_scores = pd.merge(dataset, model_task, on = ["type_lower"], how = "inner")
        model_task_scores = model_task_scores.drop(columns = ["type_lower"])
        model_task_scores = model_task_scores.rename(
            columns = {
                "percentage_altered" : "% of samples perturbed",
                "kappa" : "Cohen's kappa",
                "type" : "Perturbation"
            }
        ).reindex(columns = ["Perturbation",
                             "Cohen's kappa", "% of samples perturbed"])
        
        model_task_scores.to_csv(rf"Tables/Other/{m}-{t}.csv", index = False, sep = ";", decimal = ",")