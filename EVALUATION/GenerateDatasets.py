import os
import sys

# Using only one GPU, they are being friendly enough
# to let me use the server, don't get greedy :D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# To find my library
sys.path.append("/home/daniroalv/miniconda3/envs/CodigoTFG/CodigoTFG")

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextClassificationPipeline
)

from sklearn.metrics import cohen_kappa_score
from datasets import load_dataset, DatasetDict
import Transformations as Trf2
import numpy as np
import torch

np.random.seed(8888)
torch.manual_seed(8888)

aux_hso = load_dataset("hate_speech_offensive")["train"].train_test_split(test_size = 0.2, seed = 8888, stratify_by_column = "class")

cola_test = Trf2.PerturbedDataset(load_dataset("glue", "cola")["test"])
mnli_test = Trf2.PerturbedDataset(load_dataset("glue", "mnli")["test_matched"])
sst2_test = Trf2.PerturbedDataset(load_dataset("glue", "sst2")["test"])
paws_test = Trf2.PerturbedDataset(load_dataset("paws", "labeled_final")["test"])
hso_test = Trf2.PerturbedDataset(aux_hso["test"])

dataset_names = ["cola", "mnli", "sst2", "paws", "hsol"]
fields = ["sentence", ["premise", "hypothesis"], "sentence", ["sentence1", "sentence2"], "tweet"]

for i, test_dataset in enumerate([cola_test, mnli_test, sst2_test, paws_test, hso_test]):
    dataset_dict = DatasetDict()
    cad = "dataset_name;percentage_altered\n"
    for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
        for percentage_perturbation in [0.01, 0.05, 0.1]:
            print(dataset_names[i], perturbation, percentage_perturbation)
            if perturbation == "Keyboard":
                perturbation_object = Trf2.CharacterPerturbation(
                    perturbation_type = perturbation,
                    proportion_characters = percentage_perturbation,
                    include_special_chars = False, 
                    include_numeric = False,
                    include_upper_case = False)
            else:    
                perturbation_object = Trf2.CharacterPerturbation(
                    perturbation_type = perturbation, 
                    proportion_characters = percentage_perturbation)

            test_pert, percentage_altered = test_dataset.perturb(
                perturbation = perturbation_object, 
                x_fields = fields[i])

            cad += f"{dataset_names[i]}_{perturbation.lower()}_{percentage_perturbation};{percentage_altered}\n"
            dataset_dict[f"{dataset_names[i]}_perturbed_{perturbation.lower()}_{percentage_perturbation}"] = test_pert.dataset

    for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
        for percentage_perturbation in [0.1, 0.2, 0.3]:
            print(dataset_names[i], perturbation, percentage_perturbation)
            perturbation_object = Trf2.WordPerturbation(
                perturbation_type = perturbation, 
                proportion_words = percentage_perturbation)

            test_pert, percentage_altered = test_dataset.perturb(
                perturbation = perturbation_object, 
                x_fields = fields[i])

            cad += f"{dataset_names[i]}_{perturbation.lower()}_{percentage_perturbation};{percentage_altered}\n"
            dataset_dict[f"{dataset_names[i]}_perturbed_{perturbation.lower()}_{percentage_perturbation}"] = test_pert.dataset

    for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:
        print(dataset_names[i], perturbation)
        perturbation_object = Trf2.OtherPerturbation(
            perturbation_type = perturbation)

        test_pert, percentage_altered = test_dataset.perturb(
            perturbation = perturbation_object, 
            x_fields = fields[i])

        cad += f"{dataset_names[i]}_{perturbation.lower()};-;{percentage_altered}\n"
        dataset_dict[f"{dataset_names[i]}_perturbed_{perturbation.lower()}_{percentage_perturbation}"] = test_pert.dataset

    with open(f"{dataset_names[i]}_perturbed.csv", "w") as file:
        file.write(cad)

    dataset_dict.save_to_disk(f"{dataset_names[i]}_perturbed")