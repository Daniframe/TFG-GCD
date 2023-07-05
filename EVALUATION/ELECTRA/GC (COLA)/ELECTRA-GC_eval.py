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
from datasets import load_dataset, load_from_disk
import Transformations as Trf2

best_model = AutoModelForSequenceClassification.from_pretrained(
    "ELECTRA-GC_lr-5e-05_nep-7")

tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
cola_test = Trf2.PerturbedDataset(load_dataset("glue", "cola")["test"])

pipeline = TextClassificationPipeline(model = best_model, tokenizer = tokenizer, top_k = 1)

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.01, 0.05, 0.1]:
        perturbed_dataset = load_from_disk(
            f"../../cola_perturbed/cola_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        predictions_normal = Trf2.predict_dataset(
            dataset = cola_test,
            pipe = pipeline,
            x_fields = "sentence")

        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "sentence")

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"ELECTRA-GC_character_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.1, 0.2, 0.3]:
        perturbed_dataset = load_from_disk(
            f"../../cola_perturbed/cola_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        predictions_normal = Trf2.predict_dataset(
            dataset = cola_test,
            pipe = pipeline,
            x_fields = "sentence")

        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "sentence")

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"ELECTRA-GC_word_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;kappa\n"
for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:
    perturbed_dataset = load_from_disk(
        f"../../cola_perturbed/cola_perturbed_{perturbation.lower()}")
    
    predictions_normal = Trf2.predict_dataset(
        dataset = cola_test,
        pipe = pipeline,
        x_fields = "sentence")

    predictions_perturbed = Trf2.predict_dataset(
        dataset = perturbed_dataset,
        pipe =  pipeline,
        x_fields = "sentence")

    kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
    cad += f"{perturbation};{kappa}\n"

with open(f"ELECTRA-GC_other_evaluation.csv", "w") as file:
    file.write(cad)