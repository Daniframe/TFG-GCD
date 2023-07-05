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
    "ELECTRA-SA_lr-5e-05_nep-5/checkpoint-2345")

tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
sst2_test = Trf2.PerturbedDataset(load_dataset("glue", "sst2")["test"])

pipeline = TextClassificationPipeline(model = best_model, tokenizer = tokenizer, top_k = 1, device = 0)

predictions_normal = Trf2.predict_dataset(
    dataset = sst2_test,
    pipe = pipeline,
    x_fields = "sentence")

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.01, 0.05, 0.1]:
        perturbed_dataset = load_from_disk(
            f"../../sst2_perturbed/sst2_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        try:
            predictions_perturbed = Trf2.predict_dataset(
                dataset = perturbed_dataset,
                pipe =  pipeline,
                x_fields = "sentence")
        except:
            pass

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"ELECTRA-SA_character_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.1, 0.2, 0.3]:
        perturbed_dataset = load_from_disk(
            f"../../sst2_perturbed/sst2_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        try:
            predictions_perturbed = Trf2.predict_dataset(
                dataset = perturbed_dataset,
                pipe =  pipeline,
                x_fields = "sentence")
        except:
            pass

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"ELECTRA-SA_word_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;kappa\n"
for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:
    perturbed_dataset = load_from_disk(
        f"../../sst2_perturbed/sst2_perturbed_{perturbation.lower()}")

    try:
        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "sentence")
    except:
        pass

    kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
    cad += f"{perturbation};{kappa}\n"

with open(f"ELECTRA-SA_other_evaluation.csv", "w") as file:
    file.write(cad)