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
    "XLNet-HSOL_lr-5e-05_nep-7/checkpoint-6076")

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
hso_test = Trf2.PerturbedDataset(
    load_dataset("hate_speech_offensive")["train"].train_test_split(
        test_size = 0.2, seed = 8888, stratify_by_column = "class")["test"])

pipeline = TextClassificationPipeline(model = best_model, tokenizer = tokenizer, top_k = 1)

predictions_normal = Trf2.predict_dataset(
    dataset = hso_test,
    pipe = pipeline,
    x_fields = "tweet")

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.01, 0.05, 0.1]:
        perturbed_dataset = load_from_disk(
            f"../../hsol_perturbed/hsol_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "tweet")

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"XLNet-HSOL_character_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.1, 0.2, 0.3]:
        perturbed_dataset = load_from_disk(
            f"../../hsol_perturbed/hsol_perturbed_{perturbation.lower()}_{perturbation_proportion}")

        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "tweet")

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"XLNet-HSOL_word_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;kappa\n"
for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:
    perturbed_dataset = load_from_disk(
        f"../../hsol_perturbed/hsol_perturbed_{perturbation.lower()}")
    
    predictions_perturbed = Trf2.predict_dataset(
        dataset = perturbed_dataset,
        pipe =  pipeline,
        x_fields = "tweet")

    kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
    cad += f"{perturbation};{kappa}\n"

with open(f"XLNet-HSOL_other_evaluation.csv", "w") as file:
    file.write(cad)