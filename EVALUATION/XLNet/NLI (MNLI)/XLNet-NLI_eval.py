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
    "XLNet-NLI_lr-1e-05_nep-10/checkpoint-9380")

tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
mnli_test = Trf2.PerturbedDataset(load_dataset("glue", "mnli")["test_matched"])

pipeline = TextClassificationPipeline(model = best_model, tokenizer = tokenizer, top_k = 1, device = 0)

predictions_normal = Trf2.predict_dataset(
    dataset = mnli_test,
    pipe = pipeline,
    x_fields = ["premise", "hypothesis"],
    mode = "multiple")

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.01, 0.05, 0.1]:
        perturbed_dataset = load_from_disk(
            f"../../mnli_perturbed/mnli_perturbed_{perturbation.lower()}_{perturbation_proportion}")

        try:
            predictions_perturbed = Trf2.predict_dataset(
                dataset = perturbed_dataset,
                pipe =  pipeline,
                x_fields = ["premise", "hypothesis"],
                mode = "multiple")
        except:
            pass

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"XLNet-NLI_character_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.1, 0.2, 0.3]:
        perturbed_dataset = load_from_disk(
            f"../../mnli_perturbed/mnli_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        try:
            predictions_perturbed = Trf2.predict_dataset(
                dataset = perturbed_dataset,
                pipe =  pipeline,
                x_fields = ["premise", "hypothesis"],
                mode = "multiple")
        except:
            pass

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"XLNet-NLI_word_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;kappa\n"
for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:
    perturbed_dataset = load_from_disk(
        f"../../mnli_perturbed/mnli_perturbed_{perturbation.lower()}")
    
    try:
        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = ["premise", "hypothesis"],
            mode = "multiple")
    except:
        pass

    kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
    cad += f"{perturbation};{kappa}\n"

with open(f"XLNet-NLI_other_evaluation.csv", "w") as file:
    file.write(cad)