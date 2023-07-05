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
    "DistilBERT-NLI_lr-5e-05_nep-7/checkpoint-3283")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
mnli_test = Trf2.PerturbedDataset(load_dataset("glue", "mnli")["test_matched"])

pipeline = TextClassificationPipeline(model = best_model, tokenizer = tokenizer, top_k = 1)

predictions_normal = Trf2.predict_dataset(
    dataset = mnli_test,
    pipe = pipeline,
    x_fields = ["premise", "hypothesis"],
    mode = "multiple")

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
# for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
#     for perturbation_proportion in [0.01, 0.05, 0.1]:
#         perturbed_dataset = load_from_disk(
#             f"../../mnli_perturbed/mnli_perturbed_{perturbation.lower()}_{perturbation_proportion}")
#         predictions_perturbed = Trf2.predict_dataset(
#             dataset = perturbed_dataset,
#             pipe =  pipeline,
#             x_fields = ["premise", "hypothesis"],
#             mode = "multiple")

#         kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
#         cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

# with open(f"DistilBERT-NLI_character_level_evaluation.csv", "w") as file:
#     file.write(cad)

# cad = "type;percentage_perturbation;percentage_altered;kappa\n"
# for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
#     for perturbation_proportion in [0.1, 0.2, 0.3]:
#         perturbed_dataset = load_from_disk(
#             f"../../mnli_perturbed/mnli_perturbed_{perturbation.lower()}_{perturbation_proportion}")

#         predictions_perturbed = Trf2.predict_dataset(
#             dataset = perturbed_dataset,
#             pipe =  pipeline,
#             x_fields = ["premise", "hypothesis"],
#             mode = "multiple")

#         kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
#         cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

# with open(f"DistilBERT-NLI_word_level_evaluation.csv", "w") as file:
#     file.write(cad)

# cad = "type;percentage_perturbation;kappa\n"
cad = ""
for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:
    if perturbation == "WordCase":
        try:
            perturbed_dataset = load_from_disk(
                f"../../mnli_perturbed/mnli_perturbed_{perturbation.lower()}")

            predictions_perturbed = Trf2.predict_dataset(
                dataset = perturbed_dataset,
                pipe =  pipeline,
                x_fields = ["premise", "hypothesis"],
                mode = "multiple",
                exclude_samples = [458, 5876, 7020])

            kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
            cad += f"{perturbation};{kappa}\n"
        except:
            pass

with open(f"DistilBERT-NLI_other_evaluation.csv", "a") as file:
    file.write(cad)