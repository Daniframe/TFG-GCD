import os
import sys

# Using only one GPU to avoid server congestion
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# To find my library
# Obviously change the path to where your environment is located
sys.path.append("/home/daniroalv/miniconda3/envs/CodigoTFG/CodigoTFG")

from transformers import (
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    AutoTokenizer
)

from sklearn.metrics import cohen_kappa_score
from datasets import load_dataset, load_from_disk
import Transformations as Trf2

# You will need a saved checkpoint to load the model. The one used here is in:
# https://drive.google.com/drive/folders/1rZ_JHXsZQG_VM5Lkbveufm4blSGhymXY?usp=sharing
best_model = AutoModelForSequenceClassification.from_pretrained(
    "DistilBERT-GC_lr-5e-05_nep-10/checkpoint-2680")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
cola_test = Trf2.PerturbedDataset(load_dataset("glue", "cola")["test"])

pipeline = TextClassificationPipeline(model = best_model, tokenizer = tokenizer, top_k = 1)

predictions_normal = Trf2.predict_dataset(
    dataset = cola_test,
    pipe = pipeline,
    x_fields = "sentence")

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_CHARACTER_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.01, 0.05, 0.1]:

        # The perturbed datasets are available in huggingface
        # in the username DaniFrame, in this case ColaPerturbed:
        # You can load the dataset as follows:

        perturbed_dataset = load_dataset(
            "DaniFrame/ColaPerturbed", 
            split = f"cola_perturbed_{perturbation.lower()}_{perturbation_proportion}"
        )

        # Alternatively, if you have them in disk, use this:

        # perturbed_dataset = load_from_disk(
        #     f"../../cola_perturbed/cola_perturbed_{perturbation.lower()}_{perturbation_proportion}")
        
        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "sentence")

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"DistilBERT-GC_character_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;percentage_altered;kappa\n"
for perturbation in Trf2.ALLOWED_WORD_LEVEL_PERTURBATIONS:
    for perturbation_proportion in [0.1, 0.2, 0.3]:

        perturbed_dataset = load_dataset(
            "DaniFrame/ColaPerturbed", 
            split = f"cola_perturbed_{perturbation.lower()}_{perturbation_proportion}"
        )

        # perturbed_dataset = load_from_disk(
        #     f"../../cola_perturbed/cola_perturbed_{perturbation.lower()}_{perturbation_proportion}")

        predictions_perturbed = Trf2.predict_dataset(
            dataset = perturbed_dataset,
            pipe =  pipeline,
            x_fields = "sentence")

        kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
        cad += f"{perturbation};{perturbation_proportion};{kappa}\n"

with open(f"DistilBERT-GC_word_level_evaluation.csv", "w") as file:
    file.write(cad)

cad = "type;percentage_perturbation;kappa\n"
for perturbation in Trf2.ALLOWED_OTHER_PERTURBATIONS:

    perturbed_dataset = load_dataset(
        "DaniFrame/ColaPerturbed", 
        split = f"cola_perturbed_{perturbation.lower()}"
    )

    # perturbed_dataset = load_from_disk(
    #     f"../../cola_perturbed/cola_perturbed_{perturbation.lower()}")

    predictions_perturbed = Trf2.predict_dataset(
        dataset = perturbed_dataset,
        pipe =  pipeline,
        x_fields = "sentence")

    kappa = cohen_kappa_score(predictions_normal, predictions_perturbed)
    cad += f"{perturbation};{kappa}\n"

with open(f"DistilBERT-GC_other_evaluation.csv", "w") as file:
    file.write(cad)