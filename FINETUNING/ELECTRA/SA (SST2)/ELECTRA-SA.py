import os
import sys

# Using only one GPU to avoid servr congestion
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Path to the Transformations library. Change it to where Transformations.py
# is located or get the library file in the same working directory as this file
sys.path.append("/home/daniroalv/miniconda3/envs/CodigoTFG/CodigoTFG")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
import Transformations as Trf2
from itertools import product
from copy import deepcopy
import torch

# Custom callback to also obtain the metrics of the training dataset
class CustomCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset = self._trainer.train_dataset, metric_key_prefix = "train")
            return control_copy

# Dataset loading: sst2
sst2 = load_dataset("glue", "sst2")
sst2_train_temp = Trf2.PerturbedDataset(sst2["train"])
sst2_val_temp = sst2["validation"]

#Subsample train dataset to 15000 rows
sst2_train_temp = sst2_train_temp.subsample(n = 15000, y_field = "label")

# Model and tokenizer checkpoint
checkpoint = "google/electra-base-discriminator"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Function to use in map: tokenization
def tok_map(sample):
    return tokenizer(sample["sentence"], truncation = True, padding = "max_length")#, max_length = 96)

# Tokenization and preparation of dataset to the trainer
sst2_train = sst2_train_temp.map(tok_map, batched = True)
sst2_val = sst2_val_temp.map(tok_map, batched = True)

sst2_train = sst2_train.rename_column("label", "labels")
sst2_val = sst2_val.rename_column("label", "labels")

# Hyperameters to optimise:
ini_learning_rate = [1e-5, 5e-5, 1e-4]
n_epochs = [5, 7, 10]

torch.manual_seed(8888)
cad_scores = "learning_rate;n_epochs;score\n"
first_train = True
first_val = True

for n_ep, ini_lr in product(n_epochs, ini_learning_rate):

    # Model: ELECTRA with a classification head of 2 classes
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

    # Computation of metrics: accuracy f1 score to deal with imbalanced datasts
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {"accuracy" : acc, "f1" : f1}

    output_dir = rf"./ELECTRA-SA_lr-{ini_lr}_nep-{n_ep}"
    batch_size = 32
    training_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = n_ep,
        learning_rate = ini_lr,
        evaluation_strategy = "epoch",
        weight_decay = 0.01,
        logging_steps = len(sst2_train) // batch_size,
        fp16 = True,
        seed = 8888,
        save_strategy = "epoch",
        save_total_limit = 1
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = sst2_train,
        eval_dataset = sst2_val,
        compute_metrics = compute_metrics
    )

    # Uncomment the following line if you want to compute the metrics
    # for the training dataset each epoch

    # trainer.add_callback(CustomCallback(trainer)
    trainer.train()

    preds_train = trainer.predict(sst2_train)
    preds_val = trainer.predict(sst2_val)

    predicted_labels_train = preds_train.predictions.argmax(-1)
    predicted_labels_val = preds_val.predictions.argmax(-1)

    train_f1 = preds_train.metrics["test_f1"]
    val_f1 = preds_val.metrics["test_f1"]

    # Custom score: 20% f1-score train, 80% f1-score validation
    score = 0.2 * train_f1 + 0.8 * val_f1
    cad_scores += f"{ini_lr};{n_ep};{score}\n"

    # Adhering to good practices in ML: providing results to avoid recomputation
    with open(rf"./ELECTRA-SA_lr-{ini_lr}_nep-{n_ep}_train.csv", "w", encoding = "utf-8") as f:

        if first_train:
            f.write("idx;prediction\n")
            first_train = False

        for i in range(len(sst2_train)):
            f.write(f"{sst2_train[i]['idx']};{predicted_labels_train[i]}\n")

    with open(rf"./ELECTRA-SA_lr-{ini_lr}_nep-{n_ep}_val.csv", "w", encoding = "utf-8") as f:

        if first_val:
            f.write("idx;prediction\n")
            first_val = False

        for i in range(len(sst2_val)):
            f.write(f"{sst2_val[i]['idx']};{predicted_labels_val[i]}\n")

    with open(rf"./ELECTRA-SA_hyperparameter_evaluation.csv", "a", encoding = "utf-8") as f:
        f.write(cad_scores)
    
    cad_scores = ""
