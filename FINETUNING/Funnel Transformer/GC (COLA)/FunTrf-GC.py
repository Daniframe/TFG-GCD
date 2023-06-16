import os

# Using only one GPU, they are being friendly enough
# to let me use the server, don't get greedy :D
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

remote_username = "daniroalv"
remote_host = "falco.dsic.upv.es"
current_dir = os.getcwd()

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
from itertools import product
from copy import deepcopy
import numpy as np
import subprocess
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

# Dataset loading: cola
cola = load_dataset("glue", "cola")

# Model and tokenizer checkpoint
checkpoint = "funnel-transformer/small"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Function to use in map: tokenization
def tok_map(sample):
    return tokenizer(sample["sentence"], truncation = True, padding = "max_length", max_length = 512)

# Tokenization and preparation of dataset to the trainer
tokenized_dataset = cola.map(tok_map, batched = True)
tokenized_dataset.rename_column("label", "labels")

cola_train = tokenized_dataset["train"]
cola_val = tokenized_dataset["validation"]

# Hyperameters to optimise:
ini_learning_rate = [1e-5, 5e-5, 1e-4]
n_epochs = [5, 7, 10]

torch.manual_seed(8888)
cad_scores = "learning_rate;n_epochs;score\n"
first_train = True
first_val = True

for n_ep, ini_lr in product(n_epochs, ini_learning_rate):

    # Model: DistilBERT with a classification head of 2 classes
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

    # Computation of metrics: f1 score to deal with imbalanced datasts
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {"accuracy" : acc, "f1" : f1}

    output_dir = rf"./FunTrf-GC_lr-{ini_lr}_nep-{n_ep}"
    batch_size = 32
    training_args = TrainingArguments(
        output_dir = output_dir,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size = batch_size,
        num_train_epochs = n_ep,
        learning_rate = ini_lr,
        evaluation_strategy = "epoch",
        weight_decay = 0.01,
        logging_steps = len(cola_train) // batch_size,
        fp16 = True,
        seed = 8888,
        save_strategy = "epoch",
        save_total_limit = 1
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = cola_train,
        eval_dataset = cola_val,
        compute_metrics = compute_metrics
    )

    trainer.add_callback(CustomCallback(trainer))
    trainer.train()

    preds_train = trainer.predict(cola_train)
    preds_val = trainer.predict(cola_val)

    predicted_labels_train = preds_train.predictions.argmax(-1)
    predicted_labels_val = preds_val.predictions.argmax(-1)

    train_f1 = preds_train.metrics["test_f1"]
    val_f1 = preds_val.metrics["test_f1"]

    score = 0.2 * train_f1 + 0.8 * val_f1
    cad_scores += f"{ini_lr};{n_ep};{score}\n"

    with open(rf"./FunTrf-GC_lr-{ini_lr}_nep-{n_ep}_train.csv", "w", encoding = "utf-8") as f:

        if first_train:
            f.write("idx;prediction\n")
            first_train = False

        for i in range(len(cola_train)):
            f.write(f"{cola_train[i]['idx']};{predicted_labels_train[i]}\n")

    with open(rf"./FunTrf-GC_lr-{ini_lr}_nep-{n_ep}_val.csv", "w", encoding = "utf-8") as f:

        if first_val:
            f.write("idx;prediction\n")
            first_val = False

        for i in range(len(cola_val)):
            f.write(f"{cola_val[i]['idx']};{predicted_labels_val[i]}\n")

    with open(rf"./FunTrf-GC_hyperparamter_evaluation.csv", "a", encoding = "utf-8") as f:
        f.write(cad_scores)
    
    cad_scores = ""