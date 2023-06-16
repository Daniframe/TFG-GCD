import os

# Using only one GPU to avoid server congestion
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_dataset
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

# Dataset loading: cola
cola = load_dataset("glue", "cola")

# Model and tokenizer checkpoint
checkpoint = "google/electra-base-discriminator"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Function to use in map: tokenization
def tok_map(sample):
    return tokenizer(sample["sentence"], truncation = True, padding = "max_length")

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
cad_opt = "idx;prediction\n"

for n_ep, ini_lr in product(n_epochs, ini_learning_rate):

    # Model: DistilBERT with a classification head of 2 classes
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels = 2)

    # Computation of metrics: accuracy and f1 score to deal with imbalanced datasts
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds)
        acc = accuracy_score(labels, preds)
        return {"accuracy" : acc, "f1" : f1}

    output_dir = rf"./ELECTRA-GC_lr-{ini_lr}_nep-{n_ep}"
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
        seed = 8888
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        train_dataset = cola_train,
        eval_dataset = cola_val,
        compute_metrics = compute_metrics
    )

    # Uncomment the following line if you want to compute the metrics
    # for the training dataset each epoch

    # trainer.add_callback(CustomCallback(trainer)
    trainer.train()

    preds_train = trainer.predict(cola_train)
    preds_val = trainer.predict(cola_val)

    predicted_labels_train = preds_train.predictions.argmax(-1)
    predicted_labels_val = preds_val.predictions.argmax(-1)

    train_f1 = preds_train.metrics["test_f1"]
    val_f1 = preds_val.metrics["test_f1"]

    # Custom score: 20% f1-score train, 80% f1-score validation
    score = 0.2 * train_f1 + 0.8 * val_f1
    cad_scores += f"{ini_lr};{n_ep};{score}\n"

    # Adhering to good practices in ML: providing results to avoid recomputation
    with open(rf"./ELECTRA-GC_lr-{ini_lr}_nep-{n_ep}_train.csv", "w", encoding = "utf-8") as f:
        for i, pred in enumerate(predicted_labels_train):
            cad_opt += f"{i};{pred}\n"
            f.write(cad_opt)
            cad_opt = ""

    with open(rf"./ELECTRA-GC_lr-{ini_lr}_nep-{n_ep}_train.csv", "w", encoding = "utf-8") as f:
        f.write("idx;prediction\n")
        for i, pred in enumerate(predicted_labels_train):
            cad_opt += f"{i};{pred}\n"
            f.write(cad_opt)
            cad_opt = ""

    with open(rf"./ELECTRA-GC_hyperameter_evaluation.csv", "a", encoding = "utf-8") as f:
        f.write(cad_scores)
    
    cad_scores = ""