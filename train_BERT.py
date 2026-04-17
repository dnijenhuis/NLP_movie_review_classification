# This module trains a ROBERTA model on the IMDB training CSV.
# It uses the non-validation rows for training and the validation rows for evaluation.
# The trained model is stored, and validation metrics are saved to a CSV.
# Note: this module's name and several other modules use 'BERT' even though ROBERTA was used.
# The reason for this is that it only became certain at the end of development that specifically
# ROBERTA of the BERT-family would be used. References to BERT were not updated to reduce the
# chance of errors in references being caused.

from pathlib import Path  # Needed for path handling.
from datetime import datetime  # Needed for adding an ISO datetime stamp.
import json  # Needed for storing metrics as JSON.
import os  # Needed for setting the hash seed.
import time  # Needed for measuring training time.
from dataclasses import dataclass  # Needed for the dataset class.
from typing import Any  # Needed for flexible type hints.
import numpy as np  # Needed for arrays and argmax.
import pandas as pd  # Needed for reading and writing CSV files.
import torch  # Needed for the model and tensors.
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # Evaluation.
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer, TrainingArguments

# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
IN_FILE = DATA_DIR / "imdb_train_preprocessed_for_BERT.csv" # The training CSV pre-processed for BERT.
OUT_DIR = DATA_DIR / "ROBERTA_imdb_model" # Folder for ROBERTA.
METRICS_JSON = OUT_DIR / "metrics_val.json" # JSON file with metadata.
RESULTS_FILE = DATA_DIR / "validation_metrics.csv" # CSV with evaluation metrics.

# Column names.
TEXT_COL = "Review_preprocessed_BERT" # Column with reviews pre-processed for BERT.
LABEL_COL = "POS"
VAL_COL = "validation_set"

# Model configuration.
MODEL_NAME = "roberta-base" # Chosen model.
SEED = 20260228
MAX_LENGTH = 500 # Maximum  number of tokens per review taken into account. Average of set is 290.74.

# Training configuration.
BATCH_SIZE_TRAIN = 8 # Training examples per batch.
BATCH_SIZE_EVAL = 16 # Evaluation examples per batch.
EPOCHS = 1 # Number of training epochs.
LR = 0.00001 # Learning rate, should be balanced. Too high -> overshoot optimum, too low -> get stuck in local optimum.
WEIGHT_DECAY = 0.01 # Regularization to reduce overfitting.


# This dataset wrapper provides tokenized inputs and labels to the Trainer.
@dataclass
class SimpleDataset(torch.utils.data.Dataset): # Defines a dataset object.
    encodings: dict[str, Any] # Tokenized text.
    labels: np.ndarray  # Numeric class labels.

    def __len__(self) -> int: # Returns the number of examples as int.
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # For 1 row, creates dictionary containing input_ids, attention_mask, labels.
        # Converts to torch.tensor.
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item

def set_seed(seed: int) -> None:
    # This function sets random seeds for reproducibility.
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(eval_pred) -> dict[str, float]:
    # This function calculates the validation metrics for the Trainer.
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f_score),
    }


def main() -> None:
    # Read the input CSV.
    df = pd.read_csv(IN_FILE)

    # Standardize the label and validation columns.
    df[LABEL_COL] = df[LABEL_COL].astype(int) # Labels are integers.
    df[VAL_COL] = df[VAL_COL].astype(bool) # Validation is boolean.

    # Split the data into training and validation sets.
    train_df = df.loc[~df[VAL_COL]].copy() # Rows where validation_set is False go to training.
    val_df = df.loc[df[VAL_COL]].copy() # Rows where validation_set is True go to validation.

    # Set the random seed before tokenization and training.
    set_seed(SEED)

    # Load the tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    # Tokenize the training and validation texts.
    train_encodings = tokenizer(
        train_df[TEXT_COL].fillna("").tolist(),
        truncation=True,
        max_length=MAX_LENGTH,
    )

    val_encodings = tokenizer(
        val_df[TEXT_COL].fillna("").tolist(),
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Create the datasets for the Trainer.
    train_dataset = SimpleDataset(
        encodings=train_encodings,
        labels=train_df[LABEL_COL].to_numpy(dtype=np.int64),
    )

    val_dataset = SimpleDataset(
        encodings=val_encodings,
        labels=val_df[LABEL_COL].to_numpy(dtype=np.int64),
    )

    # Load the classification model.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
    )

    # Ensure the output directory exists.
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define the training configuration.
    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        seed=SEED,
        eval_strategy="epoch", # Evaluate after each epoch.
        save_strategy="epoch", # Save checkpoint each epoch.
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE_TRAIN,
        per_device_eval_batch_size=BATCH_SIZE_EVAL,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True, # Reload best checkpoint after training.
        metric_for_best_model="f1", # Best model is chosen based on F-score, this is an 'overall' type of metric.
        greater_is_better=True, # Higher F1 is better
        report_to=[],
        disable_tqdm=True, # Fewer prints in terminal.
        logging_strategy="no", # Fewer prints in terminal.
    )

    # Create the Trainer object.
    trainer = Trainer(
        model=model, # The model.
        args=training_args, # The arguments as set above.
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train the model and measure the training time.
    start_train = time.perf_counter()
    trainer.train()
    end_train = time.perf_counter()

    train_time_seconds = end_train - start_train
    print(f"\nTraining time (seconds): {train_time_seconds:.4f}")

    # Evaluate the trained model on the validation set.
    eval_metrics = trainer.evaluate()

    # Generate validation predictions.
    pred_output = trainer.predict(val_dataset)
    logits = pred_output.predictions
    preds = np.argmax(logits, axis=1)

    # Calculate the confusion matrix values.
    y_val = val_df[LABEL_COL].to_numpy(dtype=np.int64)
    tn, fp, fn, tp = confusion_matrix(y_val, preds, labels=[0, 1]).ravel()

    # Save the trained model and tokenizer.
    trainer.save_model(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))

    # Store the validation metrics and metadata as JSON.
    metrics_out = {
        "model_name": MODEL_NAME,
        "max_length": int(MAX_LENGTH),
        "epochs": int(EPOCHS),
        "lr": float(LR),
        "weight_decay": float(WEIGHT_DECAY),
        "train_size": int(len(train_df)),
        "val_size": int(len(val_df)),
        "training_time_seconds": float(train_time_seconds),
        "metrics": {key: float(value) for key, value in eval_metrics.items()},
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }

    METRICS_JSON.write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    # Create one validation results row.
    results_row = pd.DataFrame(
        [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "ROBERTA",
            "TP": int(tp),
            "FP": int(fp),
            "TN": int(tn),
            "FN": int(fn),
            "accuracy": float(eval_metrics["eval_accuracy"]),
            "precision": float(eval_metrics["eval_precision"]),
            "recall": float(eval_metrics["eval_recall"]),
            "f-score": float(eval_metrics["eval_f1"]),
            "training_time_seconds": float(train_time_seconds),
        }]
    )

    # Append the results to the existing validation metrics CSV.
    if RESULTS_FILE.exists():
        existing_df = pd.read_csv(RESULTS_FILE)
        results_df = pd.concat([existing_df, results_row], ignore_index=True)
    else:
        results_df = results_row

    # Put the timestamp column in the first position.
    first_col = results_df.pop("timestamp")
    results_df.insert(0, "timestamp", first_col)

    # Save the updated validation metrics CSV.
    results_df.to_csv(RESULTS_FILE, index=False)

    print("")
    print("ROBERTA model trained and evaluated.")


if __name__ == "__main__":
    main()