# This module trains a Multinomial Naive Bayes model on the BOW training CSV.
# It uses the non-validation rows for training and the validation rows for validation.
# The trained model is stored in a .joblib file.
# The evaluation results are appended to a CSV with model results.

from pathlib import Path  # Needed for path handling.
from datetime import datetime  # Needed for adding an ISO datetime stamp.
import time  # Needed for measuring training time.
import joblib  # Needed for saving the trained model.
import pandas as pd  # Needed for reading and writing CSV files.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Evaluation.
from sklearn.naive_bayes import MultinomialNB  # Needed for the MNB model.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
IN_FILE = DATA_DIR / "imdb_train_preprocessed_BOW.csv"
OUT_MODEL = DATA_DIR / "mnb_bow_model.joblib"
RESULTS_FILE = DATA_DIR / "validation_metrics.csv"

# Column names.
LABEL_COL = "POS"
VAL_COL = "validation_set"
BOW_PREFIX = "bow_"

# Model configuration.
ALPHA = 1.0 # Helps with the zero-probability problem.


def main():
    # Load the training CSV.
    df = pd.read_csv(IN_FILE)

    # Extract the BOW columns.
    bow_cols = [col for col in df.columns if col.startswith(BOW_PREFIX)]

    # Extract labels.
    y = df[LABEL_COL].astype(int)

    # Create the train and validation masks.
    val_mask = df[VAL_COL].astype(bool)
    train_mask = ~val_mask

    # Extract the BOW features.
    X = df[bow_cols].fillna(0)

    # Split the data into training and validation sets.
    X_train = X.loc[train_mask]
    y_train = y.loc[train_mask]
    X_val = X.loc[val_mask]
    y_val = y.loc[val_mask]

    # Initialize the model.
    model = MultinomialNB(alpha=ALPHA)

    # Train the model.
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()
    train_time_seconds = end_train - start_train

    # Predict the validation labels.
    y_pred = model.predict(X_val)

    # Calculate the evaluation metrics.
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f_score = f1_score(y_val, y_pred)

    # Store the trained model and metadata.
    payload = {
        "model": model,
        "bow_columns": bow_cols,
        "label_col": LABEL_COL,
        "validation_col": VAL_COL,
        "alpha": ALPHA,
        "input_file": str(IN_FILE),
        "training_time_seconds": train_time_seconds,
    }

    joblib.dump(payload, OUT_MODEL)

    # Create one results row.
    results_row = pd.DataFrame(
        [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "MNB_BOW",
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f-score": f_score,
            "training_time_seconds": train_time_seconds,
        }]
    )

    # Append the results to the results CSV.
    if RESULTS_FILE.exists():
        existing_df = pd.read_csv(RESULTS_FILE)
        results_df = pd.concat([existing_df, results_row], ignore_index=True)
    else:
        results_df = results_row

    # Put the timestamp column in the first position.
    first_col = results_df.pop("timestamp")
    results_df.insert(0, "timestamp", first_col)

    # Save the updated results CSV.
    results_df.to_csv(RESULTS_FILE, index=False)

    print("MNB_BOW model trained and evaluated.")


if __name__ == "__main__":
    main()