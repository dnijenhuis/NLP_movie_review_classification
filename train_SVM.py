# This module trains an SVM model on the TF-IDF training and validation data.
# It uses the training part of the TF-IDF bundle for training and the validation
# part for validation. The trained model is stored in a .joblib file.
# The validation results are appended to an existing CSV with validation metrics.

from pathlib import Path  # Needed for path handling.
from datetime import datetime  # Needed for adding an ISO datetime stamp.
import time  # Needed for measuring training time.
import joblib  # Needed for loading and saving .joblib files.
import pandas as pd  # Needed for reading and writing CSV files.
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  # Evaluation.
from sklearn.svm import LinearSVC  # Needed for the SVM model.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
IN_FILE = DATA_DIR / "tfidf_train_val_bundle.joblib"
OUT_MODEL = DATA_DIR / "svm_tfidf_model.joblib"
RESULTS_FILE = DATA_DIR / "validation_metrics.csv"

# SVM configuration.
C = 0.1  # Regularization strength, reduces overfitting.
MAX_ITER = 1000  # Maximum number of training iterations.


def main():
    # Load the TF-IDF bundle.
    bundle = joblib.load(IN_FILE)

    # Extract the TF-IDF features and labels.
    X_train = bundle["X_train"]
    X_val = bundle["X_val"]
    y_train = bundle["y_train"]
    y_val = bundle["y_val"]

    # Initialize the SVM model.
    model = LinearSVC(
        C=C,
        max_iter=MAX_ITER,
    )

    # Train the model.
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()
    train_time_seconds = end_train - start_train

    # Predict the validation labels.
    y_pred = model.predict(X_val)

    # Calculate the validation metrics.
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f_score = f1_score(y_val, y_pred)

    # Store the trained model and metadata.
    payload = {
        "model": model,
        "vectorizer": bundle["vectorizer"],
        "text_col": bundle["text_col"],
        "tfidf_params": bundle["tfidf_params"],
        "C": C,
        "max_iter": MAX_ITER,
        "input_file": str(IN_FILE),
        "training_time_seconds": train_time_seconds,
    }

    joblib.dump(payload, OUT_MODEL)

    # Create one validation results row.
    results_row = pd.DataFrame(
        [{
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "SVM_TFIDF",
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
    print("SVM_TFIDF model trained and evaluated.")


if __name__ == "__main__":
    main()