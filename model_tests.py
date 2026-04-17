# This module tests the trained MNB, SVM, and ROBERTA models on the test set.
# It stores the test evaluation metrics in a CSV.

from pathlib import Path  # Needed for path handling.
from datetime import datetime  # Needed for adding an ISO datetime stamp.
import joblib  # Needed for loading trained classical models.
import numpy as np  # Needed for arrays.
import pandas as pd  # Needed for reading and writing CSV files.
import torch  # Needed for the transformer model.
from sklearn.feature_extraction.text import CountVectorizer  # Needed for rebuilding the BOW representation.
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support  # Evaluation.
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Transformer model and tokenizer.

from train_BERT import MAX_LENGTH  # Use the same max length as during training.

# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
TEST_CLASSICAL = DATA_DIR / "imdb_test_preprocessed_MNB_and_SVM.csv" # CSV with test data, pre-processed.
TEST_ROBERTA = DATA_DIR / "imdb_test_preprocessed_for_BERT.csv" # CSV with test data, pre-processed for BERT.
OUT_FILE = DATA_DIR / "test_metrics.csv" # CSV with test metrics.

MNB_MODEL = DATA_DIR / "mnb_bow_model.joblib" # MNB model parameters.
SVM_MODEL = DATA_DIR / "svm_tfidf_model.joblib" # SVM model parameters.
ROBERTA_MODEL_DIR = DATA_DIR / "ROBERTA_imdb_model" # ROBERTA model parameters.


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # This function calculates the test metrics.
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f-score": float(f_score),
    }


def run_mnb() -> dict[str, float]:
    # Load the preprocessed test CSV.
    df = pd.read_csv(TEST_CLASSICAL)

    # Extract the test texts and labels.
    text_col = "Review_preprocessed"
    texts = df[text_col].fillna("").astype(str).tolist()
    y_true = df["POS"].astype(int).to_numpy()

    # Load the trained MNB model and BOW columns.
    payload = joblib.load(MNB_MODEL)
    model = payload["model"]
    bow_columns = payload["bow_columns"]

    # Rebuild the BOW representation using the stored vocabulary.
    terms = [term.replace("bow_", "", 1) for term in bow_columns]
    vocabulary = {term: i for i, term in enumerate(terms)}
    vectorizer = CountVectorizer(vocabulary=vocabulary)
    X_test = vectorizer.transform(texts)

    # Convert to DataFrame so feature names match training.
    X_test_df = pd.DataFrame.sparse.from_spmatrix(X_test, columns=bow_columns)

    # Predict the test labels.
    y_pred = model.predict(X_test_df)

    return compute_metrics(y_true, y_pred)


def run_svm() -> dict[str, float]:
    # Load the preprocessed test CSV.
    df = pd.read_csv(TEST_CLASSICAL)

    # Load the trained SVM payload.
    payload = joblib.load(SVM_MODEL)
    model = payload["model"]
    vectorizer = payload["vectorizer"]
    text_col = payload["text_col"]

    # Extract the test texts and labels.
    texts = df[text_col].fillna("").astype(str).tolist()
    y_true = df["POS"].astype(int).to_numpy()

    # Transform the raw test texts to TF-IDF features.
    X_test = vectorizer.transform(texts)

    # Predict the test labels.
    y_pred = model.predict(X_test)

    return compute_metrics(y_true, y_pred)


def run_roberta() -> dict[str, float]:
    # Load the test CSV specifically pre-processed for ROBERTA/BERT.
    df = pd.read_csv(TEST_ROBERTA)

    # Extract the test texts and labels.
    texts = df["Review_preprocessed_BERT"].fillna("").astype(str).tolist()
    y_true = df["POS"].astype(int).to_numpy()

    # If a GPU is available, use it. Otherwise use CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Converts text into token IDs.
    tokenizer = AutoTokenizer.from_pretrained(str(ROBERTA_MODEL_DIR), local_files_only=True)

    # Loads model locally (and only locally).
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ROBERTA_MODEL_DIR),
        local_files_only=True,
    )
    model.to(device)
    model.eval() # Set to evaluation mode.

    # Predict the test labels in batches.
    batch_size = 32 # Batch size of number of rows.
    preds = []

    with torch.no_grad(): # Disable gradient tracking because this is the test, not training.
        for i in range(0, len(texts), batch_size):
            # Loop through test set in batches.
            batch_texts = texts[i:i + batch_size]

            encodings = tokenizer(
                # Encode raw review text to model inputs.
                batch_texts,
                padding=True, # Pads shorter sequences so all reviews in the batch have equal length.
                truncation=True, # Cuts of reviews longer than maximum length.
                max_length=MAX_LENGTH, # Use the same max length as during the (last) training run.
                return_tensors="pt",
            )

            encodings = {key: value.to(device) for key, value in encodings.items()}

            # Logits -> for binary classification (as in this project) the model outputs 2 scores
            # per review. Each score represents a raw score of how likely the datapoint belongs to that
            # score. These are raw scores, not probabilities.
            logits = model(**encodings).logits # Forward pass through the network.

            batch_preds = torch.argmax(logits, dim=1).cpu().numpy() # Convert logits to predicted class.
            preds.extend(batch_preds.tolist()) # Store predictions to list.

    y_pred = np.array(preds, dtype=np.int64) # Convert predictions to array.

    return compute_metrics(y_true, y_pred)


def main() -> None:
    # Create per model one row with the test metrics.
    rows = [
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "MNB_BOW",
            **run_mnb(),
        },
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "SVM_TFIDF",
            **run_svm(),
        },
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "model": "ROBERTA",
            **run_roberta(),
        },
    ]

    # Store the test metrics in a CSV with the same structure as validation_metrics.csv.
    out_df = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "model",
            "TP",
            "FP",
            "TN",
            "FN",
            "accuracy",
            "precision",
            "recall",
            "f-score",
        ],
    )

    out_df.to_csv(OUT_FILE, index=False)

    print("")
    print("Test metrics stored.")


if __name__ == "__main__":
    main()