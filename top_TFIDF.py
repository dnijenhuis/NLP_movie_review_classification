# This module extracts the top TF-IDF n-grams for POS and NEG reviews.
# It creates two CSV files:
# 1. Top 50 per class by raw average TF-IDF.
# 2. Top 50 per class by class-distinctive mean TF-IDF difference.

from pathlib import Path  # Needed for path handling.

import joblib  # Needed for loading the TF-IDF bundle.
import numpy as np  # Needed for numeric operations.
import pandas as pd  # Needed for writing the output CSV files.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
IN_FILE = DATA_DIR / "tfidf_train_val_bundle.joblib"
OUT_FILE_RAW = DATA_DIR / "top_tfidf_ngrams_per_class.csv"
OUT_FILE_DISTINCTIVE = DATA_DIR / "top_tfidf_ngrams_per_class_distinctive.csv"

# Output configuration.
TOP_K = 50


def main() -> None:
    # Load the TF-IDF bundle.
    bundle = joblib.load(IN_FILE)

    # Extract the training features, labels, and vectorizer.
    X_train = bundle["X_train"]
    y_train = np.asarray(bundle["y_train"])
    vectorizer = bundle["vectorizer"]

    # Get the feature names.
    feature_names = vectorizer.get_feature_names_out()

    # Create masks for POS and NEG rows.
    pos_mask = y_train == 1
    neg_mask = y_train == 0

    # Calculate the average TF-IDF score per class.
    pos_mean = np.asarray(X_train[pos_mask].mean(axis=0)).ravel()
    neg_mean = np.asarray(X_train[neg_mask].mean(axis=0)).ravel()

    # Get the indices of the top raw TF-IDF features per class.
    pos_raw_top_idx = np.argsort(pos_mean)[::-1][:TOP_K]
    neg_raw_top_idx = np.argsort(neg_mean)[::-1][:TOP_K]

    # Store the top POS n-grams by raw average TF-IDF.
    pos_raw_out = pd.DataFrame({
        "class": "POS",
        "rank": range(1, TOP_K + 1),
        "ngram": feature_names[pos_raw_top_idx],
        "mean_tfidf": pos_mean[pos_raw_top_idx],
    })

    # Store the top NEG n-grams by raw average TF-IDF.
    neg_raw_out = pd.DataFrame({
        "class": "NEG",
        "rank": range(1, TOP_K + 1),
        "ngram": feature_names[neg_raw_top_idx],
        "mean_tfidf": neg_mean[neg_raw_top_idx],
    })

    # Combine and save the raw TF-IDF output.
    raw_out_df = pd.concat([pos_raw_out, neg_raw_out], ignore_index=True)
    raw_out_df.to_csv(OUT_FILE_RAW, index=False)

    # Calculate class-distinctive TF-IDF scores.
    pos_distinctive = pos_mean - neg_mean
    neg_distinctive = neg_mean - pos_mean

    # Get the indices of the top distinctive features per class.
    pos_distinctive_top_idx = np.argsort(pos_distinctive)[::-1][:TOP_K]
    neg_distinctive_top_idx = np.argsort(neg_distinctive)[::-1][:TOP_K]

    # Store the top POS n-grams by class distinctiveness.
    pos_distinctive_out = pd.DataFrame({
        "class": "POS",
        "rank": range(1, TOP_K + 1),
        "ngram": feature_names[pos_distinctive_top_idx],
        "mean_tfidf_pos": pos_mean[pos_distinctive_top_idx],
        "mean_tfidf_neg": neg_mean[pos_distinctive_top_idx],
        "distinctiveness_score": pos_distinctive[pos_distinctive_top_idx],
    })

    # Store the top NEG n-grams by class distinctiveness.
    neg_distinctive_out = pd.DataFrame({
        "class": "NEG",
        "rank": range(1, TOP_K + 1),
        "ngram": feature_names[neg_distinctive_top_idx],
        "mean_tfidf_neg": neg_mean[neg_distinctive_top_idx],
        "mean_tfidf_pos": pos_mean[neg_distinctive_top_idx],
        "distinctiveness_score": neg_distinctive[neg_distinctive_top_idx],
    })

    # Combine and save the class-distinctive output.
    distinctive_out_df = pd.concat([pos_distinctive_out, neg_distinctive_out], ignore_index=True)
    distinctive_out_df.to_csv(OUT_FILE_DISTINCTIVE, index=False)

    print("")
    print(f"Stored top TF-IDF n-grams by raw average TF-IDF to: {OUT_FILE_RAW}")
    print(f"Stored top TF-IDF n-grams by class distinctiveness to: {OUT_FILE_DISTINCTIVE}")


if __name__ == "__main__":
    main()