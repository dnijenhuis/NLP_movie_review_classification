# This module converts the reviews in the training CSV to a TF-IDF representation. This is later
# used by the MNB- and SVM-models. The test CSV is, naturally, not used at all in this module.
# It reads the preprocessed training CSV, splits the data into training and validation
# sets based on the column 'validation_set', and converts the review text to TF-IDF features.
# Only using the non-validation reviews is important because it prevents data leakage
# from the validation set into the training process. The validation reviews are still transformed to TF-IDF so
# the trained model can be evaluated. The .joblib, however, is used later on in the pipeline for
# transforming the test-CSV into a TF-IDF representation in order to test the model.
# The resulting matrices, labels, indices and vectorizer are stored in a .joblib so they can be
# reused later without having to execute this module again.

from pathlib import Path  # Needed for path handling.

import joblib  # Needed for saving the TF-IDF output.
import pandas as pd  # Needed for the CSV.
from sklearn.feature_extraction.text import TfidfVectorizer  # Needed for TF-IDF.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
TRAIN_FILE = "imdb_train_preprocessed_MNB_and_SVM.csv"
BUNDLE_FILE = "tfidf_train_val_bundle.joblib"

# Column names.
LABEL_COL = "POS"
SPLIT_COL = "validation_set"
TEXT_COL = "Review_preprocessed"

# TF-IDF configuration.
NGRAM_RANGE = (1, 3) # The range of the ngrams.
MIN_DF = 2 # Terms are ignored that appear in fewer than this number of reviews.
MAX_DF = 0.95 # Terms are ignored that appear in more than this percentage of reviews.
MAX_FEATURES = 7500 # maximum number of features kept.
SUBLINEAR_TF = True # TF-IDF uses a function with log instead of 'simple' term frequency.

def main() -> None:
    # Load the preprocessed training CSV.
    train_path = DATA_DIR / TRAIN_FILE
    out_path = DATA_DIR / BUNDLE_FILE
    df = pd.read_csv(train_path)

    # Convert labels and split column to integers.
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL]).astype(int)
    df[SPLIT_COL] = pd.to_numeric(df[SPLIT_COL]).astype(int)

    # Split training and validation rows.
    train_mask = df[SPLIT_COL] == 0
    val_mask = df[SPLIT_COL] == 1

    # Extract training and validation texts and labels.
    x_train_text = df.loc[train_mask, TEXT_COL].fillna("").astype(str).tolist()
    y_train = df.loc[train_mask, LABEL_COL].to_numpy()

    x_val_text = df.loc[val_mask, TEXT_COL].fillna("").astype(str).tolist()
    y_val = df.loc[val_mask, LABEL_COL].to_numpy()

    # Store row indices for traceability.
    idx_train = df.index[train_mask].to_numpy()
    idx_val = df.index[val_mask].to_numpy()

    # Initialize the TF-IDF vectorizer.
    vectorizer = TfidfVectorizer(
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        max_features=MAX_FEATURES,
        sublinear_tf=SUBLINEAR_TF,
    )

    # Fit TF-IDF only on the training data, but transform both the train and validation data.
    X_train = vectorizer.fit_transform(x_train_text)
    X_val = vectorizer.transform(x_val_text)

    # Bundle all relevant objects together for later reuse into a dictionary.
    bundle = {
        "vectorizer": vectorizer,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "idx_train": idx_train,
        "idx_val": idx_val,
        "text_col": TEXT_COL,
        "tfidf_params": {
            "ngram_range": NGRAM_RANGE,
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "max_features": MAX_FEATURES,
            "sublinear_tf": SUBLINEAR_TF,
        },
    }

    # Save the bundle to disk.
    joblib.dump(bundle, out_path, compress=3)

    print("")
    print(f"Wrote: {out_path}")
    print(f"Train rows: {X_train.shape[0]:,}")
    print(f"Validation rows: {X_val.shape[0]:,}")
    print(f"TF-IDF features: {X_train.shape[1]:,} (cap={MAX_FEATURES:,})")


if __name__ == "__main__":
    main()