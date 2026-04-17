# This module adds Bag-of-Words columns to the preprocessed training CSV.
# It uses only the non-validation reviews to fit the BOW vectorizer.
# Then it transforms both the training and validation reviews with that same vectorizer.
# The original columns are kept and the BOW columns are appended.
# This prevents data leakage from the validation set into the feature engineering step.

import re  # Needed for converting tokens to a safe CSV column name.
from pathlib import Path  # Needed for path handling.

import pandas as pd  # Needed for CSV handling and building the BOW DataFrame.
from sklearn.feature_extraction.text import CountVectorizer  # Needed for BOW vectorization.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
IN_FILE = DATA_DIR / "imdb_train_preprocessed_MNB_and_SVM.csv"
OUT_FILE = DATA_DIR / "imdb_train_preprocessed_BOW.csv"

# Column names.
TEXT_COL = "Review_preprocessed"
SPLIT_COL = "validation_set"

# Prefix for the generated BOW columns.
BOW_PREFIX = "bow_"

# BOW configuration.
MAX_FEATURES = 7500  # Maximum number of features / columns.
NGRAM_RANGE = (1, 3)  # N-gram range.
MIN_DF = 2  # Terms are ignored if they appear in fewer than this number of training reviews.
MAX_DF = 0.95  # Terms are ignored if they appear in more than this fraction of training reviews.


def _safe_colname(term: str, max_len: int = 80) -> str:
    # Convert a term to a safe CSV column name.
    term = term.strip().lower()
    term = re.sub(r"\s+", "_", term)
    term = re.sub(r"[^a-z0-9_]+", "", term)
    return (BOW_PREFIX + term)[:max_len]


def _validate_input(df: pd.DataFrame) -> None:
    # Check whether the required columns exist.
    missing_cols = [col for col in [TEXT_COL, SPLIT_COL] if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required column(s): {missing_cols}")

    # Check whether the split column contains usable boolean values.
    split_values = df[SPLIT_COL].dropna().unique().tolist()
    invalid_values = [v for v in split_values if v not in [True, False]]
    if invalid_values:
        raise ValueError(
            f"Column '{SPLIT_COL}' contains unexpected values: {invalid_values}. "
            f"Expected only True and False."
        )

    # Check whether both train and validation rows are present.
    n_val = int(df[SPLIT_COL].sum())
    n_train = int((~df[SPLIT_COL]).sum())

    if n_train == 0:
        raise ValueError("No non-validation rows found. Cannot fit BOW vectorizer.")
    if n_val == 0:
        raise ValueError("No validation rows found. Expected a validation split in the input file.")


def main() -> None:
    # Load the preprocessed training CSV.
    df = pd.read_csv(IN_FILE)

    # Convert split column explicitly to boolean if needed.
    if df[SPLIT_COL].dtype != bool:
        if df[SPLIT_COL].dropna().isin([True, False]).all():
            df[SPLIT_COL] = df[SPLIT_COL].astype(bool)

    _validate_input(df)

    # Extract the review texts.
    texts_all = df[TEXT_COL].fillna("").astype(str)

    # Create masks for train and validation rows.
    is_val = df[SPLIT_COL]
    is_train = ~is_val

    # Use only non-validation reviews for fitting the vectorizer.
    texts_train = texts_all[is_train].tolist()

    # Initialize the BOW vectorizer.
    vectorizer = CountVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
    )

    # Fit on training reviews only.
    vectorizer.fit(texts_train)

    # Transform all rows with the fitted vectorizer.
    X_all = vectorizer.transform(texts_all.tolist())
    terms = vectorizer.get_feature_names_out()
    bow_cols = [_safe_colname(term) for term in terms]

    # Convert sparse matrix to dense because CSV cannot store sparse data.
    X_dense = X_all.toarray()

    # Create DataFrame with BOW columns.
    bow_df = pd.DataFrame(X_dense, columns=bow_cols, index=df.index)

    # Append the BOW columns to the original DataFrame.
    out_df = pd.concat([df, bow_df], axis=1)

    # Save the result.
    out_df.to_csv(OUT_FILE, index=False)

    print("")
    print(f"Wrote: {OUT_FILE}")
    print(f"Rows total: {len(df):,}")
    print(f"Training rows used for fitting: {int(is_train.sum()):,}")
    print(f"Validation rows transformed only: {int(is_val.sum()):,}")
    print(f"BOW features: {len(bow_cols):,}")


if __name__ == "__main__":
    main()