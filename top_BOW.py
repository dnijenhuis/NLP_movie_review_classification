# This module extracts the top BOW n-grams for POS and NEG reviews.
# It creates two CSV files:
# 1. Top 50 per class by raw total count.
# 2. Top 50 per class by class-distinctive mean count difference.

from pathlib import Path  # Needed for path handling.

import pandas as pd  # Needed for reading and writing CSV files.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
IN_FILE = DATA_DIR / "imdb_train_preprocessed_BOW.csv"
OUT_FILE_TOTAL = DATA_DIR / "top_bow_ngrams_per_class.csv"
OUT_FILE_DISTINCTIVE = DATA_DIR / "top_bow_ngrams_per_class_distinctive.csv"

# Column names.
LABEL_COL = "POS"

# Prefix for the generated BOW columns.
BOW_PREFIX = "bow_"

# Output configuration.
TOP_K = 50


def main() -> None:
    # Load the BOW training CSV.
    df = pd.read_csv(IN_FILE)

    # Extract all BOW feature columns.
    bow_cols = [col for col in df.columns if col.startswith(BOW_PREFIX)]

    if not bow_cols:
        raise ValueError("No BOW columns found.")

    # Split the data into POS and NEG subsets.
    pos_df = df[df[LABEL_COL] == 1]
    neg_df = df[df[LABEL_COL] == 0]

    # Calculate total BOW counts per class.
    pos_total = pos_df[bow_cols].sum(axis=0)
    neg_total = neg_df[bow_cols].sum(axis=0)

    # Get the top features by raw total count.
    pos_total_top = pos_total.sort_values(ascending=False).head(TOP_K)
    neg_total_top = neg_total.sort_values(ascending=False).head(TOP_K)

    # Store the top POS n-grams by raw total count.
    pos_total_out = pd.DataFrame({
        "class": "POS",
        "rank": range(1, len(pos_total_top) + 1),
        "ngram_column": pos_total_top.index,
        "ngram": [col[len(BOW_PREFIX):] for col in pos_total_top.index],
        "bow_total_count": pos_total_top.values,
    })

    # Store the top NEG n-grams by raw total count.
    neg_total_out = pd.DataFrame({
        "class": "NEG",
        "rank": range(1, len(neg_total_top) + 1),
        "ngram_column": neg_total_top.index,
        "ngram": [col[len(BOW_PREFIX):] for col in neg_total_top.index],
        "bow_total_count": neg_total_top.values,
    })

    # Combine and save the raw total count output.
    total_out_df = pd.concat([pos_total_out, neg_total_out], ignore_index=True)
    total_out_df.to_csv(OUT_FILE_TOTAL, index=False)

    # Calculate mean BOW counts per class.
    pos_mean = pos_df[bow_cols].mean(axis=0)
    neg_mean = neg_df[bow_cols].mean(axis=0)

    # Calculate class-distinctive scores.
    pos_distinctive = pos_mean - neg_mean
    neg_distinctive = neg_mean - pos_mean

    # Get the top distinctive features per class.
    pos_distinctive_top = pos_distinctive.sort_values(ascending=False).head(TOP_K)
    neg_distinctive_top = neg_distinctive.sort_values(ascending=False).head(TOP_K)

    # Store the top POS n-grams by class distinctiveness.
    pos_distinctive_out = pd.DataFrame({
        "class": "POS",
        "rank": range(1, len(pos_distinctive_top) + 1),
        "ngram_column": pos_distinctive_top.index,
        "ngram": [col[len(BOW_PREFIX):] for col in pos_distinctive_top.index],
        "mean_bow_pos": pos_mean[pos_distinctive_top.index].values,
        "mean_bow_neg": neg_mean[pos_distinctive_top.index].values,
        "distinctiveness_score": pos_distinctive_top.values,
    })

    # Store the top NEG n-grams by class distinctiveness.
    neg_distinctive_out = pd.DataFrame({
        "class": "NEG",
        "rank": range(1, len(neg_distinctive_top) + 1),
        "ngram_column": neg_distinctive_top.index,
        "ngram": [col[len(BOW_PREFIX):] for col in neg_distinctive_top.index],
        "mean_bow_neg": neg_mean[neg_distinctive_top.index].values,
        "mean_bow_pos": pos_mean[neg_distinctive_top.index].values,
        "distinctiveness_score": neg_distinctive_top.values,
    })

    # Combine and save the class-distinctive output.
    distinctive_out_df = pd.concat([pos_distinctive_out, neg_distinctive_out], ignore_index=True)
    distinctive_out_df.to_csv(OUT_FILE_DISTINCTIVE, index=False)

    print("")
    print(f"Stored top BOW n-grams by raw total count to: {OUT_FILE_TOTAL}")
    print(f"Stored top BOW n-grams by class distinctiveness to: {OUT_FILE_DISTINCTIVE}")


if __name__ == "__main__":
    main()