# This module splits the training dataset into an actual training set and a validation
# set. It does this by randomly assigning 20% of the POS rows and 20% of the NEG rows in the
# training set the label 'validation_set'. Naturally, this is only done for the training set and
# not for the test set.

import pandas as pd
import numpy as np
from pathlib import Path # Needed for handling paths.

SEED = 20260225 # A fixed seed is used for auditing/logging/reproducibility.
VALIDATION_FRACTION = 0.20 # The desired fraction 80/20 split is common practice.
TOTAL_ROWS = 25000 # The number of rows in the training set. This is a given number.

# Directory of training CSV.
DATA_PATH = Path("PATH_TO_IMDB_TRAIN_CSV")

LABEL_COL = "POS" # Needed for making sure that 20% of both the POS and NEG reviews are labeled as validation.
SPLIT_COL = "validation_set" # New column indicating whether a row belongs to validation set (1) or not (0).

# This variable is used later for checking whether the function was executed correctly.
EXPECTED_VAL_ROWS = int(VALIDATION_FRACTION * TOTAL_ROWS)


def main() -> None:
    # This function splits the training dataset into a training dataset and a validation set.
    # It does this randomly and takes equally a fraction for the POS and NEG reviews.
    df = pd.read_csv(DATA_PATH)

    # Create validation column.
    df[SPLIT_COL] = 0

    rng = np.random.default_rng(SEED)

    # Split the rows in 2 segments: POS and NEG. Select 20% of each class.
    idx_neg = df.index[df[LABEL_COL] == 0] # All NEG reviews.
    idx_pos = df.index[df[LABEL_COL] == 1] # All POS reviews.

    # Select 20% from the 2 segments.
    val_neg = rng.choice(idx_neg, size=int(len(idx_neg) * VALIDATION_FRACTION), replace=False)
    val_pos = rng.choice(idx_pos, size=int(len(idx_pos) * VALIDATION_FRACTION), replace=False)

    # Assigning value '1' to the selected 20%.
    df.loc[val_neg, SPLIT_COL] = 1
    df.loc[val_pos, SPLIT_COL] = 1

    # Counts the number of validation rows.
    val_count = int((df[SPLIT_COL] == 1).sum())

    # Write the new data to CSV.
    df.to_csv(DATA_PATH, index=False)

    print(f"Validation rows: {val_count}")
    print(f"Number of validation rows correct: {val_count == EXPECTED_VAL_ROWS}")
    print("Validation split created.")


if __name__ == "__main__":
    main()