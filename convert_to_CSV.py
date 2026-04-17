# This module converts the downloaded and validated IMDB dataset into two CSV-files.
# First a training CSV and then a test CSV. These respectively contain the labeled training and
# test data. Specifically, it takes the following data per review file and stores in the CSVs:
# - Binary value indicating that a review is postive with the value '1' (0 if negative);
# - File unique ID (this is a created unique ID out of POS or NEG, and the existing non-unique ID. The reason
#   for this is that the data does not have true uniqe IDs yet);
# - IMDB score (this is not used for this project but could be used for different projects);
# - Review text.

import csv # Needed for handling CSVs.
from pathlib import Path # Needed for path handling.


def main() -> None:
    # This function turns the train and test data into two CSVs.
    dataset_root = Path("PATH_TO_EXTRACTED_DATASET_ROOT") # Input data.
    out_csv_train = Path("PATH_TO_IMDB_TRAIN_CSV") # Output training data.
    out_csv_test = Path("PATH_TO_IMDB_TEST_CSV") # Output test data.

    # Beneath are the various (sub)directories.
    train_pos_dir = dataset_root / "train" / "pos"
    train_neg_dir = dataset_root / "train" / "neg"
    test_pos_dir = dataset_root / "test" / "pos"
    test_neg_dir = dataset_root / "test" / "neg"

    # Create training data CSV.
    train_rows = [] # List for training rows.

    # For every file in the training data, split the file name and store the type of review (POS or NEG), the
    # unique ID, the score and the review text, and add this to the train_rows list. This list is then used as input
    # for creating the CSV.
    for folder, pos_value, prefix in [(train_pos_dir, 1, "POS"), (train_neg_dir, 0, "NEG")]:
        for file_path in folder.glob("*.txt"):
            stem = file_path.stem # Take filename without '.txt' and store to 'stem'.
            parts = stem.split("_") # Split on '_'.
            unique_number = parts[0] # First part is the unique ID (not really unique yet).
            score = int(parts[1]) # Second element is the score.
            unique_id = f"{prefix}{unique_number}" # Unique ID consisting of POS or NEG, and the existing ID.
            review = file_path.read_text(encoding="utf-8", errors="replace") # Take review text from the .txt.
            train_rows.append([pos_value, unique_id, score, review]) # Append to list.

    # Sorting is done on Unique ID alphabetically. The number-part in the unique ID is converted to an integer
    # to make sure that sorting goes correctly (e.g. POS2 should come before POS10, and not the other way around).
    train_rows.sort(key=lambda r: (r[1][:3], int(r[1][3:]))) # Sort data from NEG0 to POS12499.

    # After storing all the rows in train_rows, the CSV is created and the data is stored in the CSV.
    # The columns are POS (0 or 1), Unique_ID (truly unique ID within the respective CSV),
    # Score (not used but kept anyway) and Review (the review text).
    with out_csv_train.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["POS", "Unique_ID", "Score", "Review"])
        writer.writerows(train_rows)

    print("Wrote training CSV.")

    # Create test data CSV. The logic for the rest of this code is the same as for the training CSV above.
    test_rows = []

    for folder, pos_value, prefix in [(test_pos_dir, 1, "POS"), (test_neg_dir, 0, "NEG")]: # Testdirs instead of train.
        for file_path in folder.glob("*.txt"):
            stem = file_path.stem
            parts = stem.split("_")
            unique_number = parts[0]
            score = int(parts[1])
            unique_id = f"{prefix}{unique_number}"
            review = file_path.read_text(encoding="utf-8", errors="replace")
            test_rows.append([pos_value, unique_id, score, review])

    test_rows.sort(key=lambda r: (r[1][:3], int(r[1][3:])))

    with out_csv_test.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["POS", "Unique_ID", "Score", "Review"])
        writer.writerows(test_rows)

    print("Wrote test CSV.")


if __name__ == "__main__":
    main()