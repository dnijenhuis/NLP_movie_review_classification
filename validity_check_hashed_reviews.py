# This module performs hash-based integrity checks on the created IMDB CSV-files.
# Also new CSVs are created (for both the train and test CSV). These are copies of the
# existing CSVs, but with three additional columns:
# - hash_csv_review;
# - hash_original_file;
# - hash_match.
# In the new CSVs, for every row, the review text in the CSV is hashed and stored in
# 'hash_csv_review'. Also, the corresponding original .txt file is looked up, hashed, and stored in
# 'hash_original_file'. After this, the hash comparison is stored in column 'hash_match'.
# Technically the creation of the new CSVs is not needed for performing the hash checks. However,
# it is my personal preference to be able to inspect the data visually. Additionally, storing the
# results of the hash checks improves the audit trail.

import csv  # Needed for handling CSVs.
from pathlib import Path  # Needed for path handling.
import hashlib  # Needed for using hashes.


def process_csv(split: str, input_csv: Path, output_csv: Path, dataset_root: Path) -> None:
    # This function performs the hash checks for the training CSV and the test CSV.
    rows = [] # Dictionary containing the rows.

    # First copy every row from the old CSV and add 3 new empty columns needed for the hash check.
    with input_csv.open("r", newline="", encoding="utf-8") as fin:
        reader = csv.DictReader(fin)

        for row in reader:
            row["hash_csv_review"] = ""
            row["hash_original_file"] = ""
            row["hash_match"] = ""
            rows.append(row)

    # For every row, create the hash of the review text in the CSV and the hash of the
    # corresponding original .txt file. Store both hashes in the row.
    for row in rows:
        pos_value = row["POS"].strip()
        label_folder = "pos" if pos_value == "1" else "neg"

        uid = row["Unique_ID"].strip()
        doc_id = int(uid[3:])

        score = int(row["Score"].strip())
        review = row["Review"]

        row["hash_csv_review"] = hashlib.sha256(review.encode("utf-8")).hexdigest()

        # Based on the variables above, the original file location is reconstructed.
        # In retrospect I could have maybe better created the hash directly when converting the
        # data to a CSV in the module 'convert_to_CSV'.
        original_path = dataset_root / split / label_folder / f"{doc_id}_{score}.txt"
        original_text = original_path.read_text(encoding="utf-8", errors="replace")

        row["hash_original_file"] = hashlib.sha256(original_text.encode("utf-8")).hexdigest()

    # In a separate loop, compare both hash columns and store the result in 'hash_match'.
    for row in rows:
        is_match = row["hash_csv_review"] == row["hash_original_file"] # Here the actual check is done.
        row["hash_match"] = str(is_match).upper()

    # Write the new CSV.
    with output_csv.open("w", newline="", encoding="utf-8") as fout:
        fieldnames = [
            "POS",
            "Unique_ID",
            "Score",
            "Review",
            "hash_csv_review",
            "hash_original_file",
            "hash_match",
        ]

        writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)

    match_count = sum(1 for row in rows if row["hash_match"] == "TRUE")
    mismatch_count = sum(1 for row in rows if row["hash_match"] == "FALSE")

    print(f"Wrote {output_csv.name}")
    print(f"hash_match TRUE: {match_count} | hash_match FALSE: {mismatch_count}")


def main() -> None:
    # This function defines the input and output CSVs and calls the processing function.
    base_dir = Path("PATH_TO_DATA_DIR")
    dataset_root = base_dir / "Extracted" / "aclImdb"

    train_input_csv = base_dir / "imdb_train.csv"
    test_input_csv = base_dir / "imdb_test.csv"

    train_output_csv = base_dir / "imdb_train_hash_check.csv"
    test_output_csv = base_dir / "imdb_test_hash_check.csv"

    # Call the process_CSV function and process the train and test data.
    process_csv("train", train_input_csv, train_output_csv, dataset_root)
    process_csv("test", test_input_csv, test_output_csv, dataset_root)


if __name__ == "__main__":
    main()