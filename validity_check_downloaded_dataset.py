# This module checks whether the dataset has been downloaded completely and correctly.
# It does this by checking the hash of the dataset against the (known) MD5.
# Furthermore, this module performs several logic tests like the expected downloaded directory,
# whether the expeded number of (train and test) datapoints are present, whether the datapoint IDs
# are logical and complete, etc.

import hashlib # Needed for checking the download against the MD5.
from pathlib import Path # Needed for handling paths.

# The MD5 is retrieved from https://docs.pytorch.org/text/0.17.0/_modules/torchtext/datasets/imdb.html
# and hardcoded. It should not be changed.
expected_md5 = "7c2ac02c03563afcf9b574c7e56c153a"

archive_path = Path("PATH_TO_IMDB_ARCHIVE")
dataset_root = Path("PATH_TO_EXTRACTED_DATASET_ROOT")

# Both the training and test set should consist of 12500 good reviews and 12500 bad reviews.
# So 12500 * 2 * 2 = 50000 in total. Each of the 4 subsets should have a unique ID. Also,
# the scores (although not used further in this project) should be in the range 1 to 10.
EXPECTED_COUNT = 12500
MIN_ID, MAX_ID = 0, 12499
MIN_SCORE, MAX_SCORE = 1, 10

def verify_md5() -> None:
    # This function checks the hash of the downloaded file against the known MD5.
    local_md5 = hashlib.md5(archive_path.read_bytes()).hexdigest()
    print("MD5/hash check:", local_md5.lower() == expected_md5.lower())
    if local_md5.lower() != expected_md5.lower():
        raise AssertionError("MD5 mismatch.")


def validate_structure() -> None:
    # This function performs several checks to determine whether the downloaded and extracted
    # dataset has the expected structure (expected based on the included README of the dataset).
    # It does this for each of the 4 subsets (train/pos, train/neg, test/pos, train/neg).
    checks = [
        ("train_pos", dataset_root / "train" / "pos"), # Label and path of each of the 4 subsets.
        ("train_neg", dataset_root / "train" / "neg"),
        ("test_pos", dataset_root / "test" / "pos"),
        ("test_neg", dataset_root / "test" / "neg"),
    ]

    for name, folder in checks:
        # For each folder, check if it exists.
        if not folder.exists():
            raise AssertionError(f"[{name}] Folder not found: {folder}")

        paths = list(folder.iterdir()) # Lists all content.

        # Check if all files have .txt format as expected.
        non_txt = [p for p in paths if p.suffix.lower() != ".txt"]
        if non_txt:
            raise AssertionError(f"[{name}] Found non-.txt files: {non_txt[:5]}")

        # Check if the directory has the expected 12500 files.
        if len(paths) != EXPECTED_COUNT:
            raise AssertionError(f"[{name}] Expected {EXPECTED_COUNT} files, found {len(paths)}")

        seen_ids = set() # Store the seen IDs to a set.

        for p in paths:
            # For every .txt file/review, split the review ID and the given score.
            # For this the "_" is used.
            stem = p.stem
            parts = stem.split("_")

            id_str, score_str = parts # Assign the split ID and score to variables.

            # Check whether both the ID and the scores are numerical as expected.
            if not id_str.isdigit():
                raise AssertionError(f"[{name}] Non-numeric id: {p.name}")
            if not score_str.isdigit():
                raise AssertionError(f"[{name}] Non-numeric score: {p.name}")

            doc_id = int(id_str) # Turn ID to integer so Python can perform calculations with it.
            score = int(score_str) # Turn score to integer so Python can perform calculations with it.

            # IDs should be in the range from 0 to 12499.
            if doc_id < MIN_ID or doc_id > MAX_ID:
                raise AssertionError(f"[{name}] ID out of range [{MIN_ID}..{MAX_ID}]: {p.name}")

            # Scores should be in the range from 1 to 10.
            if score < MIN_SCORE or score > MAX_SCORE:
                raise AssertionError(f"[{name}] Score out of range [{MIN_SCORE}..{MAX_SCORE}]: {p.name}")

            # Check for 'duplicate unique' IDs.
            if doc_id in seen_ids:
                raise AssertionError(f"[{name}] Duplicate ID {doc_id} found (e.g., {p.name})")

            seen_ids.add(doc_id) # After processing datapoint, add it to the set.

        print(f"The subset {name} has: {len(paths)} files, valid and unique IDs, and valid scores.")

    print("All structure checks passed.")


def main() -> None:
    verify_md5()
    validate_structure()


if __name__ == "__main__":
    main()