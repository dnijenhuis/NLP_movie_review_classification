# This module validates the created IMDB training and test CSV files.
# It checks whether the files have the expected structure and values.
# Specifically, it verifies:
# - The file has the .csv extension and exists;
# - The header matches one of the allowed column name sets;
# - The total number of rows is correct;
# - The number of positive and negative reviews is correct;
# - All rows contain review text;
# - All Unique_ID values are, in fact, unique;
# - The Score column contains integers in the expected range.
#
# If any of these checks fail, an error is raised. If all checks pass,
# a summary of the validation results is printed.

import csv  # Needed for handling CSVs.
from pathlib import Path  # Needed for path handling.

# Below are the expected CSV file paths.
train_csv = Path("PATH_TO_IMDB_TRAIN_CSV")
test_csv = Path("PATH_TO_IMDB_TEST_CSV")

EXPECTED_ROWS = 25000  # Both the training CSV and test CSV should have 25000 rows.
EXPECTED_POS = 12500  # Both the training CSV and test CSV should have 12500 positive reviews.
EXPECTED_NEG = 12500  # Both the training CSV and test CSV should have 12500 negative reviews.
MIN_SCORE = 1  # Though score is not used in this project, the extracted scores are still expected to be valid/logical.
MAX_SCORE = 10

# Expected CSV column names/header.
REQUIRED_HEADERS = ["POS", "Unique_ID", "Score", "Review"]
OPTIONAL_EXTRA_HEADER = "validation_set" # Allows to also run this module (again) without errors after validation split.
ALLOWED_HEADERS = [
    REQUIRED_HEADERS,
    REQUIRED_HEADERS + [OPTIONAL_EXTRA_HEADER],
]


def validate_csv(path: Path) -> None:
    # This function performs several validity tests on the created CSVs.
    if path.suffix.lower() != ".csv":  # Check if files have in fact a CSV-extension.
        raise AssertionError(f"{path.name}: not a .csv file")

    if not path.exists():  # Check if the path/file exists.
        raise FileNotFoundError(f"File not found: {path}")

    # Below are counters and a set used for performing various checks.
    row_count = 0
    pos_count = 0
    neg_count = 0
    unique_ids = set()

    with path.open("r", newline="", encoding="utf-8") as f:  # Open CSV.
        reader = csv.reader(f)
        try:
            header = next(reader)  # Check if the CSV has a header row.
        except StopIteration:
            raise AssertionError(f"{path.name}: empty file (no header)")

        if header not in ALLOWED_HEADERS:
            raise AssertionError(
                f"{path.name}: header mismatch.\n"
                f"Expected one of: {ALLOWED_HEADERS}\n"
                f"Found          : {header}"
            )

        expected_column_count = len(header)

        for i, row in enumerate(reader, start=2):
            # For every row, increase the row_count variable by 1. Check if the row has the expected number
            # of columns.
            row_count += 1

            if len(row) != expected_column_count:
                raise AssertionError(
                    f"{path.name}: row {i} has {len(row)} columns, expected {expected_column_count}"
                )

            pos_str = row[0]
            uid = row[1]
            score_str = row[2]
            review = row[3]

            if uid in unique_ids:  # Check for duplicate 'unique' IDs.
                raise AssertionError(f"{path.name}: duplicate Unique_ID '{uid}' at row {i}")
            unique_ids.add(uid)

            # Below it is checked whether the row has an expected value in each column.
            # It is expected that every cell in column 'Review' has at least a value.
            if review is None or review.strip() == "":
                raise AssertionError(f"{path.name}: empty Review at row {i}")

            # Value should be either 0 or 1.
            if pos_str not in ("0", "1"):
                raise AssertionError(f"{path.name}: POS not 0/1 at row {i}: {pos_str}")

            # Below the number of positive and negative reviews are counted so the total can be checked later.
            if pos_str == "1":
                pos_count += 1
            else:
                neg_count += 1

            try:
                score = int(score_str)
            except ValueError:  # The score is expected to be an integer and should be in the range 1-10.
                raise AssertionError(f"{path.name}: Score not int at row {i}: {score_str}")

            if score < MIN_SCORE or score > MAX_SCORE:
                raise AssertionError(
                    f"{path.name}: Score out of range [{MIN_SCORE}..{MAX_SCORE}] at row {i}: {score}"
                )

    # The total number of rows should be equal to the expected number.
    if row_count != EXPECTED_ROWS:
        raise AssertionError(
            f"{path.name}: expected {EXPECTED_ROWS} rows (no header), found {row_count}"
        )

    # The total number of positive and negative rows should be equal to the expected number.
    if pos_count != EXPECTED_POS or neg_count != EXPECTED_NEG:
        raise AssertionError(
            f"{path.name}: expected {EXPECTED_POS} POS and {EXPECTED_NEG} NEG, "
            f"found {pos_count} POS and {neg_count} NEG"
        )

    print(
        f"{path.name}: "
        f"Header valid = True, "
        f"Total number of rows = {row_count == EXPECTED_ROWS} ({row_count}),\n"
        f"Number of POS and NEG reviews = (POS={pos_count}, NEG={neg_count}), "
        f"Score range = [{MIN_SCORE}-{MAX_SCORE}], "
        f"All rows contain review text = True, "
        f"Unique IDs = {len(unique_ids)}"
    )


def main() -> None:
    # Perform validate_csv function on training CSV and on test CSV.
    validate_csv(train_csv)
    print("")
    validate_csv(test_csv)
    print("")
    print("All CSV checks passed.")


if __name__ == "__main__":
    main()