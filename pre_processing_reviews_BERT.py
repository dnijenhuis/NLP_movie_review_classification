# This module preprocesses the review text for the BERT model. BERT requires only minimal preprocessing.
# Therefore this module performs:
# - Removal of HTML artifacts;
# - Unicode normalization;
# - Whitespace normalization.
# The cleaned text is stored in a new column 'Review_preprocessed_BERT'.
# New CSV-files are created so separate input files are used for BERT.
#
# The test data is also preprocessed. An alternative approach would be to leave the test data
# unprocessed to better simulate raw real-world input data. However, because this preprocessing
# is computationally very light, the same preprocessing steps can easily be included in the production pipeline.

import html  # Needed for decoding HTML entities.
import re  # Needed for regex cleaning.
import unicodedata  # Needed for unicode normalization.
from pathlib import Path  # Needed for path handling.

import pandas as pd  # Needed for CSV handling.


# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")

TRAIN_IN = "imdb_train.csv"
TEST_IN = "imdb_test.csv"

TRAIN_OUT = "imdb_train_preprocessed_for_BERT.csv"
TEST_OUT = "imdb_test_preprocessed_for_BERT.csv"

# Column names.
TEXT_COL = "Review" # Raw text from .txt converted to CSV.
OUT_COL = "Review_preprocessed_BERT" # Ouput column, used by the BERT-model later on.

# Regex patterns used for cleaning.
_RE_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_RE_TAGS = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    # Remove HTML artifacts and normalize text.
    text = _RE_BR.sub(" ", text)
    text = _RE_TAGS.sub(" ", text)
    text = html.unescape(text)

    text = unicodedata.normalize("NFKC", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


def process_file(input_path: Path, output_path: Path) -> None:
    # Load the CSV and preprocess the review text.
    df = pd.read_csv(input_path)

    if TEXT_COL not in df.columns:
        raise KeyError(f"Expected column '{TEXT_COL}' not found in {input_path.name}")

    texts_raw = df[TEXT_COL].fillna("").astype(str).tolist()

    # Perform the BERT preprocessing.
    processed = [clean_text(t) for t in texts_raw]

    # Store the cleaned text in a new column.
    df[OUT_COL] = pd.Series(processed, index=df.index)

    # Write the new CSV.
    df.to_csv(output_path, index=False)

    empty_count = int((df[OUT_COL].str.len() == 0).sum()) # To make sure all rows have a preprocessed text.

    print("")
    print(f"Wrote: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Empty BERT-preprocessed reviews: {empty_count}")


def main() -> None:
    # Define input and output files and process the train and test datasets.
    train_in = DATA_DIR / TRAIN_IN
    test_in = DATA_DIR / TEST_IN

    train_out = DATA_DIR / TRAIN_OUT
    test_out = DATA_DIR / TEST_OUT

    process_file(train_in, train_out)
    process_file(test_in, test_out)


if __name__ == "__main__":
    main()