# This module preprocesses the review text in the IMDB CSV-files. This is done for the MNB and SVM models. For BERT a
# different preprocessing is performed. This module's preprocessing performs several steps:
# - Remove HTML artifacts;
# - Normalize unicode and whitespace;
# - Lowercase the text;
# - Tokenize using spaCy;
# - Lemmatize tokens;
# - Remove stopwords (while keeping negations).
# The processed review text is stored in a new column 'Review_preprocessed'.
# New CSV-files are created so different input files are used for MNB and SVM on one hand, and for BERT on the other.

import html  # Needed for decoding HTML entities.
import re  # Needed for regex tasks.
import unicodedata  # Needed for unicode normalization.
from pathlib import Path  # Needed for path handling.
import pandas as pd  # Needed for CSV handling.
import spacy  # Needed for tokenization and lemmatization.
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # Needed for stopword list.


# Configuration options. Can be used for increasing speed (especially by turning off lemmatization).
USE_LOWERCASE = True
USE_STOPWORD_REMOVAL = True
USE_LEMMATIZATION = True

# Paths and filenames.
DATA_DIR = Path("PATH_TO_DATA_DIR")
TRAIN_FILE = "imdb_train.csv" # The checked and validated training data (including training/val split).
TEST_FILE = "imdb_test.csv" # The checked and validated test data.
OUT_SUFFIX = "_preprocessed_MNB_and_SVM" # To indicate that the data is preprocessed (for MNB and SVM, not for BERT).

# Column names.
TEXT_COL = "Review" # Review text in CSV.
OUT_COL = "Review_preprocessed" # Preprocessed review text in CSV.

# Stopword list.
# Negations are removed from the stopword (so are kept in data) list because they are informational.
_NEGATIONS = {"no", "nor", "not", "never"}
STOPWORDS = set(ENGLISH_STOP_WORDS) - _NEGATIONS

# Regex patterns used for cleaning the text.
_RE_BR = re.compile(r"<br\s*/?>", flags=re.IGNORECASE)
_RE_TAGS = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")


def basic_clean(text: str) -> str:
    # Perform basic cleaning before tokenization.
    text = _RE_BR.sub(" ", text)  # Replace <br> tags.
    text = _RE_TAGS.sub(" ", text)  # Remove remaining HTML tags.
    text = html.unescape(text)  # Decode HTML entities.
    text = unicodedata.normalize("NFKC", text)  # Normalize unicode.
    text = _RE_WHITESPACE.sub(" ", text).strip()  # Normalize whitespace.

    if USE_LOWERCASE:
        text = text.lower()

    return text


def process_texts(texts: list[str], nlp) -> list[str]:
    # Tokenize and preprocess the cleaned texts.
    docs = nlp.pipe(texts, batch_size=64) # Tokenization.

    processed = [] # List containing the preprocessed texts.

    for doc in docs:
        # For every token in every review text, perform lemmatization and stopword filtering.
        tokens_out = []

        for tok in doc:
            # Skip punctuation and whitespace tokens.
            if tok.is_space or tok.is_punct:
                continue

            if USE_LEMMATIZATION:
                token = (tok.lemma_ or "").lower()
            else:
                token = tok.text.lower() if USE_LOWERCASE else tok.text

            if not token:
                continue

            # Remove stopwords except negations.
            if USE_STOPWORD_REMOVAL:
                if token in _NEGATIONS:
                    tokens_out.append(token)
                    continue

                if token in STOPWORDS:
                    continue

            tokens_out.append(token)

        processed.append(" ".join(tokens_out))

    return processed


def process_file(input_path: Path, output_path: Path, nlp) -> None:
    # This function loads the CSV, preprocesses the review text, and writes a new CSV.
    df = pd.read_csv(input_path)

    if TEXT_COL not in df.columns:
        raise KeyError(f"Expected column '{TEXT_COL}' not found in {input_path.name}")

    # Convert the review column to a list of strings.
    texts_raw = df[TEXT_COL].fillna("").astype(str).tolist()

    # Call function to perform basic cleaning.
    texts_clean = [basic_clean(t) for t in texts_raw]

    # Call function for token processing (lemmatization and stopword removal).
    processed = process_texts(texts_clean, nlp)

    # Store the result in a new column.
    df[OUT_COL] = pd.Series(processed, index=df.index)

    # Write the new CSV.
    df.to_csv(output_path, index=False)

    print("")
    print(f"Wrote: {output_path}")
    print(f"Rows: {len(df)}")
    print(f"Empty preprocessed reviews: {(df[OUT_COL].str.len() == 0).sum()}")


def main() -> None:
    # Initialize the spaCy pipeline.
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    train_in = DATA_DIR / TRAIN_FILE
    test_in = DATA_DIR / TEST_FILE

    train_out = DATA_DIR / f"{Path(TRAIN_FILE).stem}{OUT_SUFFIX}.csv"
    test_out = DATA_DIR / f"{Path(TEST_FILE).stem}{OUT_SUFFIX}.csv"

    # Process the training and test CSV-files.
    process_file(train_in, train_out, nlp)
    process_file(test_in, test_out, nlp)


if __name__ == "__main__":
    main()