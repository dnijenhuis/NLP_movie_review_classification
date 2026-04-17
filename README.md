# CLASSIFYING MOVIE REVIEWS THROUGH NLP-TECHNIQUES


This repository contains a full NLP-pipeline for binary sentiment classification on the Maas et al. (2011) Large Movie Review Dataset (IMDB). 
The project compares three different approaches for classifying movie reviews as positive or negative:

- Multinomial Naive Bayes (MNB) with Bag-of-Words (BOW)
- Support Vector Machine (SVM) with TF-IDF
- RoBERTa transformer-based model

## Project goal

The goal of this project is to build an NLP-solution that:

- downloads, extracts and validates the IMDB dataset
- converts the raw `.txt` reviews into a CSV format
- pre-processes the text
- performs feature engineering
- trains multiple NLP-models
- evaluates the models on a test set

## Dataset

The project uses the **Maas et al. (2011) Large Movie Review Dataset**.

Dataset characteristics:

- 50,000 movie reviews in total
- 25,000 training reviews
- 25,000 test reviews
- training and test sets each contain:
  - 12,500 positive reviews
  - 12,500 negative reviews

Within the training set, this NLP-solution creates a validation split:

- 20% validation
- 80% training
- the split is class-balanced for positive and negative reviews

## Pipeline overview

The pipeline is built in separate Python modules and can be run step by step or from the main pipeline script.

### 1. Download and extraction
- `download_dataset.py`
- Downloads the IMDB-dataset and extracts it.

### 2. Dataset validation
- `validity_check_downloaded_dataset.py`
- Verifies the MD5 hash and checks the extracted folder structure, file counts, ID ranges, and score ranges.

### 3. Conversion to CSV
- `convert_to_CSV.py`
- Converts the raw review files into structured training and test CSV files.

### 4. CSV validation and integrity checks
- `validity_and_completeness_check_CSV.py`
- Validates row counts, labels, uniqueness of IDs, headers, and score ranges.
- `validity_check_hashed_reviews.py`
- Adds hash-based integrity checks to compare CSV review text with the original source files.

### 5. Train/validation split
- `split_training_set_validation_set.py`
- Creates a reproducible 80/20 validation split inside the training CSV.

### 6. Preprocessing
Two separate preprocessing routes are used, one for MNB/SVM and one for RoBERTa.

#### For MNB and SVM
- `pre_processing_reviews.py`
- Removes HTML artifacts
- normalizes Unicode and whitespace
- lowercases text
- tokenizes with spaCy
- lemmatizes tokens
- removes stopwords
- keeps negations such as `no`, `nor`, `not`, and `never`

#### For RoBERTa
- `pre_processing_reviews_BERT.py`
- Uses lighter preprocessing
- removes HTML artifacts
- normalizes Unicode and whitespace

Note: some script and output names contain the term `BERT`, but the final transformer model used in this project is **RoBERTa**. The reason for this is that this specific member of the BERT-family was chosen quite late during project development.

### 7. Feature engineering

#### Bag-of-Words
- `perform_BOW.py`
- Fits BOW on the training portion only to avoid validation leakage
- Creates BOW features for the MNB model

Configuration used:
- `max_features = 7500`
- `ngram_range = (1, 3)`
- `min_df = 2`
- `max_df = 0.95`

#### TF-IDF
- `perform_TF_IDF.py`
- Fits TF-IDF on the training portion only to avoid validation leakage
- stores the vectorizer and matrices in a `.joblib` bundle

Configuration used:
- `max_features = 5000`
- `ngram_range = (1, 3)`
- `min_df = 3`
- `max_df = 0.90`
- `sublinear_tf = True`

### 8. Model training

#### MNB
- `train_MNB.py`
- Trains a Multinomial Naive Bayes model on BOW features

Configuration:
- `alpha = 1.0`

#### SVM
- `train_SVM.py`
- Trains a LinearSVC classifier on TF-IDF features

Configuration:
- `C = 0.1`
- `max_iter = 1000`

#### RoBERTa
- `train_BERT.py`
- Trains a **RoBERTa-base** sequence classification model

Configuration:
- model: `roberta-base`
- `max_length = 500`
- `batch_size_train = 8`
- `batch_size_eval = 16`
- `epochs = 1`
- `learning_rate = 0.00001`
- `weight_decay = 0.01`

### 9. Analysis of important n-grams
- `top_BOW.py`
- `top_TFIDF.py`

These scripts extract the most important BOW and TF-IDF n-grams per class.

### 10. Final test evaluation
- `model_tests.py`
- Loads the trained models and evaluates them on the held-out test set
- Stores the results in `test_metrics.csv`

### 11. Pipeline runner
- `main_NLP_pipeline.py`
- Central script to run the selected modules in sequence

## Final test results

The final test metrics are included in the repository as a CSV-file. The three model types show a clear performance ranking:
1. RoBERTa achieved the best overall test performance.
2. SVM + TF-IDF performed strongly and clearly outperformed MNB.
3. MNB + BOW was the weakest of the three, but still produced solid baseline results.

RoBERTa produced the highest scores, but it also required much higher computational costs than the classical models. The CSV with test metrics is included in the GitHub upload: `test_metrics.csv`.

## Requirements
Main libraries used in this project:
- pandas
- numpy
- scikit-learn
- joblib
- spaCy
- torch
- transformers

## How to use
1. Before running the project, update the file paths in the Python scripts.
2. Install the required Python libraries in your environment.
3. You can run the full pipeline in the main .py, or run the scripts separately.

Enable or disable specific steps inside `main_NLP_pipeline.py` by commenting or uncommenting the relevant modules.

## Future improvements
- make all paths configurable through a single settings file
- add automatic reporting of preprocessing and model settings to the validation and test CSVs.
- add a web scraping tool/module which retrieves movie reviews from (e.g.) IMDB and turns them into .txt files. The solution in its current form requires the reviews to be in a .txt format.

## Reference dataset
Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning word vectors for sentiment analysis*. In Proceedings of the 49th Annual Meeting of the Association for Computa-tional Linguistics: Human Language Technologies (pp. 142–150). Association for Computational Linguistics.
