# This module executes the full NLP project pipeline.
# It runs the selected modules in sequence, from data preparation to model training and testing.

import download_dataset
import validity_check_downloaded_dataset
import convert_to_CSV
import validity_and_completeness_check_CSV
import validity_check_hashed_reviews
import split_training_set_validation_set
import pre_processing_reviews
import pre_processing_reviews_BERT
import perform_BOW
import perform_TF_IDF
import train_MNB
import train_SVM
import train_BERT
import top_BOW
import top_TFIDF
import model_tests
import torch

# This prints whether CUDA is available for ROBERTA.
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No CUDA device")

def run_step(name: str, func) -> None:
    # This function prints a clear banner before executing a module.
    banner = f" STARTING MODULE: {name} "
    print("\n" * 2)
    print("=" * 80)
    print(banner.center(80, "="))
    print("=" * 80)
    func()

def main() -> None:
    # This function executes the selected modules in sequence.
    # Use '#' to disable modules that do not need to be executed.

    # Data dowload, preparation and checks.
    run_step("download_dataset", download_dataset.main)
    run_step("validity_check_downloaded_dataset", validity_check_downloaded_dataset.main)
    run_step("convert_to_CSV", convert_to_CSV.main)
    run_step("validity_and_completeness_check_CSV", validity_and_completeness_check_CSV.main)
    run_step("validity_check_hashed_reviews", validity_check_hashed_reviews.main)

    # Split and preprocessing.
    run_step("split_training_set_validation_set", split_training_set_validation_set.main)
    run_step("pre_processing_reviews", pre_processing_reviews.main)
    run_step("pre_processing_reviews_BERT", pre_processing_reviews_BERT.main)

    # Feature engineering.
    run_step("perform_TF_IDF", perform_TF_IDF.main)
    run_step("perform_BOW", perform_BOW.main)

    # Model training.
    run_step("train_MNB", train_MNB.main)
    run_step("train_SVM", train_SVM.main)
    run_step("train_BERT", train_BERT.main)

    # Optional analysis.
    run_step("top_BOW", top_BOW.main)
    run_step("top_TFIDF", top_TFIDF.main)

    # Final test.
    run_step("model_tests", model_tests.main)


if __name__ == "__main__":
    main()