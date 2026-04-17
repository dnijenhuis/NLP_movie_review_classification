# This module is for downloading and extracting the IMDB-dataset.
# The directories need to be adjusted before use.

import urllib.request # Needed for downloading the dataset from the URL.
import tarfile # Needed for handling TAR-files (like the IMDB-dataset).
from pathlib import Path # Needed for path handling.


def main() -> None:
    # This function downloads and extracts the IMDB-dataset for this project.
    # It also creates the necessary directories.
    URL_download = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    download_dir = Path("PATH_TO_DOWNLOAD_DIR")
    extract_dir = Path("PATH_TO_EXTRACT_DIR")

    # Create paths if needed.
    download_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    archive_path = download_dir / "aclImdb_v1.tar.gz" # Name the downloaded file.

    print("Downloading")
    # Downloads the dataset to the directory using the determined name.
    urllib.request.urlretrieve(URL_download, archive_path)

    print("Extracting")
    # Extracts the file to the directory "Extracted".
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    print("Downloading and extracting. Dataset root:", extract_dir / "aclImdb")


if __name__ == "__main__":
    main()