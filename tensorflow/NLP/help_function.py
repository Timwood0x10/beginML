import gensim.downloader as api
from gensim.models import Word2Vec
import tensorflow as tf
import os

"""
# NLP Helper Functions
This module provides utility functions for NLP tasks, including:
- Downloading and processing text datasets
- Loading pre-trained word embeddings
"""

# Commented code for downloading and saving Word2Vec model
# dataset = api.load("text8")
# model = Word2Vec(dataset)
# model.save("./data/text8-word2vec.bin")


def download_and_read(url):
    """
    Downloads and reads a text dataset from a URL.

    This function handles:
    1. Checking if the file already exists locally
    2. Downloading and extracting if needed
    3. Reading and parsing the SMS spam collection dataset

    Args:
        url (str): URL to download the dataset from

    Returns:
        tuple: (texts, labels) where texts is a list of SMS messages
               and labels is a list of binary values (1 for spam, 0 for ham)
    """
    local_file = url.split("/")[-1]
    cache_dir = "."
    cache_subdir = "datasets"

    # Check if the file already exists
    file_path = os.path.join(cache_dir, cache_subdir, "SMSSpamCollection")
    if os.path.exists(file_path):
        print(f"File already exists at: {file_path}")
    else:
        print("Downloading file...")
        path = tf.keras.utils.get_file(
            local_file,
            url,
            extract=True,
            cache_dir=cache_dir,
            cache_subdir=cache_subdir,
        )
        print(f"Downloaded file path: {path}")

        # The file might be in a subdirectory after extraction
        extract_dir = os.path.dirname(path)

        # Find the actual path of SMSSpamCollection
        for root, dirs, files in os.walk(extract_dir):
            if "SMSSpamCollection" in files:
                file_path = os.path.join(root, "SMSSpamCollection")
                break

    print(f"Using file at: {file_path}")

    # Read the file
    labels, texts = [], []
    with open(file_path, "r") as fin:
        for line in fin:
            label, text = line.strip().split("\t")
            labels.append(1 if label == "spam" else 0)
            texts.append(text)

    return texts, labels


# URL for the SMS Spam Collection dataset
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
