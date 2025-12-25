import argparse
import gensim.downloader as api
import tensorflow.keras as K
import numpy as np
import os
import shutil
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix
from help_function import download_and_read, DATASET_URL

"""
# SMS Spam Classification Model
This script implements a CNN-based model for classifying SMS messages as spam or ham (non-spam).
It uses word embeddings (either trained from scratch or pre-trained) and a 1D convolutional layer
to extract features from text sequences.
"""

# Load and preprocess the SMS dataset
texts, labels = download_and_read(DATASET_URL)
# Tokenize and pad text
tokenizer = K.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
# print(f"Found {len(word_index)} unique tokens.")

text_sequences = K.preprocessing.sequence.pad_sequences(sequences)
num_records = len(text_sequences)
max_seqlen = len(text_sequences[0])
# print("{:d} sentences, max length: {:d}".format(num_records, max_seqlen))

# Convert labels to categorical format for classification
NUM_CLASSES = 2
cat_labels = K.utils.to_categorical(labels, NUM_CLASSES)

# Create vocabulary mappings (word to index and index to word)
word2idx = tokenizer.word_index
idx2word = {v: k for k, v in word2idx.items()}
word2idx["PAD"] = 0
idx2word[0] = "PAD"
vocab_size = len(word2idx)
print("vocab size: {:d}".format(vocab_size))

# Create TensorFlow datasets for training, validation, and testing
dataset = tf.data.Dataset.from_tensor_slices((text_sequences, cat_labels))
dataset = dataset.shuffle(10000)
test_size = num_records // 4
val_size = (num_records - test_size) // 10
test_dataset = dataset.take(test_size)
val_dataset = dataset.skip(test_size).take(val_size)
train_dataset = dataset.skip(test_size + val_size)
BATCH_SIZE = 128
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)


def build_embedding_matrix(sequences, word2idx, embedding_dim, embedding_file):
    """
    Builds or loads a word embedding matrix for the vocabulary.

    Args:
        sequences: Tokenized and padded text sequences
        word2idx: Dictionary mapping words to indices
        embedding_dim: Dimension of the embedding vectors
        embedding_file: Path to save/load the embedding matrix

    Returns:
        numpy.ndarray: Embedding matrix of shape (vocab_size, embedding_dim)
    """
    if os.path.exists(embedding_file):
        E = np.load(embedding_file)
    else:
        vocab_size = len(word2idx)
        E = np.zeros((vocab_size, embedding_dim))
        word_vectors = api.load(EMBEDDING_MODEL)
        for word, idx in word2idx.items():
            try:
                # Replace word_vec with get_vector
                E[idx] = word_vectors.get_vector(word)
            except KeyError:  # word not in embedding
                pass
        np.save(embedding_file, E)
    return E


# Configure embedding parameters
EMBEDDING_DIM = 300
DATA_DIR = "data"
EMBEDDING_NUMPY_FILE = os.path.join(DATA_DIR, "E.npy")
EMBEDDING_MODEL = "glove-wiki-gigaword-300"
E = build_embedding_matrix(
    text_sequences, word2idx, EMBEDDING_DIM, EMBEDDING_NUMPY_FILE
)
print("Embedding matrix:", E.shape)


class SpamClassifierModel(tf.keras.Model):
    """
    CNN-based model for spam classification.

    This model uses word embeddings followed by a 1D convolutional layer,
    spatial dropout, global max pooling, and a dense output layer.

    The model can be configured to:
    - Train embeddings from scratch
    - Use pre-trained embeddings as fixed feature extractors
    - Use pre-trained embeddings and fine-tune them
    """

    def __init__(
        self,
        vocab_sz,
        embed_sz,
        input_length,
        num_filters,
        kernel_sz,
        output_sz,
        run_mode,
        embedding_weights,
        **kwargs
    ):
        super(SpamClassifierModel, self).__init__(**kwargs)
        if run_mode == "scarch":
            # Train embeddings from scratch
            self.embedding = K.layers.Embedding(
                vocab_sz, embed_sz, input_length=input_length, trainable=True
            )
        elif run_mode == "vectorizer":
            # Use pre-trained embeddings as fixed feature extractors
            self.embedding = K.layers.Embedding(
                vocab_sz,
                embed_sz,
                input_length=input_length,
                weights=[embedding_weights],
                trainable=True,
            )
        else:
            # Use pre-trained embeddings and fine-tune them
            self.embedding = K.layers.Embedding(
                vocab_sz,
                embed_sz,
                input_length=input_length,
                weights=[embedding_weights],
                trainable=True,
            )
        self.conv = tf.keras.layers.Conv1D(
            filters=num_filters, kernel_size=kernel_sz, activation="relu"
        )
        self.dropout = tf.keras.layers.SpatialDropout1D(0.2)
        self.pool = tf.keras.layers.GlobalMaxPooling1D()
        self.dense = tf.keras.layers.Dense(output_sz, activation="softmax")

    def call(self, inputs, training=None):
        """Forward pass of the model"""
        x = self.embedding(inputs)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        return self.dense(x)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Spam classifier model")
    parser.add_argument(
        "--mode",
        type=str,
        default="pretrained",
        choices=["scarch", "vectorizer", "pretrained"],
        help="Mode to run the model: from scratch, as vectorizer, or with pretrained embeddings",
    )
    args = parser.parse_args()
    run_mode = args.mode

    # Model configuration
    conv_num_filters = 256
    conv_kernel_size = 3
    model = SpamClassifierModel(
        vocab_size,
        EMBEDDING_DIM,
        max_seqlen,
        conv_num_filters,
        conv_kernel_size,
        NUM_CLASSES,
        run_mode,
        E,
    )
    model.build(input_shape=(None, max_seqlen))

    # Compile model with appropriate loss and metrics
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    NUM_EPOCHS = 3
    # Class weights to handle imbalanced dataset
    # Data distribution is 4827 ham and 747 spam (total 5574), which
    # works out to approx 87% ham and 13% spam, so we take reciprocals
    # and this works out to being each spam (1) item as being
    # approximately 8 times as important as each ham (0) message.
    CLASS_WEIGHTS = {0: 1, 1: 8}

    # Train model
    model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=val_dataset,
        class_weight=CLASS_WEIGHTS,
    )

    # Evaluate against test set
    labels, predictions = [], []
    for Xtest, Ytest in test_dataset:
        Y_test = model.predict_on_batch(Xtest)
        ytest = np.argmax(Y_test, axis=1)
        ytest = np.argmax(Y_test, axis=1)
        labels.extend(ytest.tolist())
        predictions.extend(ytest.tolist())

    # Print evaluation metrics
    print("test accuracy: {:.3f}".format(accuracy_score(labels, predictions)))
    print("confusion matrix")
    print(confusion_matrix(labels, predictions))
