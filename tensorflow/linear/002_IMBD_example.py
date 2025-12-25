import tensorflow as tf
from tensorflow.keras import datasets, layers, models, preprocessing
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import tensorflow.keras as keras

"""
# IMDB Sentiment Analysis Model
This script implements a neural network model for sentiment analysis on the IMDB movie reviews dataset.
The model classifies movie reviews as positive or negative using a simple architecture with:
- Word embeddings to convert text to numerical vectors
- Global max pooling to extract the most important features
- Dense layers with dropout for classification
"""

# Model configuration parameters
MAX_LEN = 200  # Maximum length of input sequences (reviews)
N_WORDS = 1_000  # Vocabulary size - only consider the top N words
EPOCHSS = 20  # Maximum number of training epochs
BATCH_SIZE = 500  # Number of samples per gradient update
DIM_EMBEDDING = 256  # Dimension of the word embedding vectors
OUTPUT_DIR = "./notes/images"  # Directory to save output images
MODEL_NAME = "imbd_model"  # Name for saving model artifacts


def load_data():
    """
    Load and preprocess the IMDB dataset.

    Returns:
        tuple: Training and testing data with their labels
              ((X_train, y_train), (X_test, y_test))
    """
    (X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=N_WORDS)
    # Pad sequences to have the same length
    # because the nn requires a fixed input size to be of uniform size
    X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN)
    X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=MAX_LEN)
    return (X_train, y_train), (X_test, y_test)


def build_model():
    """
    Build and return the sentiment analysis model architecture.

    The model consists of:
    1. Embedding layer to convert word indices to dense vectors
    2. Dropout for regularization
    3. Global max pooling to extract the most important features
    4. Two dense layers with L2 regularization and ReLU activation
    5. Output layer with sigmoid activation for binary classification

    Returns:
        keras.Model: Compiled Keras sequential model
    """
    model = models.Sequential()
    # input: embedding layer
    # The model will take as input an integer matrix of size (batch, input_
    # length).
    #  The model will output dimension (input_length, dim_embedding).
    # used for nlp
    model.add(layers.Embedding(N_WORDS, DIM_EMBEDDING, input_length=MAX_LEN))
    model.add(layers.Dropout(0.3))

    # Takes the maximum value of either feature vector from each of the n_words
    # features.
    model.add(layers.GlobalMaxPooling1D())
    model.add(
        layers.Dense(
            128,
            name="dense_layer_1",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
    )
    model.add(layers.Dropout(0.3))
    model.add(
        layers.Dense(
            128,
            name="dense_layer_2",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(0.01),
        ),
    )
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation="sigmoid"))

    return model


if __name__ == "__main__":
    # Display TensorFlow version for reproducibility
    print(f"TensorFlow version: {tf.__version__}")

    # Load and preprocess the IMDB dataset
    (X_train, y_train), (X_test, y_test) = load_data()

    # Build the model architecture
    model = build_model()

    """
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 200, 256)          256000    
                                                                 
 dropout (Dropout)           (None, 200, 256)          0         
                                                                 
 global_max_pooling1d (Glob  (None, 256)               0         
 alMaxPooling1D)                                                 
                                                                 
 dense_layer_1 (Dense)       (None, 128)               32896     
                                                                 
 dropout_1 (Dropout)         (None, 128)               0         
                                                                 
 dense_layer_2 (Dense)       (None, 128)               16512     
                                                                 
 dropout_2 (Dropout)         (None, 128)               0         
                                                                 
 dense (Dense)               (None, 1)                 129       
                                                                 
=================================================================
Total params: 305537 (1.17 MB)
Trainable params: 305537 (1.17 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________

    """
    # print(f"summary model: {model.summary()}\n")

    # Compile the model with binary cross-entropy loss (suitable for binary classification)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Implement early stopping to optimize model overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    # Train the model with early stopping
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHSS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=early_stopping,
    )

    # Evaluate the model on test data
    model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)

    # Print model summary
    print(f"summary model: {model.summary()}\n")

    # Visualize training history
    # Create a figure with two subplots
    plt.figure(figsize=(12, 6))

    # Plot accuracy metrics
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="train Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    # Plot loss metrics
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")

    # Save and display the visualization
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/{MODEL_NAME}.png")
    plt.show(block=False)
    plt.close()

    # Generate predictions on test data
    predictions = model.predict(X_test)
    print(predictions)
