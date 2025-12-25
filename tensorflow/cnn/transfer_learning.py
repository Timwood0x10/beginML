import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# create the base pre-trained model
base_model = InceptionV3(
    weights="imagenet", include_top=False, input_shape=(299, 299, 3)
)

# Create the base pre-trained model
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
# let's add a fully-connected layer as first layer
x = layers.Dense(1024, activation="relu")(x)
# and a logistic layer with 200 classes as last layer
predictions = layers.Dense(200, activation="softmax")(x)
# model to train
model = models.Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(
    optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Configure data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)


import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../data"))
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(299, 299),  # InceptionV3 expects 299x299 images
    batch_size=32,
    class_mode="categorical",
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir, target_size=(299, 299), batch_size=32, class_mode="categorical"
)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
)


model.save()
