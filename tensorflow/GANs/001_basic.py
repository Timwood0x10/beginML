import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import time

# load MNIST datasets
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype("float32")
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1][-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 128  # Smaller batch size helps stabilize training
train_dataset = (
    tf.data.Dataset.from_tensor_slices(train_images)
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
)


# setup the generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))  # 使用更大的alpha值

    model.add(layers.Reshape((7, 7, 256)))

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))

    model.add(
        layers.Conv2DTranspose(
            1, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )

    return model


# setup the discriminator model
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            64, (5, 5), strides=(2, 2), padding="same", input_shape=[28, 28, 1]
        )
    )
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 添加一层以增加网络深度
    model.add(layers.Conv2D(256, (3, 3), strides=(1, 1), padding="same"))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


# loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


# Modify discriminator loss with label smoothing
def discriminator_loss(real_output, fake_output):
    real_labels = tf.ones_like(real_output) * 0.9  # add label smoothing to real labels
    real_loss = cross_entropy(real_labels, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


# setup the optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)  # use beta_1=0.5
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

generator = make_generator_model()
discriminator = make_discriminator_model()


def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()  # time trace

        epoch_gen_loss = 0
        epoch_disc_loss = 0
        num_batches = 0

        for image_batch in dataset:
            for _ in range(2):
                disc_loss = train_discriminator(image_batch)
                epoch_disc_loss += disc_loss

            gen_loss = train_generator()
            epoch_gen_loss += gen_loss

            num_batches += 1

        print(
            f"Epoch {epoch+1}, Generator Loss: {epoch_gen_loss/num_batches}, "
            f"Discriminator Loss: {epoch_disc_loss/(num_batches*2)}, "
            f"Time: {time.time()-start} sec"
        )

        generate_and_save_images(generator, epoch + 1)


# split training steps for discriminator
@tf.function
def train_discriminator(images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return disc_loss


# splet training steps for generator
@tf.function
def train_generator():
    noise = tf.random.normal([BATCH_SIZE, 100])

    with tf.GradientTape() as gen_tape:
        generated_images = generator(noise, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )

    return gen_loss


def generate_and_save_images(model, epoch):
    noise = tf.random.normal([16, 100])
    generated_images = model(noise, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0] * 0.5 + 0.5, cmap="gray")
        plt.axis("off")

    plt.savefig(f"gan_image_epoch_{epoch}.png")
    plt.close()


EPOCHS = 20
train(train_dataset, EPOCHS)
