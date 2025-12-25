import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt

"""
# Gaussian Process Regression

This script implements Gaussian Process Regression (GPR) using TensorFlow Probability.
Gaussian Process Regression is a powerful non-parametric Bayesian approach that:
1. Models complex non-linear relationships
2. Provides uncertainty estimates for predictions
3. Requires minimal hyperparameter tuning
4. Works well with small datasets

The implementation uses an Exponentiated Quadratic kernel (RBF) and optimizes
the kernel parameters using maximum likelihood estimation.
"""


# Generate synthetic data
def generate_data(n_samples=50, noise_std=0.1):
    np.random.seed(42)
    x_data = np.linspace(-1.5, 1.5, n_samples).astype(np.float32)

    # Generate non-linear function: f(x) = sin(3x) + x^2
    true_function = lambda x: np.sin(3 * x) + x**2
    y_data = true_function(x_data) + np.random.normal(
        0, noise_std, size=n_samples
    ).astype(np.float32)

    return x_data, y_data, true_function


# Get data
x_data, y_data, true_function = generate_data()

# Define kernel function
kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(
    amplitude=tf.Variable(1.0, name="amplitude"),
    length_scale=tf.Variable(1.0, name="length_scale"),
)

# Create Gaussian process regression model
gp = tfp.distributions.GaussianProcess(
    kernel=kernel, index_points=x_data[:, tf.newaxis], observation_noise_variance=0.1
)

# Calculate log likelihood
log_likelihood = gp.log_prob(y_data)


# Optimize kernel parameters
def loss_fn():
    return -gp.log_prob(y_data)


optimizer = tf.optimizers.Adam(learning_rate=0.01)


# Use GradientTape for optimization
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        loss = loss_fn()
    gradients = tape.gradient(loss, [kernel.amplitude, kernel.length_scale])
    optimizer.apply_gradients(zip(gradients, [kernel.amplitude, kernel.length_scale]))
    return loss


# Training loop
for i in range(1000):
    loss_value = train_step()
    if i % 100 == 0:
        print(
            f"Step {i}: Loss = {loss_value.numpy()}, "
            f"Amplitude = {kernel.amplitude.numpy()}, "
            f"Length Scale = {kernel.length_scale.numpy()}"
        )

# Prediction
x_test = np.linspace(-2, 2, 100).astype(np.float32)

# Use GaussianProcessRegressionModel for prediction
gp_pred = tfp.distributions.GaussianProcessRegressionModel(
    kernel=kernel,
    index_points=x_test[:, tf.newaxis],
    observation_index_points=x_data[:, tf.newaxis],
    observations=y_data,
    observation_noise_variance=0.1,
)


# Set font for English text
plt.rcParams["font.sans-serif"] = ["Arial"]  # Font that supports English
plt.rcParams["axes.unicode_minus"] = True  # Correctly display minus sign


# Get prediction mean and variance
mean = gp_pred.mean()
stddev = tf.sqrt(gp_pred.variance())

# Calculate evaluation metrics
y_true_test = true_function(x_test)
mse = tf.reduce_mean(tf.square(mean - y_true_test))
rmse = tf.sqrt(mse)
mae = tf.reduce_mean(tf.abs(mean - y_true_test))

# Calculate R² score
ss_total = tf.reduce_sum(tf.square(y_true_test - tf.reduce_mean(y_true_test)))
ss_residual = tf.reduce_sum(tf.square(y_true_test - mean))
r2 = 1 - ss_residual / ss_total

# Calculate prediction interval coverage
in_interval = tf.reduce_mean(
    tf.cast(
        tf.logical_and(
            y_true_test >= mean - 2 * stddev, y_true_test <= mean + 2 * stddev
        ),
        tf.float32,
    )
)

print(f"Model Evaluation Metrics:")
print(f"MSE: {mse.numpy():.4f}")
print(f"RMSE: {rmse.numpy():.4f}")
print(f"MAE: {mae.numpy():.4f}")
print(f"R²: {r2.numpy():.4f}")
print(f"95% Confidence Interval Coverage: {in_interval.numpy():.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color="blue", alpha=0.5, label="Observed Data")
plt.plot(x_test, mean, "r-", label="GP Prediction Mean")
plt.fill_between(
    x_test,
    mean - 2 * stddev,
    mean + 2 * stddev,
    color="red",
    alpha=0.2,
    label="95% Confidence Interval",
)

# Plot true function
y_true = true_function(x_test)
plt.plot(x_test, y_true, "g--", label="True Function")

plt.title("Gaussian Process Regression")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig("gaussian_process_regression.png")
plt.show()

print(
    f"Final Kernel Parameters: Amplitude = {kernel.amplitude.numpy()}, Length Scale = {kernel.length_scale.numpy()}"
)
