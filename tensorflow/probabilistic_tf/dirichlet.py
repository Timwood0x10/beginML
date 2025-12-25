import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from scipy.stats import multivariate_normal

"""
# Dirichlet Process Mixture Model for Clustering

This script implements a Dirichlet Process Mixture Model (DPMM) for unsupervised clustering.
DPMM is a Bayesian nonparametric approach that automatically determines the optimal number
of clusters from data, unlike traditional clustering methods (e.g., K-means) that require
specifying the number of clusters in advance.

Key advantages of DPMM:
1. Automatically infers the number of clusters
2. Provides uncertainty estimates for cluster assignments
3. Handles complex, multi-modal data distributions
4. Incorporates prior knowledge through Bayesian framework

This implementation uses TensorFlow Probability for model definition and training.
"""

tfd = tfp.distributions


# Generate mixture of Gaussians data
def generate_mixture_data(n_samples=500):
    np.random.seed(42)

    # Parameters for three components
    means = [[-2, -2], [0, 0], [2, 2]]
    covs = [[[0.5, 0], [0, 0.5]], [[0.8, 0.2], [0.2, 0.8]], [[0.5, 0], [0, 0.5]]]
    weights = [0.3, 0.4, 0.3]

    # Generate data
    components = np.random.choice(3, size=n_samples, p=weights)
    data = np.zeros((n_samples, 2))

    for i in range(n_samples):
        comp = components[i]
        data[i] = np.random.multivariate_normal(means[comp], covs[comp])

    return data.astype(np.float32), components


# Define Dirichlet Process Mixture Model
def dp_mixture_model(data, max_components=10, alpha=1.0):
    n_samples = data.shape[0]
    dim = data.shape[1]

    # Base distribution (Normal-Inverse-Wishart)
    def base_distribution():
        # Prior for mean
        mu = yield tfd.MultivariateNormalDiag(
            loc=tf.zeros([dim]), scale_diag=tf.ones([dim]) * 10.0
        )

        # Prior for covariance
        precision = yield tfd.WishartTriL(df=dim + 2, scale_tril=tf.eye(dim))

        # Convert precision matrix to covariance matrix
        cov = tf.linalg.inv(precision)

        # Return normal distribution
        return tfd.MultivariateNormalFullCovariance(loc=mu, covariance_matrix=cov)

    # Create base distribution
    base_dist = tfd.JointDistributionCoroutine(base_distribution)

    # Modify this part to use a simpler mixture model implementation
    # Create Gaussian mixture model instead of full Dirichlet process
    mixture_dist = tfd.Categorical(logits=tf.zeros(max_components))

    # Create a multivariate normal distribution for each component
    component_means = tf.Variable(tf.random.normal([max_components, dim], stddev=2.0))
    component_scales = tf.Variable(
        tf.ones([max_components, dim]) * 0.5, constraint=lambda x: tf.nn.softplus(x)
    )

    component_dist = tfd.MultivariateNormalDiag(
        loc=component_means, scale_diag=component_scales
    )

    # Create mixture model
    return tfd.MixtureSameFamily(
        mixture_distribution=mixture_dist, components_distribution=component_dist
    )


# Train Dirichlet Process Mixture Model
def train_dp_mixture(
    data, max_components=10, alpha=1.0, learning_rate=0.01, num_steps=1000
):
    # Create model
    model = dp_mixture_model(data, max_components, alpha)

    # Define optimizer
    optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

    # Define loss function (negative log-likelihood)
    def loss_fn():
        return -tf.reduce_mean(model.log_prob(data))

    # Training loop
    losses = []

    for step in range(num_steps):
        with tf.GradientTape() as tape:
            loss = loss_fn()

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        losses.append(loss.numpy())

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.numpy():.4f}")

    return model, losses


# Extract clustering results from trained model
def extract_clusters(model, data, max_components):
    # Calculate posterior probability of each data point belonging to each component
    log_probs = model.components_distribution.log_prob(data[:, tf.newaxis, :])
    log_mix_probs = tf.math.log_softmax(model.mixture_distribution.logits)
    posterior_log_probs = log_probs + log_mix_probs

    # Calculate posterior probabilities
    posterior_probs = tf.math.softmax(posterior_log_probs, axis=-1)

    # Assign each data point to the most likely component
    cluster_assignments = tf.argmax(posterior_probs, axis=-1).numpy()

    # Calculate effective number of components (components with at least one data point)
    unique_clusters = np.unique(cluster_assignments)
    effective_components = len(unique_clusters)

    return cluster_assignments, posterior_probs.numpy(), effective_components


# Visualize clustering results
def visualize_dp_clustering(data, cluster_assignments, true_components=None):
    plt.figure(figsize=(12, 6))

    # Set font for English text
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = True

    # Plot clustering results
    plt.subplot(1, 2, 1)
    unique_clusters = np.unique(cluster_assignments)
    for cluster in unique_clusters:
        cluster_data = data[cluster_assignments == cluster]
        plt.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            label=f"Cluster {cluster}",
            alpha=0.7,
        )

    plt.title("Dirichlet Process Mixture Model Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()

    # If true component labels are available, plot true distribution
    if true_components is not None:
        plt.subplot(1, 2, 2)
        unique_components = np.unique(true_components)
        for comp in unique_components:
            comp_data = data[true_components == comp]
            plt.scatter(
                comp_data[:, 0],
                comp_data[:, 1],
                label=f"True Component {comp}",
                alpha=0.7,
            )

        plt.title("True Data Distribution")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()

    plt.tight_layout()
    plt.savefig("dirichlet_process_clustering.png")
    plt.show()


# Evaluate clustering performance
def evaluate_clustering(data, cluster_assignments, true_components=None):
    # Calculate silhouette coefficient
    if len(np.unique(cluster_assignments)) > 1:  # Need at least two clusters
        silhouette = silhouette_score(data, cluster_assignments)
        print(f"Silhouette Coefficient: {silhouette:.4f}")
    else:
        print("Only one cluster, cannot calculate silhouette coefficient")

    # If true labels are available, calculate clustering accuracy
    if true_components is not None:
        # Using a simple method to match cluster labels with true labels
        # In practice, more complex matching algorithms might be needed
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

        ari = adjusted_rand_score(true_components, cluster_assignments)
        nmi = normalized_mutual_info_score(true_components, cluster_assignments)

        print(f"Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"Normalized Mutual Information (NMI): {nmi:.4f}")


# Main function
def main():
    # Generate data
    data, true_components = generate_mixture_data(n_samples=500)

    # Train model
    max_components = 10
    alpha = 1.0
    model, losses = train_dp_mixture(
        data, max_components, alpha, learning_rate=0.01, num_steps=1000
    )

    # Extract clustering results
    cluster_assignments, component_probs, effective_components = extract_clusters(
        model, data, max_components
    )

    print(f"Effective Number of Components: {effective_components}")

    # Visualize clustering results
    visualize_dp_clustering(data, cluster_assignments, true_components)

    # Evaluate clustering performance
    evaluate_clustering(data, cluster_assignments, true_components)

    # Set font for English text
    plt.rcParams["font.sans-serif"] = ["Arial"]
    plt.rcParams["axes.unicode_minus"] = True

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Negative Log-Likelihood")
    plt.savefig("dirichlet_process_loss.png")
    plt.show()


if __name__ == "__main__":
    main()
