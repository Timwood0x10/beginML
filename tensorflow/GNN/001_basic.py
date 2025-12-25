import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

"""
Graph Convolutional Network (GCN) Implementation
This implementation demonstrates a basic Graph Convolutional Network using TensorFlow.
GCNs are designed to work with graph-structured data by leveraging the connectivity
information between nodes to perform node classification, link prediction, or graph classification.

Key components:
- GCNLayer: Implements the graph convolution operation
- GCN: A simple GCN model with two layers
- Example usage with a synthetic graph
"""


class GCNLayer(layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        # Get feature dimension
        feature_dim = input_shape[0][-1]
        self.weight = self.add_weight(
            shape=(feature_dim, self.units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True
        )
        self.built = True

    def call(self, inputs):
        # inputs: [node_features, adjacency_matrix]
        features, adj = inputs

        # Process batch dimension - modified to use tf.cond to support graph execution mode
        batch_size = tf.shape(features)[0]

        # Use tf.cond instead of Python if statement
        return tf.cond(
            tf.equal(batch_size, 1),
            lambda: self._process_single_graph(features, adj),
            lambda: self._process_batch_graphs(features, adj),
        )

    def _process_single_graph(self, features, adj):
        # Process a single graph
        features = tf.squeeze(features, axis=0)
        adj = tf.squeeze(adj, axis=0)

        # Normalize adjacency matrix
        D = tf.reduce_sum(adj, axis=1)
        D = tf.math.pow(D, -0.5)
        D = tf.linalg.diag(D)
        adj_norm = tf.matmul(tf.matmul(D, adj), D)

        # Graph convolution operation
        support = tf.matmul(features, self.weight)
        output = tf.matmul(adj_norm, support)
        output = output + self.bias

        if self.activation is not None:
            output = self.activation(output)

        # Restore batch dimension
        return tf.expand_dims(output, axis=0)

    def _process_batch_graphs(self, features, adj):
        # Process batch of graphs
        batch_size = tf.shape(features)[0]

        # Use tf.map_fn to process batch data
        def process_graph(inputs):
            feat, a = inputs

            # Normalize adjacency matrix
            D = tf.reduce_sum(a, axis=1)
            D = tf.math.pow(D, -0.5)
            D = tf.linalg.diag(D)
            adj_norm = tf.matmul(tf.matmul(D, a), D)

            # Graph convolution operation
            support = tf.matmul(feat, self.weight)
            out = tf.matmul(adj_norm, support)
            out = out + self.bias

            if self.activation is not None:
                out = self.activation(out)

            return out

        # Use map_fn to process each graph
        return tf.map_fn(
            process_graph,
            (features, adj),
            fn_output_signature=tf.TensorSpec(
                shape=(None, self.units), dtype=features.dtype
            ),
        )


class GCN(Model):
    def __init__(self, hidden_units, num_classes):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(hidden_units, activation=tf.nn.relu)
        self.gcn2 = GCNLayer(num_classes)
        self.dropout = layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.gcn1(inputs)
        x = self.dropout(x, training=training)
        x = self.gcn2([x, inputs[1]])
        return x

    def build(self, input_shape):
        # Explicitly build the model
        self.gcn1.build([input_shape[0], input_shape[1]])
        # Calculate gcn1 output shape
        gcn1_output_shape = (input_shape[0][0], input_shape[0][1], self.gcn1.units)
        self.gcn2.build([gcn1_output_shape, input_shape[1]])
        self.built = True


# Example: Build a simple graph
# Assume 5 nodes, each with 3 features
node_features = np.random.randn(5, 3).astype(np.float32)

# Adjacency matrix (including self-loops)
adjacency_matrix = np.array(
    [
        [1, 1, 0, 0, 1],
        [1, 1, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [1, 0, 0, 1, 1],
    ],
    dtype=np.float32,
)

# Add batch dimension
node_features = np.expand_dims(node_features, axis=0)  # Shape becomes (1, 5, 3)
adjacency_matrix = np.expand_dims(adjacency_matrix, axis=0)  # Shape becomes (1, 5, 5)

# Create GCN model
model = GCN(hidden_units=16, num_classes=2)

# Build model
model.build([(None, 5, 3), (None, 5, 5)])
"""
Model: "gcn"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ gcn_layer (GCNLayer)                 │ ?                           │              64 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ gcn_layer_1 (GCNLayer)               │ ?                           │              34 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ ?                           │               0 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 98 (392.00 B)
 Trainable params: 98 (392.00 B)
 Non-trainable params: 0 (0.00 B)
"""
# print(model.summary())

# Forward pass test
outputs = model([node_features, adjacency_matrix], training=False)
print("Output shape:", outputs.shape)

# Since we don't have label data, we create some dummy labels for demonstration
dummy_labels = np.random.randint(0, 2, size=(1, 5, 2)).astype(np.float32)

# Compile model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# Train model
history = model.fit(
    [node_features, adjacency_matrix], dummy_labels, epochs=10, batch_size=1, verbose=1
)

# Use trained model for prediction
predictions = model([node_features, adjacency_matrix], training=False)
print("Prediction shape:", predictions.shape)
print("Prediction results:", predictions)
