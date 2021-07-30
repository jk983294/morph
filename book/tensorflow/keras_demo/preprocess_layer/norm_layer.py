import tensorflow as tf
import numpy as np
from book.tensorflow.keras_demo.train.training import get_compiled_model
from tensorflow.keras.layers.experimental import preprocessing


class VariationalAutoEncoder(tf.keras.Model):
    """Combines the encoder and decoder into an end-to-end model for training."""

    def __init__(
            self,
            original_dim,
            intermediate_dim=64,
            latent_dim=32,
            name="autoencoder",
            **kwargs
    ):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, intermediate_dim=intermediate_dim)
        self.decoder = Decoder(original_dim, intermediate_dim=intermediate_dim)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(
            z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1
        )
        self.add_loss(kl_loss)
        return reconstructed


if __name__ == '__main__':
    data = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [1.5, 1.6, 1.7], ])
    normalizer = preprocessing.Normalization()
    normalizer.adapt(data)
    # normalized_data = normalizer(data)
    # print("Features mean: %.2f" % (normalized_data.numpy().mean()))
    # print("Features std: %.2f" % (normalized_data.numpy().std()))

    # make norm layer as part of model
    inputs = tf.keras.Input(shape=(10,))
    x = normalizer(inputs)
    x = tf.keras.layers.Dense(64, activation="relu")(inputs)
    outputs = tf.keras.layers.Dense(1, activation="linear")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
