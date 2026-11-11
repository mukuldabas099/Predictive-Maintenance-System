"""
models.py -- alternative autoencoder architectures that plug into the SAME
training / calibration / multi-condition machinery as LSTMAutoencoder
(they only override the network; fit(), thresholds, ONNX export, the bridge
all work unchanged).

  ConvAutoencoder   1-D convolutions over the window sequence
  TCNAutoencoder    residual dilated causal convolutions (temporal conv net)

Both are much cheaper to train than the LSTM and export cleanly to ONNX.
Use:  from models import ConvAutoencoder;  ae = ConvAutoencoder(15, 14)
"""
from tensorflow import keras
from tensorflow.keras import layers

from lstm_autoencoder import LSTMAutoencoder


class ConvAutoencoder(LSTMAutoencoder):
    def _build(self, latent_dim, units):
        inp = keras.Input(shape=(self.seq_len, self.n_features))
        x = layers.Conv1D(units, 3, padding="same", activation="relu")(inp)
        x = layers.Conv1D(units // 2, 3, padding="same", activation="relu")(x)
        x = layers.Flatten()(x)
        z = layers.Dense(latent_dim, activation="tanh")(x)
        x = layers.Dense(self.seq_len * (units // 2), activation="relu")(z)
        x = layers.Reshape((self.seq_len, units // 2))(x)
        x = layers.Conv1D(units, 3, padding="same", activation="relu")(x)
        out = layers.Conv1D(self.n_features, 3, padding="same")(x)
        m = keras.Model(inp, out, name="conv_autoencoder")
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return m


def _tcn_block(x, units, dilation):
    y = layers.Conv1D(units, 3, padding="causal", dilation_rate=dilation, activation="relu")(x)
    y = layers.Conv1D(units, 3, padding="causal", dilation_rate=dilation)(y)
    if x.shape[-1] != units:
        x = layers.Conv1D(units, 1, padding="same")(x)      # match channels for the residual
    return layers.Activation("relu")(layers.Add()([x, y]))


class TCNAutoencoder(LSTMAutoencoder):
    def _build(self, latent_dim, units):
        inp = keras.Input(shape=(self.seq_len, self.n_features))
        x = inp
        for d in (1, 2, 4):
            x = _tcn_block(x, units, d)
        z = layers.Dense(latent_dim, activation="tanh")(layers.Flatten()(x))
        x = layers.Dense(self.seq_len * units, activation="relu")(z)
        x = layers.Reshape((self.seq_len, units))(x)
        for d in (4, 2, 1):
            x = _tcn_block(x, units, d)
        out = layers.Dense(self.n_features)(x)
        m = keras.Model(inp, out, name="tcn_autoencoder")
        m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
        return m


MODELS = {"lstm": LSTMAutoencoder, "cnn": ConvAutoencoder, "tcn": TCNAutoencoder}
