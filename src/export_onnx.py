"""
export_onnx.py -- convert the trained Keras LSTM autoencoder to ONNX so the
live bridge can run with onnxruntime only (no TensorFlow on the server).

    python3 export_onnx.py            # writes outputs/lstm_autoencoder.onnx
    pip install onnxruntime           # on the serving machine

The export is verified against Keras on random inputs; it fails loudly if the
two models disagree.
"""
import os
import sys

import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def export(keras_path=None, onnx_path=None, atol=1e-4):
    from tensorflow import keras
    import onnxruntime as ort
    keras_path = keras_path or os.path.join(OUT_DIR, "lstm_autoencoder.keras")
    onnx_path = onnx_path or os.path.join(OUT_DIR, "lstm_autoencoder.onnx")
    model = keras.models.load_model(keras_path)
    _, seq_len, n_feat = model.input_shape
    model.predict(np.zeros((1, seq_len, n_feat), np.float32), verbose=0)   # build graph
    model.export(onnx_path, format="onnx")

    x = np.random.default_rng(0).normal(size=(64, seq_len, n_feat)).astype(np.float32)
    ref = model.predict(x, verbose=0)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    err = float(np.abs(ref - got).max())
    if err > atol:
        raise RuntimeError(f"ONNX output differs from Keras (max abs diff {err:.2e} > {atol})")
    print(f"Exported {onnx_path}  (max |keras - onnx| = {err:.2e})")
    return onnx_path


if __name__ == "__main__":
    export()
