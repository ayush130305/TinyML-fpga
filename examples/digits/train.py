# train.py - PC: train a small CNN on the 8x8 sklearn digits (no download needed), export a fully
# int8 .tflite, convert it to digits.tmf, check it bit-exact against LiteRT, and write samples.py
# (a few test digits for the board).
#   pip install tensorflow scikit-learn tflite ai-edge-litert
#   python train.py
import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.datasets import load_digits

# Use the repo's tinyml_fpga without installing it.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "firmware"))
import tinyml_fpga
from tinyml_fpga.check import compare

# Output folder = this folder.
HERE = os.path.dirname(os.path.abspath(__file__))

# Data: 1797 images, 8x8, values 0..16 -> 0..1.
d = load_digits()
X = (d.images / 16.0).astype(np.float32)[..., None]
y = d.target
# Hold out every 5th image for testing.
test = np.arange(len(X)) % 5 == 0

# Model: every layer type the engine accelerates (conv, depthwise, dense) plus CPU pooling/softmax.
L = tf.keras.layers
model = tf.keras.Sequential([
    tf.keras.Input((8, 8, 1)),
    L.Conv2D(8, 3, padding="same", activation="relu"),
    L.MaxPooling2D(),
    L.DepthwiseConv2D(3, padding="same", activation="relu6"),
    L.Conv2D(16, 3, strides=2, padding="valid", activation="relu"),
    L.Flatten(),
    L.Dense(10),
    L.Softmax(),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X[~test], y[~test], epochs=30, verbose=0)
print("float test accuracy: %.3f" % model.evaluate(X[test], y[test], verbose=0)[1])

# Full-integer quantisation with int8 input and output.
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = lambda: ([X[i:i + 1]] for i in range(0, len(X), 5))
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = os.path.join(HERE, "digits.tflite")
with open(tfl, "wb") as f:
    f.write(conv.convert())

# Convert for the board.
tmf = os.path.join(HERE, "digits.tmf")
tinyml_fpga.convert(tfl, tmf)
print("wrote", tmf, os.path.getsize(tmf), "bytes")

# Bit-exact check on the test images.
m = tinyml_fpga.load(tmf, backend="sim")
s, z = m.input["scale"], m.input["zp"]
samples = [np.clip(np.round(x / s) + z, -128, 127).astype(np.int8) for x in X[test][:100]]
compare(tfl, samples, backend="sim")

# int8 accuracy through tinyml_fpga.
acc = np.mean([m.classify(x)[0] == t for x, t in zip(X[test], y[test])])
print("tinyml_fpga int8 test accuracy: %.3f" % acc)

# A few test digits for the board, as plain lists.
with open(os.path.join(HERE, "samples.py"), "w") as f:
    f.write("# samples.py - test digits for main.py (written by train.py)\n")
    f.write("SAMPLES = [\n")
    for x, t in list(zip(X[test], y[test]))[:10]:
        f.write("    (%d, %s),\n" % (t, [round(float(v), 4) for v in x.reshape(-1)]))
    f.write("]\n")
print("wrote samples.py")
