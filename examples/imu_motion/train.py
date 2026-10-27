# train.py - PC: train a 1-D CNN motion classifier on BMI323 windows and export motion.tmf.
#   python train.py data/*.csv     your recordings (record.py on the board writes them)
#   python train.py --synthetic    made-up idle / shake / circle data, to try the pipeline end to end
# CSV rows: label,ax,ay,az,gx,gy,gz   (g and deg/s, one row per sample at 100 Hz)
import glob
import os
import sys

import numpy as np
import tensorflow as tf

# Use the repo's tinyml_fpga without installing it.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "firmware"))
import tinyml_fpga
from tinyml_fpga.check import compare

# Samples per window (1.28 s at 100 Hz) and channels (accel xyz + gyro xyz).
WIN, CH = 128, 6
# Class names (order = label index).
CLASSES = ["idle", "shake", "circle"]
# Scaling into roughly [-1, 1]: accel / 4 g, gyro / 500 dps (the board must use the same).
SCALE = np.array([4, 4, 4, 500, 500, 500], np.float32)


# Made-up data with the same shape as real recordings.
def synthetic(n_per_class=300, seed=0):
    # Reproducible.
    rng = np.random.default_rng(seed)
    # Time axis.
    t = np.arange(WIN) / 100.0
    # Windows and labels.
    X, y = [], []
    for c in range(3):
        for _ in range(n_per_class):
            # Gravity on z plus sensor noise.
            w = rng.normal(0, 0.02, (WIN, CH)) + np.array([0, 0, 1, 0, 0, 0])
            # Shake: fast oscillation on x, gyro on y.
            if c == 1:
                f = rng.uniform(4, 8)
                w[:, 0] += rng.uniform(0.8, 2) * np.sin(2 * np.pi * f * t)
                w[:, 4] += rng.uniform(100, 300) * np.cos(2 * np.pi * f * t)
            # Circle: slow rotation in x/y with gyro on z.
            elif c == 2:
                f = rng.uniform(0.7, 1.5)
                w[:, 0] += 0.5 * np.cos(2 * np.pi * f * t)
                w[:, 1] += 0.5 * np.sin(2 * np.pi * f * t)
                w[:, 5] += rng.uniform(60, 150)
            X.append(w)
            y.append(c)
    # Arrays.
    return np.array(X, np.float32), np.array(y)


# Recordings: consecutive rows with the same label are cut into windows (87% overlap).
def from_csv(files):
    # Windows and labels.
    X, y = [], []
    for fn in files:
        # Rows.
        d = np.loadtxt(fn, delimiter=",")
        # Split into runs of one label.
        for c in np.unique(d[:, 0]).astype(int):
            r = d[d[:, 0] == c, 1:7]
            for s in range(0, len(r) - WIN + 1, WIN // 8):
                X.append(r[s:s + WIN])
                y.append(c)
    # Arrays.
    return np.array(X, np.float32), np.array(y)


# Fixed seed so a run can be repeated.
tf.keras.utils.set_random_seed(0)
# Data.
X, y = synthetic() if "--synthetic" in sys.argv else from_csv(sorted(sum((glob.glob(a) for a in sys.argv[1:]), [])))
X = X / SCALE
print("windows:", X.shape, "per class:", np.bincount(y))
# Train / test split.
idx = np.random.default_rng(1).permutation(len(X))
test, train = idx[: len(X) // 5], idx[len(X) // 5:]

# 1-D CNN: TFLite turns Conv1D into EXPAND_DIMS + CONV_2D + RESHAPE, all handled by tinyml_fpga.
L = tf.keras.layers
model = tf.keras.Sequential([
    tf.keras.Input((WIN, CH)),
    L.Conv1D(8, 5, strides=2, activation="relu"),
    L.MaxPooling1D(2),
    L.Conv1D(16, 3, strides=2, activation="relu"),
    L.MaxPooling1D(2),
    L.Flatten(),
    L.Dense(16, activation="relu"),
    L.Dense(len(CLASSES)),
    L.Softmax(),
])
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
# Random rotation of a window: the same rotation for the accel triplet and the gyro triplet (any board orientation).
def rotate(w, rng):
    # Random orthogonal matrix from a QR decomposition, forced to a proper rotation.
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    # Rotate both triplets.
    return np.concatenate([w[:, :3] @ q.T, w[:, 3:] @ q.T], axis=1).astype(np.float32)


# Training set: the windows plus rotated copies, so the model does not depend on how the board is held.
rng_aug = np.random.default_rng(2)
Xtr = np.concatenate([X[train]] + [np.array([rotate(w, rng_aug) for w in X[train]]) for _ in range(3)])
ytr = np.tile(y[train], 4)
model.fit(Xtr, ytr, epochs=40, verbose=0)
print("float test accuracy: %.3f" % model.evaluate(X[test], y[test], verbose=0)[1])
# Same test windows, held at a random other orientation.
Xrot = np.array([rotate(w, np.random.default_rng(100 + i)) for i, w in enumerate(X[test])])
print("float test accuracy, rotated: %.3f" % model.evaluate(Xrot, y[test], verbose=0)[1])

# Full int8 export.
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = lambda: ([w[None]] for w in Xtr[::4][:400])
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tfl = os.path.join(HERE, "motion.tflite")
with open(tfl, "wb") as f:
    f.write(conv.convert())

# Convert, check bit-exact against LiteRT, report int8 accuracy and the layer placement.
tmf = os.path.join(HERE, "motion.tmf")
tinyml_fpga.convert(tfl, tmf)
m = tinyml_fpga.load(tmf, backend="sim")
m.summary()
s, z = m.input["scale"], m.input["zp"]
compare(tfl, [np.clip(np.round(x / s) + z, -128, 127).astype(np.int8) for x in X[test][:50]], backend="sim")
print("tinyml_fpga int8 test accuracy: %.3f" % np.mean([m.classify(x)[0] == t for x, t in zip(X[test], y[test])]))
print("wrote", tmf, os.path.getsize(tmf), "bytes")
