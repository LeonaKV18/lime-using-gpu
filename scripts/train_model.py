"""
train_model.py
--------------
Trains a logistic regression classifier on a real dataset and exports
model parameters in two formats consumed downstream.

Binary format (model.bin) read by main.cu:
    [int32: D][float32 x D: W][float32: bias][float32 x D: x0][float32 x D: means]

NPZ format (model.npz) read by Python scripts:
    W, bias, x0, means, feature_names, dataset, instance_idx, accuracy

Supported datasets (--dataset):
    breast_cancer   D=30  binary  sklearn built-in
    wine_binary     D=13  binary  class 0 vs {1,2}  sklearn built-in

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --dataset=wine_binary --instance=3
    python scripts/train_model.py --dataset=breast_cancer --out-dir=models
"""

import argparse
import os
import struct
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


SUPPORTED_DATASETS = ["breast_cancer", "wine_binary"]


def load_dataset(name):
    """
    Returns X (n_samples, D) float32, y (n_samples,) int, feature_names list[str].
    wine_binary collapses classes 1 and 2 into a single positive class.
    """
    if name == "breast_cancer":
        data = load_breast_cancer()
        return data.data.astype(np.float32), data.target, list(data.feature_names)
    if name == "wine_binary":
        data = load_wine()
        y = (data.target != 0).astype(int)
        return data.data.astype(np.float32), y, list(data.feature_names)
    raise ValueError(f"Unknown dataset '{name}'. Choose from: {SUPPORTED_DATASETS}")


def train_logistic(X_train, y_train):
    clf = LogisticRegression(max_iter=10000, solver="lbfgs", C=1.0)
    clf.fit(X_train, y_train)
    W    = clf.coef_[0].astype(np.float32)
    bias = np.float32(clf.intercept_[0])
    return clf, W, bias


def save_bin(path, D, W, bias, x0, means):
    """
    Writes model.bin in the format expected by load_model_bin() in main.cu.
    Layout: [int32 D][float32*D W][float32 bias][float32*D x0][float32*D means]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<i", D))
        f.write(W.astype(np.float32).tobytes())
        f.write(struct.pack("<f", float(bias)))
        f.write(x0.astype(np.float32).tobytes())
        f.write(means.astype(np.float32).tobytes())


def main():
    parser = argparse.ArgumentParser(description="Train and export logistic regression model")
    parser.add_argument("--dataset",  type=str, default="breast_cancer",
                        choices=SUPPORTED_DATASETS, help="Dataset to train on")
    parser.add_argument("--instance", type=int, default=0,
                        help="Test-set index of the instance to explain (x0)")
    parser.add_argument("--seed",     type=int, default=42,      help="Random seed")
    parser.add_argument("--out-dir",  type=str, default="models", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X, y, feature_names = load_dataset(args.dataset)
    D = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )

    clf, W, bias = train_logistic(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    means = X_train.mean(axis=0).astype(np.float32)

    if args.instance >= len(X_test):
        raise ValueError(
            f"--instance={args.instance} out of range (test set has {len(X_test)} samples)"
        )
    x0 = X_test[args.instance].astype(np.float32)

    bin_path = os.path.join(args.out_dir, f"{args.dataset}.bin")
    npz_path = os.path.join(args.out_dir, f"{args.dataset}.npz")

    save_bin(bin_path, D, W, bias, x0, means)
    np.savez(
        npz_path,
        W=W,
        bias=np.float32(bias),
        x0=x0,
        means=means,
        feature_names=np.array(feature_names),
        dataset=np.array(args.dataset),
        instance_idx=np.array(args.instance),
        accuracy=np.array(acc, dtype=np.float32),
    )

    print(f"Dataset : {args.dataset}  D={D}  train={len(X_train)}  test={len(X_test)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"W       : min={W.min():.4f}  max={W.max():.4f}  mean={W.mean():.4f}")
    print(f"bias    : {bias:.4f}")
    print(f"x0      : min={x0.min():.4f}  max={x0.max():.4f}  mean={x0.mean():.4f}")
    print(f"means   : min={means.min():.4f}  max={means.max():.4f}  mean={means.mean():.4f}")
    print(f"Saved -> {bin_path}")
    print(f"Saved -> {npz_path}")


if __name__ == "__main__":
    main()