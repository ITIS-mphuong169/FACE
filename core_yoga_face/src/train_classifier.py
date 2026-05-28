"""
train_classifier.py
-------------------
Train a skeleton-based posture/yoga classifier.

Supports:
  - RandomForest (sklearn)
  - MLP (sklearn MLPClassifier)
  - MLP-Torch (simple PyTorch MLP, optional)

Outputs:
  output_dir/
    model.pkl         (sklearn) or model.pt (torch)
    scaler.pkl
    label_encoder.pkl
    metrics.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

# local
sys.path.insert(0, str(Path(__file__).parent))
from load_dataset import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional: PyTorch MLP
# ---------------------------------------------------------------------------
def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


class TorchMLP:
    """Thin wrapper so Torch MLP has sklearn-like interface."""

    def __init__(self, input_dim: int, hidden: tuple, n_classes: int,
                 lr: float = 1e-3, epochs: int = 100, batch_size: int = 64,
                 device: str = "cpu"):
        torch, nn = _try_import_torch()
        if torch is None:
            raise ImportError("PyTorch not installed. Use --model_type mlp or rf.")

        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.classes_ = None

        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.3)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    def fit(self, X: np.ndarray, y: np.ndarray):
        import torch
        self.classes_ = np.unique(y)
        X_t = torch.FloatTensor(X).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        dataset = torch.utils.data.TensorDataset(X_t, y_t)
        loader  = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                logger.info(f"  Epoch {epoch+1}/{self.epochs}  loss={total_loss/len(loader):.4f}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.FloatTensor(X).to(self.device))
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def save(self, path: Path):
        import torch
        torch.save(self.net.state_dict(), path)
        logger.info(f"Torch model saved → {path}")

    @classmethod
    def load(cls, path: Path, input_dim: int, hidden: tuple, n_classes: int):
        import torch
        obj = cls(input_dim, hidden, n_classes)
        obj.net.load_state_dict(torch.load(path, map_location="cpu"))
        obj.net.eval()
        return obj


# ---------------------------------------------------------------------------
# Build model
# ---------------------------------------------------------------------------
def build_model(model_type: str, input_dim: int, n_classes: int):
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )
    elif model_type == "mlp_torch":
        return TorchMLP(
            input_dim=input_dim,
            hidden=(256, 128, 64),
            n_classes=n_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
def train(
    train_csv: str,
    test_csv: str,
    model_type: str = "mlp",
    output_dir: str = "outputs/model",
    label_col: str = "label",
    use_confidence: bool = True,
    scale_mode: str = "torso",
    center: bool = True,
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────
    X_train, y_train_raw, df_train = load_dataset(
        train_csv,
        use_confidence=use_confidence,
        center=center,
        scale_mode=scale_mode,
        label_col=label_col,
    )
    X_test, y_test_raw, df_test = load_dataset(
        test_csv,
        use_confidence=use_confidence,
        center=center,
        scale_mode=scale_mode,
        label_col=label_col,
    )

    # ── Encode labels ──────────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw)
    y_test  = le.transform(y_test_raw)
    n_classes = len(le.classes_)
    logger.info(f"Classes: {le.classes_}")

    # ── Scale features ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Build & train ──────────────────────────────────────────────────────
    logger.info(f"Training {model_type.upper()} model ...")
    model = build_model(model_type, X_train_s.shape[1], n_classes)
    model.fit(X_train_s, y_train)

    # ── Evaluate ───────────────────────────────────────────────────────────
    y_pred = model.predict(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_])
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {acc:.4f}")
    print(f"{'='*50}")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # ── Save ───────────────────────────────────────────────────────────────
    if model_type == "mlp_torch":
        model_path = out / "model.pt"
        model.save(model_path)
        # Also save architecture info
        arch = {"input_dim": X_train_s.shape[1], "n_classes": n_classes, "hidden": [256, 128, 64]}
        (out / "model_arch.json").write_text(json.dumps(arch))
    else:
        model_path = out / "model.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved → {model_path}")

    scaler_path = out / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    le_path = out / "label_encoder.pkl"
    joblib.dump(le, le_path)

    metrics = {
        "accuracy": float(acc),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "model_type": model_type,
        "classes": [str(c) for c in le.classes_],
        "confusion_matrix": cm.tolist(),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    logger.info(f"All outputs saved to {out}")

    # Save config for downstream scripts
    config = {
        "model_type": model_type,
        "use_confidence": use_confidence,
        "scale_mode": scale_mode,
        "center": center,
        "label_col": label_col,
        "n_features": int(X_train_s.shape[1]),
        "n_classes": n_classes,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    return model, scaler, le


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train skeleton classifier")
    parser.add_argument("--train_csv",   required=True)
    parser.add_argument("--test_csv",    required=True)
    parser.add_argument("--model_type",  default="mlp",
                        choices=["mlp", "rf", "mlp_torch"])
    parser.add_argument("--output_dir",  default="outputs/model")
    parser.add_argument("--label_col",   default="label")
    parser.add_argument("--no_confidence", action="store_true")
    parser.add_argument("--scale_mode",  default="torso",
                        choices=["torso", "bbox", "none"])
    parser.add_argument("--no_center",   action="store_true")
    args = parser.parse_args()

    train(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        model_type=args.model_type,
        output_dir=args.output_dir,
        label_col=args.label_col,
        use_confidence=not args.no_confidence,
        scale_mode=None if args.scale_mode == "none" else args.scale_mode,
        center=not args.no_center,
    )
