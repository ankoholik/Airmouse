"""
Train gesture MLP (PyTorch), then export ONNX and OpenVINO IR.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from airmouse.paths import DATASETS_DIR
from airmouse.ml.dataset import create_dataloaders_from_split_csv
from airmouse.ml.model import model
from airmouse.paths import model_weights_path, onnx_model_path, openvino_model_path


def save_torch_weights(model_module: nn.Module, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_module.state_dict(), filepath)
    print(f"PyTorch weights saved: {filepath}")


def export_onnx(weights_path: Path, out_path: Path, opset: int = 17) -> None:
    device = "cpu"
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.zeros((1, 63), dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["x"],
        output_names=["logits"],
        dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    print(f"ONNX saved: {out_path}")


def export_openvino(onnx_path: Path, xml_path: Path) -> None:
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX not found for OpenVINO conversion: {onnx_path}")

    try:
        from openvino import convert_model, save_model as ov_save_model
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "OpenVINO is not installed. Install it or run with --skip-export."
        ) from e

    ov_model = convert_model(str(onnx_path))
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    ov_save_model(ov_model, str(xml_path))
    print(f"OpenVINO IR saved: {xml_path}")


def evaluate(model_module: nn.Module, loader, device: str) -> dict:
    model_module.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device).view(-1, 63).float()
            y = y.to(device)
            logits = model_module(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred) if y_true else 0.0
    cm = confusion_matrix(y_true, y_pred).tolist() if y_true else []
    report = classification_report(y_true, y_pred, digits=4) if y_true else ""
    avg_loss = total_loss / max(1, len(y_true))
    return {"loss": avg_loss, "accuracy": acc, "confusion_matrix": cm, "report": report}


def train(
    num_epochs: int = 100,
    learning_rate: float = 0.0005,
    batch_size: int = 64,
    log_dir: Path | None = None,
    device: str = "cpu",
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    splits_dir = DATASETS_DIR / "splits"
    train_csv = splits_dir / "train.csv"
    test_csv = splits_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(
            f"Dataset splits not found. Run scripts/split_dataset.py first. Missing: {train_csv} or {test_csv}"
        )

    train_loader, test_loader, _class_to_idx = create_dataloaders_from_split_csv(
        train_csv=train_csv,
        test_csv=test_csv,
        batch_size=batch_size,
        num_workers=0,
        dataset_root=DATASETS_DIR,
    )

    model.to(device)

    print("=== PyTorch training ===")
    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(num_epochs):
        model.train()
        running = 0.0
        seen = 0
        for i, batch in enumerate(train_loader):
            if batch is None:
                continue
            data, labels = batch
            data = data.to(device).view(-1, 63).float()
            labels = labels.to(device)

            outputs = model(data)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * data.size(0)
            seen += data.size(0)

            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}], Loss: {loss.item():.4f}")

        train_loss = running / max(1, seen)
        metrics = evaluate(model, test_loader, device=device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(metrics["loss"])
        history["test_accuracy"].append(metrics["accuracy"])
        print(
            f"Epoch [{epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} "
            f"test_loss={metrics['loss']:.4f} test_acc={metrics['accuracy']:.4f}"
        )

    final_metrics = evaluate(model, test_loader, device=device)
    print("=== Test metrics ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(final_metrics["report"])

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        # Curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["train_loss"], label="train_loss")
        plt.plot(history["test_loss"], label="test_loss")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["test_accuracy"], label="test_accuracy")
        plt.title("Test accuracy")
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        curves_path = log_dir / "training_curves.png"
        plt.savefig(curves_path, dpi=150)
        plt.close()
        print(f"Training curves saved: {curves_path}")

        # Confusion matrix (as a simple image)
        cm = final_metrics["confusion_matrix"]
        if cm:
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion matrix")
            plt.xlabel("pred")
            plt.ylabel("true")
            plt.colorbar()
            cm_path = log_dir / "confusion_matrix.png"
            plt.tight_layout()
            plt.savefig(cm_path, dpi=150)
            plt.close()
            print(f"Confusion matrix saved: {cm_path}")

            (log_dir / "classification_report.txt").write_text(final_metrics["report"], encoding="utf-8")
            print(f"Classification report saved: {log_dir / 'classification_report.txt'}")

    return {"history": history, "final": final_metrics}


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Train PyTorch model, then export ONNX and OpenVINO IR.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/train"))
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip ONNX and OpenVINO export (only save .pth)",
    )
    args = parser.parse_args()

    train(
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        log_dir=args.log_dir,
        device=args.device,
    )

    weights_path = model_weights_path()
    save_torch_weights(model, weights_path)

    if args.skip_export:
        return

    onnx_path = onnx_model_path()
    print("=== ONNX export ===")
    export_onnx(weights_path, onnx_path, opset=args.opset)

    xml_path = openvino_model_path()
    print("=== OpenVINO IR export ===")
    export_openvino(onnx_path, xml_path)


if __name__ == "__main__":
    main()
