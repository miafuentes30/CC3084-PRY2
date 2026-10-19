"""Fine-tuning real de DeBERTa-v3 con una cabeza siamés simétrica.

El modelo base se descarga desde Hugging Face la primera vez que se ejecuta.
El entrenamiento completo usa las particiones oficiales y guarda un checkpoint
compatible con la aplicación en ``models/deberta_checkpoint``.

Ejemplo rápido para validar el pipeline:
    python -m src.run_deberta --max-train-rows 512 --max-val-rows 256 --max-test-rows 256
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg, evaluation, load  # noqa: E402

SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"
CHECKPOINT_DIR = ROOT / "models" / "deberta_checkpoint"
METRIC_DIR = ROOT / "results" / "metrics"
PRED_DIR = ROOT / "results" / "preds"
FIG_DIR = ROOT / "docs" / "figures"


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BattleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length: int):
        self.prompt = frame.get("prompt", frame.get("prompt_text")).fillna("").astype(str).tolist()
        self.a = frame.get("response_a", frame.get("resp_a_text")).fillna("").astype(str).tolist()
        self.b = frame.get("response_b", frame.get("resp_b_text")).fillna("").astype(str).tolist()
        self.labels = frame["winner"].astype(int).to_numpy()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded_a = self.tokenizer(
            self.prompt[index], self.a[index], max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        encoded_b = self.tokenizer(
            self.prompt[index], self.b[index], max_length=self.max_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids_a": encoded_a["input_ids"].squeeze(0),
            "attention_mask_a": encoded_a["attention_mask"].squeeze(0),
            "input_ids_b": encoded_b["input_ids"].squeeze(0),
            "attention_mask_b": encoded_b["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[index], dtype=torch.long),
        }


class SiameseRewardClassifier(nn.Module):
    """Dos ramas con pesos compartidos y logits invariantes ante A/B swap."""

    def __init__(self, encoder: nn.Module, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.reward = nn.Linear(hidden_size, 1)
        self.log_tau = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))

    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = output.last_hidden_state[:, 0]
        return self.reward(pooled).squeeze(-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        score_a = self.score(batch["input_ids_a"], batch["attention_mask_a"])
        score_b = self.score(batch["input_ids_b"], batch["attention_mask_b"])
        delta = score_a - score_b
        tau = torch.exp(self.log_tau)
        return torch.stack((delta, -delta, tau - torch.abs(delta)), dim=1)


def _move(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


@torch.no_grad()
def predict(model: SiameseRewardClassifier, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities, labels = [], []
    for batch in loader:
        batch = _move(batch, device)
        probabilities.append(torch.softmax(model(batch), dim=1).cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return np.concatenate(labels), np.concatenate(probabilities)


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = np.argmax(probabilities, axis=1)
    report = classification_report(y_true, predicted, labels=[0, 1, 2], output_dict=True, zero_division=0)
    return {
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1, 2])), 6),
        "accuracy": round(float(accuracy_score(y_true, predicted)), 6),
        "f1_macro": round(float(f1_score(y_true, predicted, average="macro")), 6),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1, 2]).tolist(),
        "per_class": {
            cfg.LABEL_NAMES[i]: {
                "precision": round(float(report[str(i)]["precision"]), 6),
                "recall": round(float(report[str(i)]["recall"]), 6),
                "f1": round(float(report[str(i)]["f1-score"]), 6),
            }
            for i in range(3)
        },
    }


def _limit(frame: pd.DataFrame, limit: int | None) -> pd.DataFrame:
    return frame if limit is None else frame.head(limit).reset_index(drop=True)


def train(args: argparse.Namespace) -> dict:
    from transformers import AutoModel, AutoTokenizer

    seed_everything(args.seed)
    started = time.perf_counter()
    cuda_supported = False
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        supported_arches = {int(arch.removeprefix("sm_")) for arch in torch.cuda.get_arch_list() if arch.startswith("sm_")}
        cuda_supported = capability[0] * 10 + capability[1] in supported_arches
        if not cuda_supported:
            print(f"CUDA no compatible con esta instalación (capacidad {capability}); se usará CPU.")
    device = torch.device("cuda" if cuda_supported else "cpu")
    print(f"Dispositivo: {device}. Modelo base: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    encoder = AutoModel.from_pretrained(args.model_name)
    encoder = encoder.float()
    model = SiameseRewardClassifier(encoder, encoder.config.hidden_size).to(device)

    train_df = _limit(load.load_split("train"), args.max_train_rows)
    val_df = _limit(load.load_split("val"), args.max_val_rows)
    test_df = _limit(load.load_split("test"), args.max_test_rows)
    train_loader = DataLoader(BattleDataset(train_df, tokenizer, args.max_length), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(BattleDataset(val_df, tokenizer, args.max_length), batch_size=args.batch_size)
    test_loader = DataLoader(BattleDataset(test_df, tokenizer, args.max_length), batch_size=args.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(batch), batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        y_val, p_val = predict(model, val_loader, device)
        val_loss = log_loss(y_val, p_val, labels=[0, 1, 2])
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": float(val_loss)}
        history.append(row)
        print(f"Época {epoch}/{args.epochs}: train_loss={row['train_loss']:.5f} val_loss={row['val_loss']:.5f}")

    y_val, p_val = predict(model, val_loader, device)
    y_test, p_test = predict(model, test_loader, device)
    val_metrics = metrics(y_val, p_val)
    test_metrics = metrics(y_test, p_test)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(CHECKPOINT_DIR)
    tokenizer.save_pretrained(CHECKPOINT_DIR)
    torch.save({"reward": model.reward.state_dict(), "log_tau": model.log_tau.detach().cpu()}, CHECKPOINT_DIR / "head.pt")

    PRED_DIR.mkdir(parents=True, exist_ok=True)
    evaluation.save_predictions_csv(val_df[cfg.COL_ID], p_val, PRED_DIR / "deberta_val.csv")
    evaluation.save_predictions_csv(test_df[cfg.COL_ID], p_test, PRED_DIR / "deberta_test.csv")
    result = {
        "model_id": "deberta",
        "model_name": "DeBERTa-v3-base Siamese Reward Fine-Tuning",
        "base_model": args.model_name,
        "checkpoint": str(CHECKPOINT_DIR.relative_to(ROOT)),
        "device": str(device),
        "seed": args.seed,
        "hyperparameters": vars(args),
        "dataset_sizes": {"train": len(train_df), "validation": len(val_df), "test": len(test_df)},
        "full_run": args.max_train_rows is None and args.max_val_rows is None and args.max_test_rows is None,
        "history": history,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    (METRIC_DIR / "deberta.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result["test_metrics"], indent=2))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-train-rows", type=int)
    parser.add_argument("--max-val-rows", type=int)
    parser.add_argument("--max-test-rows", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
