"""Calcula el puntaje del reward model para cada (prompt, respuesta) -> data/reward_scores.csv.

Opcional (variante (c) del Modelo 2). Requiere GPU, `torch` y `transformers`.
Modelo: OpenAssistant/reward-model-deberta-v3-base (MIT). Se puede reanudar:
guarda un parcial cada `--save-every` filas y lo continúa si existe.

    python -m src.compute_reward_scores --device cuda --batch-size 16
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import load  # noqa: E402

MODEL = "OpenAssistant/reward-model-deberta-v3-base"
OUT = ROOT / "data" / "reward_scores.csv"


def score(tok, model, prompts, answers, device, batch_size, max_length):
    out = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], answers[i:i + batch_size], truncation="longest_first",
                  max_length=max_length, padding=True, return_tensors="pt").to(device)
        with __import__("torch").no_grad():
            out.append(model(**enc).logits[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--nrows", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    df = load.load_train_clean(nrows=args.nrows)
    done = pd.read_csv(OUT) if OUT.exists() else pd.DataFrame(columns=[cfg.COL_ID, "reward_a", "reward_b"])
    df = df[~df[cfg.COL_ID].isin(done[cfg.COL_ID])]
    print(f"{len(done):,} ya calculadas, {len(df):,} pendientes")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32)
    model.to(args.device).eval()

    chunks = [done]
    for s in range(0, len(df), args.save_every):
        part = df.iloc[s:s + args.save_every]
        p = part["prompt_text"].fillna("").tolist()
        ra = score(tok, model, p, part["resp_a_text"].fillna("").tolist(), args.device, args.batch_size, args.max_length)
        rb = score(tok, model, p, part["resp_b_text"].fillna("").tolist(), args.device, args.batch_size, args.max_length)
        chunks.append(pd.DataFrame({cfg.COL_ID: part[cfg.COL_ID].to_numpy(), "reward_a": ra, "reward_b": rb}))
        pd.concat(chunks, ignore_index=True).to_csv(OUT, index=False)
        print(f"  {s + len(part):,}/{len(df):,}")
    print(f"Guardado en {OUT}")


if __name__ == "__main__":
    main()
