"""Modelo 2: XGBoost + variables de estilo + Bradley–Terry–Davidson (+ reward model opcional).

Variantes (ablación pedida en `docs/especificaciones_modelos.md`):
  (a) base + estilo            (sin BTD)
  (b) base + estilo + BTD
  (c) base + estilo + BTD + reward model   (si existe `data/reward_scores.csv`)

La selección de hiperparámetros y de la variante final usa solo validación;
la partición de prueba se evalúa una única vez, con la variante elegida.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config as cfg  # noqa: E402
from src import evaluation, features, load  # noqa: E402
from src.btd import BTD  # noqa: E402
from src.style_features import STYLE_NAMES, build_style_columns  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 42
MODEL_ID = "xgboost"
F_REWARD = ROOT / "data" / "reward_scores.csv"   # columnas: id, reward_a, reward_b

VARIANTS = {
    "a": "(a) Sin BTD",
    "b": "(b) Con BTD",
    "c": "(c) BTD + reward model",
}
BTD_COLS = ["btd_a", "btd_b", "dif_btd", "btd_p_a", "btd_p_b", "btd_p_tie"]
REWARD_COLS = ["reward_a", "reward_b", "dif_reward"]


# ── Recompensas (opcional) y utilidades de intercambio ──────────────────────
def attach_rewards(df: pd.DataFrame, path: Path = F_REWARD) -> pd.DataFrame:
    """Une `reward_a/reward_b` por `id` si el archivo existe (si no, devuelve `df` igual)."""
    if not Path(path).exists():
        return df
    r = pd.read_csv(path)[[cfg.COL_ID, "reward_a", "reward_b"]]
    return df.merge(r, on=cfg.COL_ID, how="left", validate="one_to_one")


def swap_df(df: pd.DataFrame) -> pd.DataFrame:
    """`features.swap_ab` + intercambio de las recompensas, si están presentes."""
    s = features.swap_ab(df)
    if {"reward_a", "reward_b"} <= set(s.columns):
        s[["reward_a", "reward_b"]] = s[["reward_b", "reward_a"]].to_numpy()
    return s


# ── Construcción de variables ───────────────────────────────────────────────
class FeatureBuilder:
    """Ajusta con entrenamiento (OHE y BTD) y transforma cualquier partición.

    Para las filas de entrenamiento el BTD se estima fuera de muestra (K-fold
    agrupado por prompt): así XGBoost no ve puntajes calculados con la propia fila.
    Val/test usan el BTD ajustado con todo el entrenamiento.
    """

    def __init__(self, n_oof_folds: int = 5, btd_l2: float = 1e-2):
        self.n_oof_folds = n_oof_folds
        self.btd_l2 = btd_l2
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.btd: BTD | None = None
        self.oof_: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    def _fit_btd(self, d: pd.DataFrame) -> BTD:
        return BTD(self.btd_l2).fit(d[cfg.COL_MODEL_A], d[cfg.COL_MODEL_B], d["winner"])

    def _oof_btd(self, train: pd.DataFrame):
        sa, sb, nu = (np.zeros(len(train)) for _ in range(3))
        groups = pd.util.hash_pandas_object(train["prompt_text"], index=False).to_numpy()
        k = min(self.n_oof_folds, len(np.unique(groups)))
        for tr_idx, te_idx in GroupKFold(n_splits=k).split(train, groups=groups):
            m = self._fit_btd(train.iloc[tr_idx])
            te = train.iloc[te_idx]
            sa[te_idx] = m.score(te[cfg.COL_MODEL_A])
            sb[te_idx] = m.score(te[cfg.COL_MODEL_B])
            nu[te_idx] = m.nu_
        return sa, sb, nu

    def _assemble(self, df: pd.DataFrame, sa, sb, nu) -> pd.DataFrame:
        df = df.reset_index(drop=True)
        base = features.build_base_features(df)
        num = base[features.COLS_NUM]
        cat = pd.DataFrame(
            self.ohe.transform(base[features.COLS_CAT]),
            columns=self.ohe.get_feature_names_out(features.COLS_CAT),
        )
        style = build_style_columns(df)

        sa, sb, nu = np.asarray(sa), np.asarray(sb), np.asarray(nu)
        d = sa - sb
        z = 2.0 * np.cosh(d / 2.0) + nu
        btd = pd.DataFrame({
            "btd_a": sa, "btd_b": sb, "dif_btd": d,
            "btd_p_a": np.exp(d / 2.0) / z, "btd_p_b": np.exp(-d / 2.0) / z,
            "btd_p_tie": nu / z,
        })

        parts = [num, cat, style, btd]
        if {"reward_a", "reward_b"} <= set(df.columns):
            rw = df[["reward_a", "reward_b"]].astype(float)
            rw["dif_reward"] = rw["reward_a"] - rw["reward_b"]
            parts.append(rw)
        return pd.concat(parts, axis=1)

    def fit_transform(self, train: pd.DataFrame) -> pd.DataFrame:
        self.ohe.fit(features.build_base_features(train)[features.COLS_CAT])
        self.btd = self._fit_btd(train)
        self.oof_ = self._oof_btd(train)
        return self._assemble(train, *self.oof_)

    def train_swapped(self, train_swapped: pd.DataFrame) -> pd.DataFrame:
        """Variables de entrenamiento con A↔B intercambiados (aumento de datos)."""
        sa, sb, nu = self.oof_
        return self._assemble(train_swapped, sb, sa, nu)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        sa = self.btd.score(df[cfg.COL_MODEL_A])
        sb = self.btd.score(df[cfg.COL_MODEL_B])
        return self._assemble(df, sa, sb, np.full(len(df), self.btd.nu_))


def variant_columns(columns: list[str], variant: str) -> list[str]:
    """Columnas de cada variante; las de BTD/reward se excluyen según corresponda."""
    drop: set[str] = set()
    if variant == "a":
        drop |= set(BTD_COLS)
    if variant in ("a", "b"):
        drop |= set(REWARD_COLS)
    return [c for c in columns if c not in drop]


# ── Entrenamiento ───────────────────────────────────────────────────────────
DEFAULT_PARAMS: dict[str, Any] = {
    "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8,
    "colsample_bytree": 0.8, "min_child_weight": 5, "reg_lambda": 5.0,
}


def make_classifier(params: dict[str, Any], device: str, n_estimators: int = 2000,
                    early_stopping_rounds: int | None = 50) -> xgb.XGBClassifier:
    p = {k: v for k, v in params.items() if k != "augment_swap"}
    return xgb.XGBClassifier(
        objective="multi:softprob", eval_metric="mlogloss", tree_method="hist",
        device=device, n_estimators=n_estimators, random_state=SEED,
        early_stopping_rounds=early_stopping_rounds, **p,
    )


def fit_model(params, device, x_tr, y_tr, x_va, y_va, x_tr_sw=None, y_tr_sw=None):
    """Entrena con early stopping en validación. Si `params['augment_swap']`, duplica el train."""
    if params.get("augment_swap") and x_tr_sw is not None:
        x_tr = pd.concat([x_tr, x_tr_sw[x_tr.columns]], ignore_index=True)
        y_tr = np.concatenate([y_tr, y_tr_sw])
    clf = make_classifier(params, device)
    clf.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    return clf


def tune(variant_data: dict, device: str, n_trials: int) -> tuple[dict, float, list]:
    """Búsqueda con Optuna (TPE, semilla 42) minimizando log loss en validación."""
    default = {**DEFAULT_PARAMS, "augment_swap": False}
    if n_trials <= 0:
        clf = fit_model(default, device, **variant_data)
        return default, float(clf.best_score), []

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 50.0, log=True),
            "augment_swap": trial.suggest_categorical("augment_swap", [False, True]),
        }
        clf = fit_model(params, device, **variant_data)
        return float(clf.best_score)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.enqueue_trial(default)
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params), float(study.best_value), [round(t.value, 5) for t in study.trials]


# ── Figuras ─────────────────────────────────────────────────────────────────
def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_variants(results: dict, path: Path, final: str) -> None:
    keys = list(results)
    ll = [results[k]["val_log_loss"] for k in keys]
    f1 = [results[k]["val_f1_macro"] for k in keys]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    bars = ax.bar(x - 0.18, ll, 0.36, color=["#2A4D8F" if k == final else "#4C72B0" for k in keys],
                  label="Log loss (menor es mejor)")
    ax2 = ax.twinx()
    ax2.bar(x + 0.18, f1, 0.36, color="#55A868", label="F1 macro (mayor es mejor)")
    ax.axhline(1.0986, color="#C44E52", ls="--", label="Línea base trivial (1.0986)")
    for b, v in zip(bars, ll):
        ax.annotate(f"{v:.4f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANTS[k] for k in keys], fontweight="bold")
    ax.set_ylim(min(ll) - 0.03, 1.11)
    ax2.set_ylim(0.2, 0.5)
    ax2.grid(False)
    ax.set_ylabel("Log loss (validación)")
    ax2.set_ylabel("F1 macro (validación)")
    ax.set_title("Modelo 2 (XGBoost): aporte de BTD y reward model", fontweight="bold")
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    _save(fig, path)


def plot_importance(clf: xgb.XGBClassifier, columns: list[str], path: Path, top: int = 20) -> list:
    imp = pd.Series(clf.get_booster().get_score(importance_type="total_gain"))
    # Con DataFrame de entrada, XGBoost nombra las variables por columna (no f0, f1, ...).
    imp = imp.reindex(columns).fillna(0.0)
    imp = (imp / imp.sum()).sort_values(ascending=False)
    t = imp.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=150)
    ax.barh(t.index, t.values, color="#4C72B0")
    ax.set_xlabel("Importancia (ganancia total normalizada)")
    ax.set_title(f"Modelo 2 (XGBoost): {top} variables más importantes", fontweight="bold")
    fig.tight_layout()
    _save(fig, path)
    return [[k, round(float(v), 5)] for k, v in imp.head(top).items()]


def plot_swap(p_orig: np.ndarray, p_swap: np.ndarray, path: Path) -> None:
    idx = np.random.RandomState(SEED).choice(len(p_orig), size=min(200, len(p_orig)), replace=False)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.scatter(p_orig[idx, 0], p_swap[idx, 1], alpha=0.6, s=20, color="#4C72B0", label="P(A|orig) vs P(B|swap)")
    ax.scatter(p_orig[idx, 1], p_swap[idx, 0], alpha=0.6, s=20, color="#DD8452", label="P(B|orig) vs P(A|swap)")
    ax.plot([0, 1], [0, 1], "r--", lw=1.2, label="Simetría ideal (y = x)")
    ax.set_xlabel("Probabilidad en configuración original")
    ax.set_ylabel("Probabilidad en configuración invertida")
    ax.set_title("Modelo 2 (XGBoost): prueba de intercambio A/B", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, path)


# ── Pipeline completo ───────────────────────────────────────────────────────
def run_pipeline(train, val, test, *, n_trials=30, device="cpu", out_root: Path = ROOT,
                 save_model: bool = True) -> dict:
    """Entrena las variantes, elige la mejor en validación y guarda todos los resultados."""
    fig_dir, pred_dir = out_root / "docs" / "figures", out_root / "results" / "preds"
    metric_dir, model_dir = out_root / "results" / "metrics", out_root / "models"
    t_start = time.time()

    has_reward = {"reward_a", "reward_b"} <= set(train.columns)
    y_tr, y_va, y_te = (d["winner"].to_numpy() for d in (train, val, test))
    y_tr_sw = pd.Series(y_tr).map({0: 1, 1: 0, 2: 2}).to_numpy()

    t0 = time.time()
    fb = FeatureBuilder()
    x_tr_all = fb.fit_transform(train)
    x_tr_sw_all = fb.train_swapped(swap_df(train))
    x_va_all, x_te_all = fb.transform(val), fb.transform(test)
    x_va_sw_all = fb.transform(swap_df(val))
    t_features = time.time() - t0
    print(f"Variables construidas en {t_features:.1f}s: {x_tr_all.shape[1]} columnas")

    variants = ["a", "b"] + (["c"] if has_reward else [])
    results: dict[str, dict] = {}
    fitted: dict[str, tuple] = {}
    for v in variants:
        cols = variant_columns(list(x_tr_all.columns), v)
        data = dict(x_tr=x_tr_all[cols], y_tr=y_tr, x_va=x_va_all[cols], y_va=y_va,
                    x_tr_sw=x_tr_sw_all[cols], y_tr_sw=y_tr_sw)
        t0 = time.time()
        params, _, history = tune(data, device, n_trials)
        t_tune = time.time() - t0
        t0 = time.time()
        clf = fit_model(params, device, **data)
        t_fit = time.time() - t0
        t0 = time.time()
        p_va = clf.predict_proba(x_va_all[cols])
        t_inf = time.time() - t0
        m = evaluation.calculate_metrics(y_va, p_va)
        results[v] = {
            "name": VARIANTS[v], "n_features": len(cols), "params": params,
            "best_iteration": int(clf.best_iteration), "val_log_loss": m["log_loss"],
            "val_accuracy": m["accuracy"], "val_f1_macro": m["f1_macro"],
            "tuning_trials": n_trials, "tuning_history": history,
            "tuning_sec": round(t_tune, 2), "train_sec": round(t_fit, 2),
            "val_inference_sec": round(t_inf, 4),
        }
        fitted[v] = (clf, cols, p_va)
        print(f"Variante {VARIANTS[v]}: val log loss = {m['log_loss']:.5f}, F1 = {m['f1_macro']:.4f}")

    final = min(results, key=lambda k: results[k]["val_log_loss"])
    clf, cols, p_va = fitted[final]
    print(f"Variante final (mejor en validación): {VARIANTS[final]}")

    # Prueba (una sola vez) y prueba de intercambio A/B
    t0 = time.time()
    p_te = clf.predict_proba(x_te_all[cols])
    t_inf_te = time.time() - t0
    p_va_sw = clf.predict_proba(x_va_sw_all[cols])
    swap = evaluation.check_ab_swap_bias(p_va, p_va_sw)
    val_m, test_m = evaluation.calculate_metrics(y_va, p_va), evaluation.calculate_metrics(y_te, p_te)
    print(f"Prueba: log loss = {test_m['log_loss']}, acc = {test_m['accuracy']}, F1 = {test_m['f1_macro']}")
    print(f"Swap A/B: diferencia media = {swap['mean_absolute_swap_diff']}")

    plot_variants(results, fig_dir / "fig_xgboost_variantes.png", final)
    top_imp = plot_importance(clf, cols, fig_dir / "fig_xgboost_importancia.png")
    evaluation.plot_and_save_confusion_matrix(
        y_te, p_te.argmax(axis=1), f"XGBoost {VARIANTS[final]} (Test)", fig_dir / "fig_confusion_xgboost.png")
    plot_swap(p_va, p_va_sw, fig_dir / "fig_xgboost_swap.png")

    evaluation.save_predictions_csv(val[cfg.COL_ID], p_va, pred_dir / f"{MODEL_ID}_val.csv")
    evaluation.save_predictions_csv(test[cfg.COL_ID], p_te, pred_dir / f"{MODEL_ID}_test.csv")
    if save_model:
        model_dir.mkdir(parents=True, exist_ok=True)
        clf.save_model(model_dir / "xgboost_final.json")
        joblib.dump({"builder": fb, "clf": clf, "columns": cols, "variant": final},
                    model_dir / "xgboost_bundle.joblib")

    metrics = {
        "model_id": MODEL_ID,
        "model_name": f"XGBoost + estilo + Bradley–Terry–Davidson {VARIANTS[final]}",
        "responsible": "Angel Esteban Esquit Hernández (23221)",
        "seed": SEED,
        "hardware": device,
        "final_variant": final,
        "reward_model_used": bool(has_reward),
        "hyperparameters": results[final]["params"] | {"n_estimators": results[final]["best_iteration"] + 1},
        "features": {"n_features": len(cols), "columns": cols},
        "btd": {"nu": round(fb.btd.nu_, 5), "l2": fb.btd_l2, "oof_folds": fb.n_oof_folds,
                "ranking": [[n, round(s, 4)] for n, s in fb.btd.ranking()]},
        "times": {
            "total_elapsed_sec": round(time.time() - t_start, 2),
            "feature_building_sec": round(t_features, 2),
            "tuning_sec": results[final]["tuning_sec"],
            "training_sec": results[final]["train_sec"],
            "val_inference_sec": results[final]["val_inference_sec"],
            "test_inference_sec": round(t_inf_te, 4),
        },
        "validation_metrics": val_m,
        "test_metrics": test_m,
        "variants_comparison": {VARIANTS[k]: {
            "n_features": r["n_features"], "log_loss": r["val_log_loss"],
            "accuracy": r["val_accuracy"], "f1_macro": r["val_f1_macro"],
            "params": r["params"], "best_iteration": r["best_iteration"],
            "tuning_trials": r["tuning_trials"], "train_time_sec": r["train_sec"],
        } for k, r in results.items()},
        "feature_importance_top": top_imp,
        "position_bias_swap_check": swap,
        "model_drive_link": None,
    }
    evaluation.save_metrics_json(metrics, metric_dir / f"{MODEL_ID}.json")
    return metrics


def export_bundle(root: Path = ROOT) -> Path:
    """Reconstruye `models/xgboost_bundle.joblib` sin reentrenar el modelo.

    El constructor de variables (OHE + BTD) es determinista dado el entrenamiento,
    así que se vuelve a ajustar y se une con `models/xgboost_final.json` y las
    columnas de la variante final guardadas en `results/metrics/xgboost.json`.
    """
    meta = json.loads((root / "results" / "metrics" / f"{MODEL_ID}.json").read_text(encoding="utf-8"))
    fb = FeatureBuilder()
    fb.fit_transform(load.load_split("train"))
    clf = xgb.XGBClassifier()
    clf.load_model(root / "models" / "xgboost_final.json")
    out = root / "models" / "xgboost_bundle.joblib"
    joblib.dump({"builder": fb, "clf": clf, "columns": meta["features"]["columns"],
                 "variant": meta["final_variant"]}, out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Modelo 2: XGBoost + estilo + BTD")
    ap.add_argument("--trials", type=int, default=30, help="Intentos de Optuna por variante (0 = sin búsqueda)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--nrows", type=int, default=None,
                    help="Prueba rápida con N filas por partición (salidas en cache/, no toca results/)")
    ap.add_argument("--export-bundle", action="store_true",
                    help="Solo reconstruir models/xgboost_bundle.joblib para la app (sin entrenar)")
    args = ap.parse_args()

    if args.export_bundle:
        print(f"Paquete exportado: {export_bundle()}")
        return

    train, val, test = (attach_rewards(load.load_split(s)) for s in ("train", "val", "test"))
    if args.nrows:
        train, val, test = (d.head(args.nrows) for d in (train, val, test))
        out = ROOT / "cache" / "smoke_xgboost"
        print(f"MODO RÁPIDO ({args.nrows} filas): salidas en {out}")
        run_pipeline(train, val, test, n_trials=args.trials, device=args.device, out_root=out, save_model=False)
    else:
        run_pipeline(train, val, test, n_trials=args.trials, device=args.device)


if __name__ == "__main__":
    main()
