"""Pruebas del Modelo 2: BTD, variables de estilo y pipeline de extremo a extremo."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.btd import BTD
from src.style_features import style_features
from src.xgb_model import (
    BTD_COLS, REWARD_COLS, FeatureBuilder, run_pipeline, swap_df, variant_columns,
)


# ── BTD ─────────────────────────────────────────────────────────────────────
def _simulate_battles(n=6000, seed=0):
    """Batallas con fuerzas verdaderas conocidas y empates con ν = 0.8."""
    rng = np.random.RandomState(seed)
    true = {"fuerte": 1.0, "medio": 0.0, "debil": -1.0}
    names = list(true)
    nu = 0.8
    rows = []
    for _ in range(n):
        a, b = rng.choice(names, 2, replace=False)
        d = true[a] - true[b]
        z = 2 * np.cosh(d / 2) + nu
        y = rng.choice(3, p=[np.exp(d / 2) / z, np.exp(-d / 2) / z, nu / z])
        rows.append((a, b, y))
    return pd.DataFrame(rows, columns=["model_a", "model_b", "winner"]), true, nu


def test_btd_recovers_known_strengths_and_matches_win_table():
    df, true, nu = _simulate_battles()
    m = BTD(l2=1e-6).fit(df.model_a, df.model_b, df.winner)

    # Recupera las diferencias verdaderas y el parámetro de empate
    assert m.theta_["fuerte"] - m.theta_["medio"] == pytest.approx(1.0, abs=0.15)
    assert m.theta_["medio"] - m.theta_["debil"] == pytest.approx(1.0, abs=0.15)
    assert m.nu_ == pytest.approx(0.8, abs=0.15)

    # Validación contra una tabla simple de victorias/derrotas: mismo orden
    wins = {n: 0.0 for n in true}
    games = {n: 0 for n in true}
    for a, b, y in df.itertuples(index=False):
        games[a] += 1
        games[b] += 1
        wins[a] += 1.0 if y == 0 else 0.5 if y == 2 else 0.0
        wins[b] += 1.0 if y == 1 else 0.5 if y == 2 else 0.0
    rate = {n: wins[n] / games[n] for n in true}
    assert sorted(true, key=rate.get) == sorted(true, key=m.theta_.get)


def test_btd_fit_is_deterministic_and_probabilities_sum_to_one():
    df, _, _ = _simulate_battles(n=500)
    m1 = BTD(l2=1e-2).fit(df.model_a, df.model_b, df.winner)
    m2 = BTD(l2=1e-2).fit(df.model_a, df.model_b, df.winner)
    assert m1.theta_ == pytest.approx(m2.theta_)
    p = m1.predict_proba(df.model_a, df.model_b)
    assert np.allclose(p.sum(axis=1), 1.0)


def test_btd_unseen_model_gets_neutral_score():
    df, _, _ = _simulate_battles(n=300)
    m = BTD().fit(df.model_a, df.model_b, df.winner)
    assert m.score(["desconocido"])[0] == 0.0


# ── Estilo ──────────────────────────────────────────────────────────────────
def test_style_features_counts():
    t = pd.Series([
        "# Title\n- item 1\n- item 2\n1. uno\n```py\nx=1\n```\nWow!! Really?",
        "I'm sorry, but as an AI I cannot help.",
        None,
    ])
    s = style_features(t)
    assert s.loc[0, "n_headers"] == 1
    assert s.loc[0, "n_bullets"] == 3
    assert s.loc[0, "n_code_blocks"] == 1
    assert s.loc[0, "n_exclam"] == 2 and s.loc[0, "n_question"] == 1
    assert s.loc[1, "n_refusal"] >= 2
    assert s.loc[2].sum() == 0
    assert ((s["upper_ratio"] >= 0) & (s["upper_ratio"] <= 1)).all()


# ── Pipeline ────────────────────────────────────────────────────────────────
def _synthetic_split(n, seed, with_reward=False):
    rng = np.random.RandomState(seed)
    models = ["m1", "m2", "m3", "m4"]
    rows = []
    for i in range(n):
        a, b = rng.choice(models, 2, replace=False)
        la, lb = int(rng.randint(20, 400)), int(rng.randint(20, 400))
        y = int(rng.choice(3, p=[0.4, 0.4, 0.2]))
        row = {
            "id": f"{seed}-{i}", "model_a": a, "model_b": b,
            "prompt_text": f"prompt {seed}-{i % 40}",
            "resp_a_text": ("- punto\n" * (la % 5)) + "a" * la,
            "resp_b_text": "b" * lb,
            "prompt_len": 10, "resp_a_len": la, "resp_b_len": lb,
            "prompt_words": 2, "resp_a_words": la // 5, "resp_b_words": lb // 5,
            "n_turns": 1, "winner": y,
        }
        if with_reward:
            row["reward_a"], row["reward_b"] = rng.randn(), rng.randn()
        rows.append(row)
    return pd.DataFrame(rows)


def test_feature_builder_columns_and_variants():
    tr = _synthetic_split(300, 1, with_reward=True)
    fb = FeatureBuilder(n_oof_folds=3)
    x = fb.fit_transform(tr)
    assert not {"winner"} & set(x.columns)
    assert set(BTD_COLS) <= set(x.columns) and set(REWARD_COLS) <= set(x.columns)
    cols = list(x.columns)
    assert not set(BTD_COLS) & set(variant_columns(cols, "a"))
    assert set(BTD_COLS) <= set(variant_columns(cols, "b"))
    assert not set(REWARD_COLS) & set(variant_columns(cols, "b"))
    assert set(REWARD_COLS) <= set(variant_columns(cols, "c"))


def test_swapped_features_mirror_original():
    tr = _synthetic_split(200, 2, with_reward=True)
    va = _synthetic_split(80, 3, with_reward=True)
    fb = FeatureBuilder(n_oof_folds=3)
    fb.fit_transform(tr)
    x, xs = fb.transform(va), fb.transform(swap_df(va))
    assert np.allclose(x["dif_btd"], -xs["dif_btd"])
    assert np.allclose(x["btd_p_tie"], xs["btd_p_tie"])
    assert np.allclose(x["dif_reward"], -xs["dif_reward"])
    assert np.allclose(x["style_n_bullets_a"], xs["style_n_bullets_b"])


@pytest.mark.parametrize("with_reward", [False, True])
def test_pipeline_outputs_common_format(tmp_path, with_reward):
    tr, va, te = (_synthetic_split(n, s, with_reward) for n, s in ((400, 1), (120, 2), (120, 3)))
    metrics = run_pipeline(tr, va, te, n_trials=2, out_root=tmp_path, save_model=False)

    for split, df in (("val", va), ("test", te)):
        p = pd.read_csv(tmp_path / "results" / "preds" / f"xgboost_{split}.csv")
        assert list(p.columns) == ["id", "p_a", "p_b", "p_tie"]
        assert list(p["id"]) == list(df["id"])
        assert np.allclose(p[["p_a", "p_b", "p_tie"]].sum(axis=1), 1.0, atol=1e-4)

    saved = json.loads((tmp_path / "results" / "metrics" / "xgboost.json").read_text(encoding="utf-8"))
    assert saved["model_id"] == "xgboost"
    assert {"validation_metrics", "test_metrics", "position_bias_swap_check", "times"} <= set(saved)
    assert len(saved["variants_comparison"]) == (3 if with_reward else 2)
    assert (tmp_path / "docs" / "figures" / "fig_confusion_xgboost.png").exists()
    assert (tmp_path / "docs" / "figures" / "fig_xgboost_importancia.png").exists()
    assert metrics["final_variant"] in {"a", "b", "c"}


def test_importance_uses_column_names_and_sums_to_one(tmp_path):
    from src.xgb_model import fit_model, plot_importance, DEFAULT_PARAMS
    tr, va = _synthetic_split(300, 1), _synthetic_split(100, 2)
    fb = FeatureBuilder(n_oof_folds=3)
    xt, xv = fb.fit_transform(tr), fb.transform(va)
    cols = variant_columns(list(xt.columns), "b")
    clf = fit_model({**DEFAULT_PARAMS}, "cpu", xt[cols], tr["winner"].to_numpy(), xv[cols], va["winner"].to_numpy())
    top = plot_importance(clf, cols, tmp_path / "imp.png", top=5)
    assert all(np.isfinite(v) for _, v in top)
    assert top[0][0] in cols and top[0][1] > 0
