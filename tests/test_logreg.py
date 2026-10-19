"""Pruebas rápidas del pipeline multinomial y sus contratos de integración."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src import features
from src.logreg_model import (
    CLASS_ORDER,
    FORBIDDEN_FEATURES,
    build_pipeline,
    make_cv_splits,
    predict_proba,
    predict_single,
    prepare_inference_row,
    prepare_xy,
    swap_consistency,
    validate_prediction_frame,
)


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    rows = []
    for group in range(12):
        for winner in CLASS_ORDER:
            a_len = 80 + group * 5 + winner * 7
            b_len = 95 + group * 3 - winner * 4
            rows.append(
                {
                    "id": f"{group}-{winner}",
                    "model_a": ["alpha", "beta", "gamma"][group % 3],
                    "model_b": ["gamma", "alpha", "beta"][group % 3],
                    "prompt_text": f"prompt compartido {group}",
                    "resp_a_text": "a" * a_len,
                    "resp_b_text": "b" * b_len,
                    "prompt_len": 20 + group,
                    "resp_a_len": a_len,
                    "resp_b_len": b_len,
                    "prompt_words": 3 + group % 4,
                    "resp_a_words": 10 + group + winner,
                    "resp_b_words": 12 + group - winner,
                    "n_turns": 1 + group % 2,
                    "winner": int(winner),
                    "winner_model_a": int(winner == 0),
                    "winner_model_b": int(winner == 1),
                    "winner_tie": int(winner == 2),
                }
            )
    return pd.DataFrame(rows)


def fitted_pipeline(sample_df: pd.DataFrame):
    x, y = prepare_xy(sample_df)
    model = build_pipeline(include_model_identity=True, c_value=0.1, max_iter=300)
    model.fit(x, y)
    return model


def test_no_label_leakage(sample_df: pd.DataFrame) -> None:
    x, _ = prepare_xy(sample_df)
    assert not FORBIDDEN_FEATURES.intersection(x.columns)
    assert set(features.COLS_NUM).issubset(x.columns)


def test_preprocessing_is_deterministic_and_unknown_models_work(sample_df: pd.DataFrame) -> None:
    model = fitted_pipeline(sample_df)
    first = predict_proba(model, sample_df)
    second = predict_proba(model, sample_df.copy())
    np.testing.assert_allclose(first, second)

    unknown = sample_df.iloc[[0]].copy()
    unknown[["model_a", "model_b"]] = ["modelo-nuevo-a", "modelo-nuevo-b"]
    probability = predict_proba(model, unknown)
    assert probability.shape == (1, 3)
    np.testing.assert_allclose(probability.sum(axis=1), 1.0)


def test_grouped_cv_never_shares_prompts(sample_df: pd.DataFrame) -> None:
    folds = make_cv_splits(sample_df, n_splits=3)
    for train_idx, validation_idx in folds:
        train_prompts = set(sample_df.iloc[train_idx]["prompt_text"])
        validation_prompts = set(sample_df.iloc[validation_idx]["prompt_text"])
        assert train_prompts.isdisjoint(validation_prompts)


def test_probability_order_and_sum(sample_df: pd.DataFrame) -> None:
    model = fitted_pipeline(sample_df)
    probabilities = predict_proba(model, sample_df)
    assert model.named_steps["classifier"].classes_.tolist() == [0, 1, 2]
    assert probabilities.shape == (len(sample_df), 3)
    assert (probabilities >= 0).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_prediction_ids_and_csv_contract(sample_df: pd.DataFrame) -> None:
    model = fitted_pipeline(sample_df)
    probabilities = predict_proba(model, sample_df)
    frame = pd.DataFrame(
        {
            "id": sample_df["id"],
            "p_a": probabilities[:, 0],
            "p_b": probabilities[:, 1],
            "p_tie": probabilities[:, 2],
        }
    )
    validate_prediction_frame(frame, sample_df["id"])
    with pytest.raises(AssertionError, match="IDs"):
        validate_prediction_frame(frame.iloc[::-1].reset_index(drop=True), sample_df["id"])


def test_pipeline_serialization_and_single_inference(sample_df: pd.DataFrame, tmp_path: Path) -> None:
    model = fitted_pipeline(sample_df)
    path = tmp_path / "pipeline.joblib"
    joblib.dump(model, path)
    recovered = joblib.load(path)
    np.testing.assert_allclose(
        predict_proba(model, sample_df), predict_proba(recovered, sample_df)
    )

    prepared = prepare_inference_row(
        ["hola", "segunda pregunta"],
        ["respuesta A", None],
        ["respuesta B", "continuación"],
        "modelo-no-visto-a",
        "modelo-no-visto-b",
    )
    assert prepared.loc[0, "n_turns"] == 2
    result = predict_single(
        path,
        "hola",
        "respuesta A",
        "respuesta B",
        "modelo-no-visto-a",
        "modelo-no-visto-b",
    )
    assert list(result) == ["p_a", "p_b", "p_tie"]
    assert sum(result.values()) == pytest.approx(1.0)


def test_swap_ab_diagnostic_runs(sample_df: pd.DataFrame) -> None:
    model = fitted_pipeline(sample_df)
    swapped_twice = features.swap_ab(features.swap_ab(sample_df))
    pd.testing.assert_frame_equal(swapped_twice, sample_df)
    diagnostic = swap_consistency(model, sample_df)
    assert diagnostic["n_observations"] == len(sample_df)
    assert 0 <= diagnostic["aggregate_mae"] <= 1
    assert 0 <= diagnostic["consistency_score_1_minus_mae"] <= 1


def test_generated_artifacts_when_present() -> None:
    """Valida los artefactos reales sin hacer obligatorio versionar el binario."""
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "results" / "metrics" / "logreg.json"
    if not metrics_path.exists():
        pytest.skip("Los artefactos reales todavía no se han generado.")

    with metrics_path.open(encoding="utf-8") as file:
        metrics = json.load(file)
    assert metrics["probability_order"] == ["p_a", "p_b", "p_tie"]
    assert metrics["selection"]["selected_before_test_evaluation"] is True
    assert metrics["versions"]["scikit_learn"]

    splits = pd.read_csv(root / "data" / "splits.csv")
    for split_name, file_name in [("val", "logreg_val.csv"), ("test", "logreg_test.csv")]:
        predictions = pd.read_csv(root / "results" / "preds" / file_name)
        expected_ids = splits.loc[splits["split"] == split_name, "id"].reset_index(drop=True)
        validate_prediction_frame(predictions, expected_ids)

    pipeline_path = root / metrics["artifacts"]["pipeline"]
    if pipeline_path.exists():
        pipeline = joblib.load(pipeline_path)
        assert pipeline.named_steps["classifier"].classes_.tolist() == [0, 1, 2]
