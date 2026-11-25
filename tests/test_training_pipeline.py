import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import EXPECTED_COLUMNS, MODEL_FEATURE_COLS, TARGET_COLS, VAL_HOURS
from src.models import dataset, explain, trainers
from src.pipelines import run_training

HOURS = 1400


def make_frame(hours=HOURS, drop=()):
    stamps = pd.date_range("2024-01-01", periods=hours, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(rng.normal(size=(hours, len(EXPECTED_COLUMNS))), columns=EXPECTED_COLUMNS)
    frame["timestamp_utc"] = stamps
    frame["source_used"] = "open_meteo"
    frame["is_imputed"] = False
    frame["is_day"] = (stamps.hour.isin(range(6, 18))).astype("int32")
    frame["temperature_2m"] = np.arange(hours, dtype=float)
    frame["wind_direction_10m"] = rng.uniform(0, 360, hours)
    frame["wind_dir_sin"] = np.sin(np.deg2rad(frame["wind_direction_10m"]))
    frame["wind_dir_cos"] = np.cos(np.deg2rad(frame["wind_direction_10m"]))
    for col in ("us_aqi", "precipitation_mm", "uv_index", "wind_speed_10m"):
        frame[col] = frame[col].abs()
    frame["precipitation_probability"] = rng.uniform(0, 100, hours)
    frame["relative_humidity_2m"] = rng.uniform(0, 100, hours)
    return frame.drop(index=list(drop)).reset_index(drop=True)


def test_feature_and_target_lists():
    assert len(TARGET_COLS) == 9
    assert len(MODEL_FEATURE_COLS) == 39
    for col in ("timestamp_utc", "source_used", "is_imputed", "wind_direction_10m"):
        assert col not in MODEL_FEATURE_COLS


def test_normalize_sorts_and_dedupes():
    frame = make_frame(50)
    shuffled = pd.concat([frame, frame.iloc[:5]]).sample(frac=1, random_state=1)
    shuffled["timestamp_utc"] = shuffled["timestamp_utc"].dt.tz_localize(None)
    out = dataset.normalize(shuffled)
    assert out["timestamp_utc"].is_monotonic_increasing
    assert out["timestamp_utc"].is_unique
    assert str(out["timestamp_utc"].dt.tz) == "UTC"
    assert len(out) == 50


def test_self_join_pairs_with_value_24h_ahead():
    frame = make_frame()
    paired = dataset.build_paired(frame)
    assert len(paired) == HOURS - 24
    assert (paired["target_temperature_2m"] - paired["temperature_2m"] == 24).all()
    assert paired["timestamp_utc"].max() == frame["timestamp_utc"].max() - pd.Timedelta(hours=24)


def test_self_join_survives_missing_hour():
    frame = make_frame(drop=[500])
    paired = dataset.build_paired(frame)
    stamps = paired["timestamp_utc"]
    base = frame["timestamp_utc"].iloc[0]
    assert base + pd.Timedelta(hours=476) not in set(stamps)
    assert base + pd.Timedelta(hours=500) not in set(stamps)
    assert (paired["target_temperature_2m"] - paired["temperature_2m"] == 24).all()
    row = paired[stamps == base + pd.Timedelta(hours=600)].iloc[0]
    assert row["target_temperature_2m"] == 624


def test_embargo_split():
    paired = dataset.build_paired(make_frame())
    train, val = dataset.split_embargo(paired)
    assert len(val) == VAL_HOURS
    assert train["timestamp_utc"].max() <= val["timestamp_utc"].min() - pd.Timedelta(hours=24)
    assert train["timestamp_utc"].max() == val["timestamp_utc"].min() - pd.Timedelta(hours=24)


def test_lstm_builds_all_validation_sequences_from_history():
    frame = make_frame()
    _, val = dataset.split_embargo(dataset.build_paired(frame))
    x, kept = dataset.lstm_sequences(frame, val["timestamp_utc"])
    assert x.shape == (VAL_HOURS, 24, len(MODEL_FEATURE_COLS))
    assert list(kept) == list(val["timestamp_utc"])
    first = frame.set_index("timestamp_utc").loc[val["timestamp_utc"].iloc[0] - pd.Timedelta(hours=23)]
    np.testing.assert_allclose(x[0, 0], first[MODEL_FEATURE_COLS].to_numpy(dtype=float))


def test_lstm_sequences_skip_non_contiguous_windows():
    frame = make_frame(300, drop=[100])
    stamps = frame["timestamp_utc"].iloc[50:200]
    _, kept = dataset.lstm_sequences(frame, stamps)
    gap = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(hours=100)
    assert all(not (gap <= t <= gap + pd.Timedelta(hours=23)) for t in kept)
    assert len(kept) == len(stamps) - 23


def test_lstm_internal_split_keeps_embargo_and_untouched_size():
    train, _ = dataset.split_embargo(dataset.build_paired(make_frame()))
    fit, stop = trainers.lstm_split(train)
    assert len(stop) == 336
    assert fit["timestamp_utc"].max() <= stop["timestamp_utc"].min() - pd.Timedelta(hours=24)


def test_naive_uses_observed_value_not_lag_column():
    _, val = dataset.split_embargo(dataset.build_paired(make_frame()))
    preds = trainers.naive_predict(val)
    pd.testing.assert_frame_equal(preds, val[TARGET_COLS].reset_index(drop=True))
    assert not np.allclose(preds["us_aqi"], val["us_aqi_lag_24h"])


def test_circular_error_uses_shortest_arc():
    mae, rmse = trainers.circular_errors(np.array([359.0]), np.array([1.0]))
    assert mae == 2.0
    assert rmse == 2.0


def test_dry_season_improvement_is_zero_not_nan():
    assert trainers.rmse_improvement_pct(0.0, 0.0) == 0.0
    assert trainers.rmse_improvement_pct(0.0, 0.3) == 0.0
    assert trainers.rmse_improvement_pct(2.0, 1.0) == pytest.approx(50.0)


def test_evaluate_reports_every_target_and_wind_direction():
    _, val = dataset.split_embargo(dataset.build_paired(make_frame()))
    truth = val[[f"target_{c}" for c in TARGET_COLS]].rename(columns=lambda c: c.removeprefix("target_"))
    report = trainers.evaluate(val, truth.reset_index(drop=True))
    for target in TARGET_COLS:
        assert report[target]["rmse"] == pytest.approx(0.0, abs=1e-9)
        assert set(report[target]) == {"rmse", "mae", "r2"}
    assert report["wind_direction_10m"]["circular_mae"] == pytest.approx(0.0, abs=1e-6)
    json.dumps(report, allow_nan=False)


def test_clip_predictions_applies_physical_bounds():
    preds = pd.DataFrame(
        {
            "us_aqi": [-5.0, 10.0],
            "temperature_2m": [-3.0, 40.0],
            "precipitation_mm": [-1.0, 2.0],
            "precipitation_probability": [-4.0, 140.0],
            "relative_humidity_2m": [-4.0, 140.0],
            "uv_index": [-2.0, 6.0],
            "wind_speed_10m": [-1.0, 5.0],
            "wind_dir_sin": [-2.0, 2.0],
            "wind_dir_cos": [-2.0, 2.0],
        }
    )
    out = dataset.clip_predictions(preds, pd.Series([1, 0]))
    assert out["us_aqi"].tolist() == [0.0, 10.0]
    assert out["precipitation_mm"].tolist() == [0.0, 2.0]
    assert out["wind_speed_10m"].tolist() == [0.0, 5.0]
    assert out["precipitation_probability"].tolist() == [0.0, 100.0]
    assert out["relative_humidity_2m"].tolist() == [0.0, 100.0]
    assert out["wind_dir_sin"].tolist() == [-1.0, 1.0]
    assert out["uv_index"].tolist() == [0.0, 0.0]
    assert out["temperature_2m"].tolist() == [-3.0, 40.0]


def test_direction_correlation_is_zero_for_constant_inputs():
    constant = np.zeros(50)
    varying = np.arange(50, dtype=float)
    assert explain.direction_correlation(constant, varying) == 0.0
    assert explain.direction_correlation(varying, constant) == 0.0
    assert explain.direction_correlation(varying, varying) == pytest.approx(1.0)


def test_shap_summary_is_finite_for_dry_season_target():
    frame = make_frame(1000)
    frame["precipitation_mm"] = 0.0
    train, val = dataset.split_embargo(dataset.build_paired(frame))
    models = trainers.train_random_forest(train, n_estimators=5, max_depth=3)
    summary = explain.shap_summary(models, val)
    assert set(summary) == set(TARGET_COLS)
    assert all(len(rows) == 10 for rows in summary.values())
    assert all(row["direction_corr"] == 0.0 for row in summary["precipitation_mm"])
    json.dumps(summary, allow_nan=False)


def test_promotion_skips_when_nothing_improves():
    new = {t: 1.0 for t in TARGET_COLS}
    prod = {t: 1.0 for t in TARGET_COLS}
    assert run_training.promoted_targets(new, prod) == []
    assert run_training.promoted_targets(new, None) == list(TARGET_COLS)
    prod["us_aqi"] = 2.0
    assert run_training.promoted_targets(new, prod) == ["us_aqi"]


def test_promotion_ignores_floating_point_jitter():
    prod = {t: 1.0 for t in TARGET_COLS}
    jitter = {t: 1.0 - 1e-15 for t in TARGET_COLS}
    assert run_training.promoted_targets(jitter, prod) == []
    gain = {**jitter, "us_aqi": 1.0 - 1e-3}
    assert run_training.promoted_targets(gain, prod) == ["us_aqi"]


def lstm_checkpoint_file(path, seed):
    torch.manual_seed(seed)
    checkpoint = trainers.init_lstm_checkpoint()
    trainers.save_lstm_checkpoint(path, checkpoint)


def test_bundle_keeps_versioned_lstm_files_side_by_side(tmp_path):
    prod_dir = tmp_path / "prod"
    new_dir = tmp_path / "new"
    for directory in (prod_dir, new_dir):
        directory.mkdir()
    lstm_checkpoint_file(prod_dir / "lstm_v1.pt", 1)
    lstm_checkpoint_file(new_dir / "lstm_v2.pt", 2)
    prod_manifest = {
        t: {"family": "lstm", "artifact": "lstm_v1.pt", "output_idx": i, "val_rmse": 9.0}
        for i, t in enumerate(TARGET_COLS)
    }
    candidates = {
        "us_aqi": {"family": "lstm", "artifact": "lstm_v2.pt", "output_idx": 0, "val_rmse": 1.0}
    }
    prod_rmse = {t: 3.0 for t in TARGET_COLS}
    bundle_dir = tmp_path / "bundle"

    targets = run_training.assemble_bundle(
        bundle_dir, candidates, new_dir, prod_manifest, prod_dir, prod_rmse, ["us_aqi"]
    )

    assert targets["us_aqi"]["artifact"] == "lstm_v2.pt"
    assert targets["us_aqi"]["val_rmse"] == 1.0
    assert targets["temperature_2m"]["artifact"] == "lstm_v1.pt"
    assert targets["temperature_2m"]["val_rmse"] == 3.0
    assert sorted(p.name for p in (bundle_dir / "models").iterdir()) == ["lstm_v1.pt", "lstm_v2.pt"]
    for name in ("lstm_v1.pt", "lstm_v2.pt"):
        loaded = torch.load(bundle_dir / "models" / name, weights_only=True)
        assert {"state_dict", "x_mean", "x_scale", "y_mean", "y_scale"} <= set(loaded)


def test_bundle_prunes_unreferenced_models(tmp_path):
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    lstm_checkpoint_file(new_dir / "lstm_v1.pt", 1)
    (new_dir / "rf_us_aqi.joblib").write_bytes(b"x")
    candidates = {
        t: {"family": "lstm", "artifact": "lstm_v1.pt", "output_idx": i, "val_rmse": 1.0}
        for i, t in enumerate(TARGET_COLS)
    }
    bundle_dir = tmp_path / "bundle"
    run_training.assemble_bundle(bundle_dir, candidates, new_dir, None, None, None, list(TARGET_COLS))
    assert [p.name for p in (bundle_dir / "models").iterdir()] == ["lstm_v1.pt"]


def test_lstm_trains_saves_and_predicts_every_validation_row(tmp_path):
    frame = make_frame()
    train, val = dataset.split_embargo(dataset.build_paired(frame))
    checkpoint = trainers.train_lstm(train, frame, max_epochs=1)
    path = tmp_path / "lstm_v1.pt"
    trainers.save_lstm_checkpoint(path, checkpoint)
    loaded = trainers.load_lstm_checkpoint(path)
    preds = trainers.lstm_predict(loaded, frame, val)
    assert len(preds) == VAL_HOURS
    assert list(preds.columns) == TARGET_COLS
    assert np.isfinite(preds.to_numpy()).all()
