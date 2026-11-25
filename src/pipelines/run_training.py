import argparse
import json
import logging
import math
import shutil
import tempfile
import time
from pathlib import Path

import joblib
import pandas as pd

from src.config import MODEL_FEATURE_COLS, MODEL_NAME, TARGET_COLS
from src.models import dataset, explain, trainers
from src.pipelines.run_eda import ARTIFACTS_DIR, compute_eda_summary, write_json

log = logging.getLogger(__name__)

BUNDLE_DIR = Path("model_bundle")
FAMILIES = ("prophet", "random_forest", "lstm")
PROMOTION_EPSILON = 1e-6


def parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m src.pipelines.run_training")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def promoted_targets(new_rmse, prod_rmse):
    if prod_rmse is None:
        return list(TARGET_COLS)
    return [t for t in TARGET_COLS if new_rmse[t] < prod_rmse[t] - PROMOTION_EPSILON]


def best_candidates(reports):
    candidates = {}
    for idx, target in enumerate(TARGET_COLS):
        family = min(FAMILIES, key=lambda f: reports[f][target]["rmse"])
        candidates[target] = {
            "family": family,
            "output_idx": idx if family == "lstm" else None,
            "val_rmse": reports[family][target]["rmse"],
        }
    return candidates


def save_candidate_models(candidates, rf, prophet, lstm, version, out_dir):
    from prophet.serialize import model_to_json

    out_dir.mkdir(parents=True, exist_ok=True)
    for target, entry in candidates.items():
        if entry["family"] == "random_forest":
            entry["artifact"] = f"rf_{target}.joblib"
            joblib.dump(rf[target], out_dir / entry["artifact"])
        elif entry["family"] == "prophet":
            entry["artifact"] = f"prophet_{target}.json"
            (out_dir / entry["artifact"]).write_text(model_to_json(prophet[target]), encoding="utf-8")
        else:
            entry["artifact"] = f"lstm_v{version}.pt"
    if any(e["family"] == "lstm" for e in candidates.values()):
        trainers.save_lstm_checkpoint(out_dir / f"lstm_v{version}.pt", lstm)
    return candidates


def assemble_bundle(bundle_dir, candidates, candidate_dir, prod_targets, prod_dir, prod_rmse, promoted):
    models_dir = Path(bundle_dir) / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    targets = {}
    for target in TARGET_COLS:
        if target in promoted:
            entry, source = dict(candidates[target]), Path(candidate_dir)
        else:
            entry = {**prod_targets[target], "val_rmse": prod_rmse[target]}
            source = Path(prod_dir)
        targets[target] = entry
        shutil.copy2(source / entry["artifact"], models_dir / entry["artifact"])
    return targets


def predict_bundle(models_dir, targets, df, rows_df):
    from prophet.serialize import model_from_json

    models_dir = Path(models_dir)
    columns, lstm_cache = {}, {}
    for target, entry in targets.items():
        path = models_dir / entry["artifact"]
        if entry["family"] == "random_forest":
            columns[target] = joblib.load(path).predict(rows_df[MODEL_FEATURE_COLS])
        elif entry["family"] == "prophet":
            model = model_from_json(path.read_text(encoding="utf-8"))
            future = pd.DataFrame({"ds": trainers.prophet_ds(rows_df).to_numpy()})
            columns[target] = model.predict(future)["yhat"].to_numpy()
        else:
            if path not in lstm_cache:
                checkpoint = trainers.load_lstm_checkpoint(path)
                lstm_cache[path] = trainers.lstm_predict(checkpoint, df, rows_df)
            columns[target] = lstm_cache[path][target].to_numpy()
    return pd.DataFrame(columns)[TARGET_COLS]


def clipped(preds, val_df):
    return dataset.clip_predictions(preds, val_df["target_is_day"])


def score_all(df, train_df, val_df):
    started = time.monotonic()
    log.info("training random forest")
    rf = trainers.train_random_forest(train_df)
    log.info("training prophet")
    prophet = trainers.train_prophet(train_df)
    log.info("training lstm")
    lstm = trainers.train_lstm(train_df, df)
    preds = {
        "naive": trainers.naive_predict(val_df),
        "prophet": trainers.prophet_predict(prophet, val_df),
        "random_forest": trainers.rf_predict(rf, val_df),
        "lstm": trainers.lstm_predict(lstm, df, val_df),
    }
    reports = {name: trainers.evaluate(val_df, clipped(p, val_df)) for name, p in preds.items()}
    log.info("trained and scored in %.0fs", time.monotonic() - started)
    return rf, prophet, lstm, reports


def comparison_table(reports):
    names = ["naive", *FAMILIES]
    header = f"{'target':<28}" + "".join(f"{n:>34}" for n in names)
    sub = f"{'':<28}" + "".join(f"{'rmse':>12}{'mae':>11}{'r2':>11}" for _ in names)
    lines = [header, sub]
    for target in TARGET_COLS:
        cells = "".join(
            f"{reports[n][target]['rmse']:>12.4f}{reports[n][target]['mae']:>11.4f}{reports[n][target]['r2']:>11.4f}"
            for n in names
        )
        lines.append(f"{target:<28}{cells}")
    wind = "".join(
        f"{reports[n]['wind_direction_10m']['circular_mae']:>12.2f}{reports[n]['wind_direction_10m']['circular_rmse']:>11.2f}{'':>11}"
        for n in names
    )
    lines.append(f"{'wind_direction (circ mae/rmse)':<28}{wind}")
    return "\n".join(lines)


def metrics_report(reports, val_window):
    improvement = {
        family: {
            t: trainers.rmse_improvement_pct(reports["naive"][t]["rmse"], reports[family][t]["rmse"])
            for t in TARGET_COLS
        }
        for family in FAMILIES
    }
    return {"val_window": val_window, "families": reports, "rmse_improvement_pct_vs_naive": improvement}


def sanitize_metrics(metrics):
    return {k: float(v) if math.isfinite(v) else 0.0 for k, v in metrics.items()}


def fetch_production(mr):
    models = mr.get_models(MODEL_NAME)
    if not models:
        return None, None
    current = max(models, key=lambda m: m.version)
    return current, Path(current.download())


def write_bundle_files(bundle_dir, targets, reports, shap_report, eda, val_window):
    manifest = {
        "targets": targets,
        "feature_cols": MODEL_FEATURE_COLS,
        "trained_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "val_window": val_window,
    }
    write_json(bundle_dir / "manifest.json", manifest)
    write_json(bundle_dir / "metrics_report.json", metrics_report(reports, val_window))
    write_json(bundle_dir / "shap_summary.json", shap_report)
    write_json(bundle_dir / "eda_summary.json", eda)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project = dataset.connect_project()
    df = dataset.read_features(project)
    train_df, val_df = dataset.split_embargo(dataset.build_paired(df))
    val_window = {
        "start_utc": val_df["timestamp_utc"].min().isoformat(),
        "end_utc": val_df["timestamp_utc"].max().isoformat(),
        "rows": len(val_df),
    }

    rf, prophet, lstm, reports = score_all(df, train_df, val_df)
    print(comparison_table(reports))
    shap_report = explain.shap_summary(rf, val_df)
    eda = compute_eda_summary(df)

    mr = None if args.dry_run else project.get_model_registry()
    current, prod_dir = (None, None) if mr is None else fetch_production(mr)
    next_version = 1 if current is None else current.version + 1

    prod_targets = prod_rmse = None
    if current is not None:
        prod_manifest = json.loads((prod_dir / "manifest.json").read_text(encoding="utf-8"))
        prod_targets = prod_manifest["targets"]
        prod_preds = clipped(predict_bundle(prod_dir / "models", prod_targets, df, val_df), val_df)
        prod_report = trainers.evaluate(val_df, prod_preds)
        prod_rmse = {t: prod_report[t]["rmse"] for t in TARGET_COLS}

    candidates = best_candidates(reports)
    new_rmse = {t: candidates[t]["val_rmse"] for t in TARGET_COLS}
    promoted = promoted_targets(new_rmse, prod_rmse)
    if not promoted:
        log.info("no target improved on v%d; skipping registration", current.version)
        return

    bundle_dir = ARTIFACTS_DIR / "model_bundle" if args.dry_run else BUNDLE_DIR
    with tempfile.TemporaryDirectory() as scratch:
        candidates = save_candidate_models(candidates, rf, prophet, lstm, next_version, Path(scratch))
        targets = assemble_bundle(
            bundle_dir,
            candidates,
            scratch,
            prod_targets,
            None if prod_dir is None else prod_dir / "models",
            prod_rmse,
            promoted,
        )
    write_bundle_files(bundle_dir, targets, reports, shap_report, eda, val_window)
    log.info("promoted targets: %s", ", ".join(promoted))

    if args.dry_run:
        log.info("dry run: bundle written to %s", bundle_dir)
        return
    metrics = sanitize_metrics({f"{t}_val_rmse": e["val_rmse"] for t, e in targets.items()})
    model = mr.python.create_model(
        name=MODEL_NAME,
        version=next_version,
        metrics=metrics,
        description="Lahore 24-hour-ahead weather and air quality model bundle.",
    )
    model.save(str(bundle_dir))
    log.info("registered %s v%d", MODEL_NAME, next_version)


if __name__ == "__main__":
    main()
