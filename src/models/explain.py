import numpy as np

from src.config import MODEL_FEATURE_COLS, TARGET_COLS

TOP_K = 10
SAMPLE_ROWS = 200


def direction_correlation(feature_vals, shap_vals):
    feature_vals, shap_vals = np.asarray(feature_vals, dtype=float), np.asarray(shap_vals, dtype=float)
    if feature_vals.std() < 1e-12 or shap_vals.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(feature_vals, shap_vals)[0, 1])


def _target_summary(model, sample):
    import shap

    shap_values = shap.TreeExplainer(model).shap_values(sample, check_additivity=False)
    mean_abs = np.abs(shap_values).mean(axis=0)
    rows = []
    for idx in np.argsort(-mean_abs, kind="stable")[:TOP_K]:
        rows.append(
            {
                "feature": MODEL_FEATURE_COLS[idx],
                "mean_abs_shap": float(mean_abs[idx]),
                "direction_corr": direction_correlation(sample.iloc[:, idx], shap_values[:, idx]),
            }
        )
    return rows


def shap_summary(models, val_df):
    sample = val_df.sample(n=min(SAMPLE_ROWS, len(val_df)), random_state=42)[MODEL_FEATURE_COLS]
    return {target: _target_summary(models[target], sample) for target in TARGET_COLS}
