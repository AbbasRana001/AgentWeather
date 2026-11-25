import logging

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from src.config import HORIZON_HOURS, MODEL_FEATURE_COLS, SEQ_LEN, TARGET_COLS
from src.models.dataset import lstm_sequences, target_frame
from src.pipelines.fetch_and_engineer import reconstruct_wind_direction

RF_PARAMS = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 5, "n_jobs": -1, "random_state": 42}
LSTM_ES_HOURS = 336
SEED = 42


def naive_predict(df):
    return df[TARGET_COLS].reset_index(drop=True)


def angular_diff(y_true_deg, y_pred_deg):
    return np.abs((np.asarray(y_true_deg) - np.asarray(y_pred_deg) + 180.0) % 360.0 - 180.0)


def circular_errors(y_true_deg, y_pred_deg):
    diff = angular_diff(y_true_deg, y_pred_deg)
    return float(diff.mean()), float(np.sqrt((diff**2).mean()))


def rmse_improvement_pct(naive_rmse, model_rmse):
    if naive_rmse < 1e-9:
        return 0.0
    return float((naive_rmse - model_rmse) / naive_rmse * 100.0)


def evaluate(val_df, preds):
    truth = target_frame(val_df)
    preds = preds.reset_index(drop=True)
    report = {}
    for target in TARGET_COLS:
        y_true, y_pred = truth[target].to_numpy(), preds[target].to_numpy()
        report[target] = {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "r2": float(r2_score(y_true, y_pred)),
        }
    pred_deg = reconstruct_wind_direction(preds["wind_dir_sin"], preds["wind_dir_cos"])
    mae, rmse = circular_errors(val_df["target_wind_direction_10m"].to_numpy(), pred_deg.to_numpy())
    report["wind_direction_10m"] = {"circular_mae": mae, "circular_rmse": rmse}
    return report


def train_random_forest(train_df, **overrides):
    params = {**RF_PARAMS, **overrides}
    features = train_df[MODEL_FEATURE_COLS]
    return {t: RandomForestRegressor(**params).fit(features, train_df[f"target_{t}"]) for t in TARGET_COLS}


def rf_predict(models, rows_df):
    features = rows_df[MODEL_FEATURE_COLS]
    return pd.DataFrame({t: models[t].predict(features) for t in TARGET_COLS})


def prophet_ds(df):
    return (df["timestamp_utc"] + pd.Timedelta(hours=HORIZON_HOURS)).dt.tz_localize(None)


def train_prophet(train_df):
    from prophet import Prophet

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    models = {}
    for target in TARGET_COLS:
        frame = pd.DataFrame({"ds": prophet_ds(train_df), "y": train_df[f"target_{target}"].to_numpy()})
        model = Prophet(
            daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True, uncertainty_samples=0
        )
        models[target] = model.fit(frame)
    return models


def prophet_predict(models, rows_df):
    future = pd.DataFrame({"ds": prophet_ds(rows_df).to_numpy()})
    return pd.DataFrame({t: models[t].predict(future)["yhat"].to_numpy() for t in TARGET_COLS})


class LSTMNet(nn.Module):
    def __init__(self):
        super().__init__()
        n_features, n_targets = len(MODEL_FEATURE_COLS), len(TARGET_COLS)
        self.lstm = nn.LSTM(n_features, 64, num_layers=2, dropout=0.1, batch_first=True)
        self.head = nn.Linear(64, n_targets)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


def lstm_split(train_df):
    stop_start = train_df["timestamp_utc"].max() - pd.Timedelta(hours=LSTM_ES_HOURS - 1)
    fit_end = stop_start - pd.Timedelta(hours=HORIZON_HOURS)
    fit = train_df[train_df["timestamp_utc"] <= fit_end].reset_index(drop=True)
    stop = train_df[train_df["timestamp_utc"] >= stop_start].reset_index(drop=True)
    return fit, stop


def _scale_stats(values):
    scale = values.std(axis=0)
    return values.mean(axis=0), np.where(scale < 1e-12, 1.0, scale)


def _checkpoint(net, x_mean, x_scale, y_mean, y_scale):
    return {
        "state_dict": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        "x_mean": [float(v) for v in x_mean],
        "x_scale": [float(v) for v in x_scale],
        "y_mean": [float(v) for v in y_mean],
        "y_scale": [float(v) for v in y_scale],
    }


def init_lstm_checkpoint():
    n_in, n_out = len(MODEL_FEATURE_COLS), len(TARGET_COLS)
    return _checkpoint(LSTMNet(), np.zeros(n_in), np.ones(n_in), np.zeros(n_out), np.ones(n_out))


def _tensors(df, rows_df, x_mean, x_scale, y_mean, y_scale):
    x, kept = lstm_sequences(df, rows_df["timestamp_utc"])
    y = target_frame(rows_df).set_axis(pd.DatetimeIndex(rows_df["timestamp_utc"])).loc[kept].to_numpy()
    return (
        torch.tensor((x - x_mean) / x_scale, dtype=torch.float32),
        torch.tensor((y - y_mean) / y_scale, dtype=torch.float32),
    )


def train_lstm(train_df, df, max_epochs=20, patience=4, batch_size=256):
    torch.manual_seed(SEED)
    fit_df, stop_df = lstm_split(train_df)
    # scalers are fit on the fit split only, so early stopping and validation never leak into them
    x_mean, x_scale = _scale_stats(fit_df[MODEL_FEATURE_COLS].to_numpy(dtype=float))
    y_mean, y_scale = _scale_stats(target_frame(fit_df).to_numpy())
    stats = (x_mean, x_scale, y_mean, y_scale)
    x_fit, y_fit = _tensors(df, fit_df, *stats)
    x_stop, y_stop = _tensors(df, stop_df, *stats)

    net = LSTMNet()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()
    best_loss, best_state, stale = float("inf"), None, 0
    for _ in range(max_epochs):
        net.train()
        order = torch.randperm(len(x_fit))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            optimizer.zero_grad()
            loss_fn(net(x_fit[idx]), y_fit[idx]).backward()
            optimizer.step()
        net.eval()
        with torch.no_grad():
            stop_loss = loss_fn(net(x_stop), y_stop).item()
        if stop_loss < best_loss:
            best_loss, stale = stop_loss, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    net.load_state_dict(best_state)
    return _checkpoint(net, *stats)


def save_lstm_checkpoint(path, checkpoint):
    torch.save(checkpoint, path)


def load_lstm_checkpoint(path):
    return torch.load(path, weights_only=True)


def lstm_predict(checkpoint, df, rows_df):
    net = LSTMNet()
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    x, kept = lstm_sequences(df, rows_df["timestamp_utc"])
    x = (x - np.asarray(checkpoint["x_mean"])) / np.asarray(checkpoint["x_scale"])
    with torch.no_grad():
        scaled = net(torch.tensor(x, dtype=torch.float32)).numpy()
    values = scaled * np.asarray(checkpoint["y_scale"]) + np.asarray(checkpoint["y_mean"])
    frame = pd.DataFrame(values, columns=TARGET_COLS, index=kept)
    if len(kept) != len(rows_df):
        raise ValueError(f"{len(rows_df) - len(kept)} rows lack a contiguous {SEQ_LEN}h feature window")
    return frame.reset_index(drop=True)
