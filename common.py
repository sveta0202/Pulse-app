from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
FINGERS = ["Index", "Middle", "Little", "Ring", "Thumb"]
ACTIONS = ["Index", "Middle", "Little", "Ring", "Thumb", "Fist", "Open", "Close"]
ACTIVE = {f: i for i, f in enumerate(FINGERS)}
CONTEXT = 1000


def load_meta():
    return pd.read_csv(DATA / "train" / "metadata.csv")


def load_all():
    meta = load_meta()
    xs, ys = {}, {}
    for rid in meta.recording_id:
        xs[rid] = np.load(DATA / "train" / "signals" / f"{rid}.npy").astype(np.float32)
        ys[rid] = np.load(DATA / "train" / "targets" / f"{rid}.npy").astype(np.float32)
    return meta, xs, ys


def finger_mae(y, p):
    return np.abs(y[CONTEXT:] - p[CONTEXT:]).reshape(-1, 5, 4).mean(axis=(0, 2))


def record_error(y, p, action):
    err = finger_mae(y, p)
    a = ACTIVE.get(action)
    if a is None:
        return float(err.mean())
    return float(0.5 * err[a] + 0.125 * (err.sum() - err[a]))


def score(meta, preds, ys):
    rows = []
    for r in meta.itertuples(index=False):
        if r.recording_id in preds:
            rows.append((r.session_id, r.action, record_error(ys[r.recording_id], preds[r.recording_id], r.action)))
    df = pd.DataFrame(rows, columns=["session_id", "action", "err"])
    return float(df.groupby("session_id").err.mean().mean()), df


def day_folds(meta):
    days = sorted(meta.day_id.unique())
    return [(meta[meta.day_id != d], meta[meta.day_id == d]) for d in days]
