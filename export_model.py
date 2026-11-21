"""Train once on trusted AIRI data and export a standalone CPU checkpoint."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from feats import DS, build
from lite import (CLF_KINDS, STEPS, STUDENT_KIND, TEACHER_KINDS, find_data,
                  train_classifier, train_regressor)
from model_adapter import MIN_SAMPLES, PulseModel, record_features


def export(data, output, distill_from=None, threads=2, steps=None):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    steps = STEPS if steps is None else steps
    common.DATA = find_data(data)
    meta, xs, ys = common.load_all()
    actions = list(meta.action)
    if set(actions) != set(common.ACTIONS):
        raise ValueError("В train/metadata.csv нужны все типы движений из common.ACTIONS.")
    x = [xs[r] for r in meta.recording_id]
    y = [ys[r] for r in meta.recording_id]
    for sig, target in zip(x, y):
        if (sig.ndim != 2 or sig.shape[1] != 6 or len(sig[::DS]) <= 500
                or target.shape != (len(sig), 20)
                or not np.isfinite(sig).all() or not np.isfinite(target).all()):
            raise ValueError("Тренировочные пары должны быть конечными (T,6)/(T,20), с T > 5000.")
    y = [angles[::DS] for angles in y]
    labels = np.array([common.ACTIONS.index(a) for a in actions])
    cnn, predictors = {}, {}
    train_feats = {}
    for kind in CLF_KINDS:
        print(f"Обучение CNN: {kind}", flush=True)
        train_feats[kind] = [build(sig, kind) for sig in x]
        predictors[kind], cnn[kind] = train_classifier(
            train_feats[kind], labels, steps["clf"], export=True)

    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
    print("Обучение LogReg", flush=True)
    lr.fit(np.stack([record_features(sig) for sig in x]), labels)
    scaler, reg = lr.steps[0][1], lr.steps[1][1]
    if not np.array_equal(reg.classes_, np.arange(len(common.ACTIONS))):
        raise ValueError("Несовместимый порядок классов LogReg.")
    xstudent = train_feats[STUDENT_KIND]
    if distill_from:
        paths = sorted(Path(distill_from).glob("*.npy"))
        if not paths:
            raise ValueError("В папке дистилляции нет файлов .npy.")
        extra = [np.load(p, allow_pickle=False).astype(np.float32) for p in paths]
        if any(sig.ndim != 2 or sig.shape[1] != 6 or len(sig) < MIN_SAMPLES
               or not np.isfinite(sig).all() for sig in extra):
            raise ValueError("Записи для дистилляции должны содержать конечную ЭМГ (T,6), T >= 5000.")
        extra_f = {k: [build(sig, k) for sig in extra] for k in CLF_KINDS}
        logps = [np.stack([predictors[k](f) for f in extra_f[k]]) for k in CLF_KINDS]
        logps.append(np.log(lr.predict_proba(np.stack([record_features(sig) for sig in extra])) + 1e-9))
        extra_actions = [common.ACTIONS[i] for i in np.mean(logps, axis=0).argmax(axis=1)]
        teacher_preds = []
        for kind in TEACHER_KINDS:
            print(f"Обучение учителя: {kind}", flush=True)
            source = train_feats[kind] if kind in train_feats else [build(sig, kind) for sig in x]
            dest = extra_f[kind] if kind in extra_f else [build(sig, kind) for sig in extra]
            teacher = train_regressor(source, y, actions, hidden=64, drop=0.2, steps=steps["teacher"],
                                      chdrop=0.15 if kind == STUDENT_KIND else 0.0)
            teacher_preds.append([teacher(f, a) for f, a in zip(dest, extra_actions)])
        xstudent = xstudent + extra_f[STUDENT_KIND]
        y = y + [np.mean(p, axis=0) for p in zip(*teacher_preds)]
        actions = actions + extra_actions

    print("Обучение регрессора envraw", flush=True)
    _, student = train_regressor(xstudent, y, actions, hidden=96, drop=0.3,
                                 steps=steps["student"], chdrop=0.15, seed=50, export=True)
    bundle = {
        "format": 1, "actions": list(common.ACTIONS), "fingers": list(common.FINGERS),
        "sample_rate": 500, "training": "distilled" if distill_from else "supervised",
        "cnn": cnn, "student": student,
        "logreg": {k: torch.as_tensor(v) for k, v in
                   dict(mean=scaler.mean_, scale=scaler.scale_, coef=reg.coef_, bias=reg.intercept_).items()},
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        torch.save(bundle, tmp)
        PulseModel(tmp).predict(x[0][:MIN_SAMPLES])
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"Готово: {path} ({path.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="папка, содержащая train/")
    parser.add_argument("--out", default="weights/pulse.pt")
    parser.add_argument("--distill-from", help="необязательная папка с дополнительными EMG .npy без углов")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads должен быть положительным")
    export(args.data, args.out, args.distill_from, args.threads)
