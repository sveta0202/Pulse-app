"""Облегчённое решение: ЭМГ (6 каналов, 500 Гц) -> 20 углов кисти. 6 нейросетей вместо 27 в simple.py.

Схема (подробно — LITE.md):
  1. Признаки (feats.py), 50 Гц:  env — «громкость» каналов, envraw — + сырой сигнал, envcorr — + корреляции каналов.
  2. Какое движение? CNN на env + CNN на envraw + LogReg -> венгерский алгоритм
     (в тесте каждое из 8 движений встречается одинаковое число раз).
  3. Учитель: 3 регрессора TCN + BiLSTM (env, envraw, envcorr), метка движения на входе -> среднее.
  4. Студент: 1 регрессор на envraw, обучен на трейне + тесте с ответами учителя. Его ответ, 50 -> 500 Гц, и есть итог.

Каждый шаг (6 сетей + LogReg) сохраняется в --work; повторный запуск продолжает с места остановки.

  python lite.py cv --quick            # проверка кода: 1 фолд, 20 шагов обучения (минуты)
  python lite.py cv --fold 0           # честная оценка на одном дне из 7 (0..6); весь CV — без --fold
  python lite.py report                # итог по готовым фолдам
  python lite.py predict               # тест -> CSV (по умолчанию <data>/test и <data>/phase2_sample_submission.csv)
  python lite.py predict --time-limit 50   # не начинать новый шаг после 50 минут; повторять, пока не напишет "saved"
"""
import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from clf_features import rec_feats
from common import ACTIONS, ACTIVE, CONTEXT, day_folds, load_all, score
from feats import DS, build

CLF_KINDS = ["env", "envraw"]               # CNN-классификаторы движения
TEACHER_KINDS = ["env", "envraw", "envcorr"]  # учитель: разные признаки ошибаются по-разному
STUDENT_KIND = "envraw"                     # лучший одиночный студент в экспериментах
STEPS = dict(clf=800, teacher=1000, student=1500)  # шаги обучения (--quick: по 20)
N_LOG = 18  # первые 18 признаков любого набора — лог-огибающие (3 масштаба x 6 каналов)


# ---------------------------------------------------------------- признаки и мелкие утилиты

_cache = {}


def features(x, kind):
    """Признаки записи (кэш: в CV одна запись нужна во многих фолдах)."""
    key = (id(x), kind)
    if key not in _cache:
        _cache[key] = build(x, kind)
    return _cache[key]


def onehot(action):
    v = np.zeros(8, np.float32)
    v[ACTIONS.index(action)] = 1
    return v


def finger_weights(action):
    """Веса 20 углов в лоссе = веса метрики: активный палец 0.5, остальные по 0.125; Fist/Open/Close — по 0.2."""
    w = np.full(5, 0.2, np.float32)
    if action in ACTIVE:
        w[:] = 0.125
        w[ACTIVE[action]] = 0.5
    return np.repeat(w, 4) * 5


def upsample(p, T):
    """Линейная интерполяция предсказания с 50 Гц обратно на 500 Гц."""
    return np.stack([np.interp(np.arange(T) / DS, np.arange(len(p)), p[:, j]) for j in range(p.shape[1])], 1)


def assign(P, per_action):
    """Венгерский алгоритм: каждое движение получает ровно per_action записей, сумма log-вероятностей максимальна."""
    rows, cols = linear_sum_assignment(np.repeat(-np.log(P + 1e-9), per_action, 1))
    out = np.zeros(len(P), int)
    out[rows] = cols // per_action
    return out


# ---------------------------------------------------------------- сети

class Block(nn.Module):
    """Residual-блок: свёртка с разрежением -> GroupNorm -> GELU -> свёртка 1x1 (+ вход)."""

    def __init__(self, c, dilation, drop):
        super().__init__()
        self.conv = nn.Conv1d(c, c, 5, padding=2 * dilation, dilation=dilation)
        self.norm = nn.GroupNorm(1, c)
        self.pw = nn.Conv1d(c, c, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return x + self.drop(self.pw(F.gelu(self.norm(self.conv(x)))))


class Regressor(nn.Module):
    """[признаки + движение] -> 5 свёрточных блоков (окно ±1.2 с) -> BiLSTM x2 (вся запись) -> 20 углов на кадр."""

    def __init__(self, cin, hidden, drop):
        super().__init__()
        self.inp = nn.Conv1d(cin + 8, hidden, 1)
        self.tcn = nn.Sequential(*[Block(hidden, 2 ** i, drop) for i in range(5)])
        self.lstm = nn.LSTM(hidden, hidden, num_layers=2, batch_first=True, bidirectional=True, dropout=drop)
        self.out = nn.Linear(2 * hidden + 8, 20)

    def forward(self, x, a):  # x: (B, T, C), a: (B, 8)
        a = a[:, None, :].expand(-1, x.shape[1], -1)
        h = self.tcn(self.inp(torch.cat([x, a], 2).transpose(1, 2))).transpose(1, 2)
        h, _ = self.lstm(h)
        return self.out(torch.cat([h, a], 2))


class Classifier(nn.Module):
    """6 свёрточных блоков -> среднее и разброс по времени -> 8 движений."""

    def __init__(self, cin, hidden=64, drop=0.3):
        super().__init__()
        self.inp = nn.Conv1d(cin, hidden, 1)
        self.tcn = nn.Sequential(*[Block(hidden, 2 ** i, drop) for i in range(6)])
        self.head = nn.Linear(2 * hidden, 8)

    def forward(self, x):  # x: (B, T, C)
        h = self.tcn(self.inp(x.transpose(1, 2)))
        return self.head(torch.cat([h.mean(2), h.std(2)], 1))


# ---------------------------------------------------------------- обучение

def augment(xb, sd, gain, chdrop):
    """Имитация дрейфа электродов между днями: выпадение каналов, случайное усиление каналов, шум."""
    bs, _, nch = xb.shape
    chan = torch.arange(nch) % 6  # какому электроду принадлежит признак
    if chdrop:
        xb = xb * (torch.rand(bs, 1, 6) > chdrop).float()[:, :, chan]
    g = torch.randn(bs, 1, 6) * gain + torch.randn(bs, 1, 1) * gain  # лог-усиление: каждого канала + общее
    mask = torch.zeros(nch)
    mask[:N_LOG] = 1  # усиление сдвигает только лог-огибающие
    return xb + g[:, :, chan] * mask / torch.tensor(sd) + 0.05 * torch.randn_like(xb)


def fit(net, X, steps, loss_fn, bs, crop, gain, chdrop=0.0, wd=1e-2, clip=True, seed=0):
    """Общий цикл обучения: случайные окна по crop кадров, AdamW + OneCycle. X — список матриц (T, C)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allx = np.concatenate(X)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6  # нормировка по обучающим данным
    Xt = [torch.tensor((x - mu) / sd, dtype=torch.float32) for x in X]
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps, pct_start=0.1)
    net.train()
    for _ in range(steps):
        idx = rng.integers(0, len(X), bs)
        starts = [rng.integers(0, len(Xt[i]) - crop) for i in idx]
        xb = torch.stack([Xt[i][s:s + crop] for i, s in zip(idx, starts)])
        loss = loss_fn(net, augment(xb, sd, gain, chdrop), idx, starts)
        opt.zero_grad()
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
    net.eval()
    return mu, sd  # та же нормировка для новых записей


def pack_network(net, mu, sd):
    """Tensor-only artifact: no Python model objects or closures."""
    return {"state": net.state_dict(), "mu": torch.as_tensor(mu), "sd": torch.as_tensor(sd)}


def normalize(x, mu, sd):
    return torch.tensor((x - mu) / sd, dtype=torch.float32)


@torch.no_grad()
def reg_angles(net, xt, action):
    return net(xt[None], torch.from_numpy(onehot(action))[None])[0].numpy()


@torch.no_grad()
def classifier_log_probs(net, xt, crop=500, batch_size=8):
    starts = range(0, len(xt) - crop + 1, crop // 4)
    if not starts:
        raise ValueError("Запись короче окна классификатора")
    outputs = []
    for i in range(0, len(starts), batch_size):
        windows = torch.stack([xt[s:s + crop] for s in starts[i:i + batch_size]])
        outputs.append(F.log_softmax(net(windows), 1))
    return torch.cat(outputs, dim=0).mean(0).numpy()


def train_regressor(X, Y, actions, hidden, drop, steps, chdrop, seed=0, crop=400, export=False):
    """Регрессор углов. Лосс — L1 с весами метрики. Возвращает predict(x, action) -> (T/10, 20)."""
    Yt = [torch.tensor(y) for y in Y]
    A = torch.tensor(np.stack([onehot(a) for a in actions]))
    W = torch.tensor(np.stack([finger_weights(a) for a in actions]))

    def loss_fn(net, xb, idx, starts):
        yb = torch.stack([Yt[i][s:s + crop] for i, s in zip(idx, starts)])
        return ((net(xb, A[idx]) - yb).abs() * W[idx][:, None, :]).mean()

    net = Regressor(X[0].shape[1], hidden, drop)
    mu, sd = fit(net, X, steps, loss_fn, bs=32, crop=crop, gain=0.5, chdrop=chdrop, seed=seed)

    @torch.no_grad()
    def predict(x, action):  # вся запись целиком
        return reg_angles(net, normalize(x, mu, sd), action)

    return (predict, pack_network(net, mu, sd)) if export else predict


def train_classifier(X, labels, steps, seed=0, crop=500, export=False):
    """CNN-классификатор движения на окнах по 10 с. Возвращает log_proba(x): среднее по окнам всей записи."""
    y = torch.tensor(labels)
    loss_fn = lambda net, xb, idx, starts: F.cross_entropy(net(xb), y[idx], label_smoothing=0.1)  # noqa: E731
    net = Classifier(X[0].shape[1])
    mu, sd = fit(net, X, steps, loss_fn, bs=64, crop=crop, gain=0.3, wd=5e-2, clip=False, seed=seed)

    @torch.no_grad()
    def log_proba(x):
        return classifier_log_probs(net, normalize(x, mu, sd), crop)

    return (log_proba, pack_network(net, mu, sd)) if export else log_proba


# ---------------------------------------------------------------- шаги с сохранением на диск

class Pause(Exception):
    pass


DEADLINE = float("inf")


def step(path, desc, fn):
    """Если результат шага уже на диске — читаем, иначе считаем и сохраняем (атомарно)."""
    path = Path(path)
    if path.exists():
        return pickle.loads(path.read_bytes())
    if time.time() > DEADLINE:
        raise Pause
    t0 = time.time()
    print(f"  {desc} ...", end=" ", flush=True)
    res = fn()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".tmp").write_bytes(pickle.dumps(res))
    os.replace(path.with_suffix(".tmp"), path)
    print(f"{time.time() - t0:.0f} с", flush=True)
    return res


# ---------------------------------------------------------------- пайплайн

def classify(d, x_tr, act_tr, x_te):
    """Вероятности 8 движений для каждой тестовой записи: геометрическое среднее 2 CNN и LogReg."""
    y = np.array([ACTIONS.index(a) for a in act_tr])

    def cnn(kind):
        lp = train_classifier([features(x, kind) for x in x_tr], y, STEPS["clf"])
        return np.stack([lp(features(x, kind)) for x in x_te])

    def logreg():
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))
        m.fit(np.stack([rec_feats(x) for x in x_tr]), y)
        return np.log(m.predict_proba(np.stack([rec_feats(x) for x in x_te])) + 1e-9)

    logps = [step(d / f"clf_{k}.pkl", f"классификатор CNN {k}", lambda k=k: cnn(k)) for k in CLF_KINDS]
    logps.append(step(d / "clf_logreg.pkl", "классификатор LogReg", logreg))
    P = np.exp(np.mean(logps, 0))
    return P / P.sum(1, keepdims=True)


def regress(d, x_tr, y_tr, act_tr, x_te, act_te):
    """Учитель (3 сети) -> его ответы на тесте -> студент (1 сеть) на трейне + тесте. Выход 50 Гц."""
    Y = [y[::DS] for y in y_tr]

    def teacher(kind):
        F_tr, F_te = [features(x, kind) for x in x_tr], [features(x, kind) for x in x_te]
        m = train_regressor(F_tr, Y, act_tr, hidden=64, drop=0.2, steps=STEPS["teacher"],
                            chdrop=0.15 if kind == "envraw" else 0.0)
        return [m(f, a) for f, a in zip(F_te, act_te)]

    preds = [step(d / f"teacher_{k}.pkl", f"учитель {k}", lambda k=k: teacher(k)) for k in TEACHER_KINDS]
    t_pred = [np.mean(p, 0) for p in zip(*preds)]

    def student():
        F_tr, F_te = [features(x, STUDENT_KIND) for x in x_tr], [features(x, STUDENT_KIND) for x in x_te]
        m = train_regressor(F_tr + F_te, Y + t_pred, list(act_tr) + list(act_te), hidden=96, drop=0.3,
                            steps=STEPS["student"], chdrop=0.15, seed=50)
        return [m(f, a) for f, a in zip(F_te, act_te)]

    s_pred = step(d / "student.pkl", f"студент {STUDENT_KIND}", student)
    return t_pred, s_pred


def predict_actions(d, x_tr, act_tr, x_te, n_groups):
    P = classify(d, x_tr, act_tr, x_te)
    choice = assign(P, n_groups) if len(x_te) == 8 * n_groups else P.argmax(1)
    return [ACTIONS[i] for i in choice], P, choice


# ---------------------------------------------------------------- режимы

def run_cv(a, work):
    meta, xs, ys = load_all()
    folds = list(enumerate(day_folds(meta)))
    for k, (tr, va) in (folds if a.fold is None else [folds[i] for i in a.fold]):
        d = work / "cv" / f"fold{k}"
        if (d / "result.pkl").exists():
            print(f"фолд {k}: уже готов")
            continue
        print(f"фолд {k} (0..6): день {va.day_id.iloc[0]}, записей {len(va)}", flush=True)
        x_tr, x_va = [xs[r] for r in tr.recording_id], [xs[r] for r in va.recording_id]
        acts, _, _ = predict_actions(d, x_tr, list(tr.action), x_va, va.session_id.nunique())
        t_pred, s_pred = regress(d, x_tr, [ys[r] for r in tr.recording_id], list(tr.action), x_va, acts)
        step(d / "result.pkl", "итог фолда",
             lambda: dict(recs=list(va.recording_id), acts=acts, teacher=t_pred, final=s_pred))
    run_report(a, work)


def run_report(a, work):
    meta, _, ys = load_all()
    files = sorted((work / "cv").glob("fold*/result.pkl"))
    if not files:
        print("Нет готовых фолдов. Сначала: python lite.py cv --fold 0")
        return
    true_act = dict(zip(meta.recording_id, meta.action))
    teach, final, correct = {}, {}, []
    for f in files:
        res = pickle.loads(f.read_bytes())
        for r, act, tp, sp in zip(res["recs"], res["acts"], res["teacher"], res["final"]):
            teach[r], final[r] = upsample(tp, len(ys[r])), upsample(sp, len(ys[r]))
            correct.append(act == true_act[r])
    s_t, _ = score(meta, teach, ys)
    s_f, df = score(meta, final, ys)
    print(f"\nготово фолдов: {len(files)} из 7, сессий: {df.session_id.nunique()}")
    print(f"точность движения: {np.mean(correct):.4f}")
    print(f"wMAE учителя:      {s_t:.4f}")
    print(f"wMAE итог:         {s_f:.4f}" + ("" if len(files) == 7 else "   (не все фолды — оценка предварительная)"))
    print(df.groupby("action").err.mean().round(3).to_string())


def run_predict(a, work):
    meta, xs, ys = load_all()
    template = pd.read_csv(a.template)
    rids = list(dict.fromkeys(template.recording_id))
    d = work / "predict"
    d.mkdir(parents=True, exist_ok=True)
    ids_file = d / "recording_ids.txt"
    if ids_file.exists() and ids_file.read_text().split() != rids:
        raise SystemExit(f"В {d} лежат шаги для другого теста. Удалите папку или задайте другой --work")
    ids_file.write_text("\n".join(rids))
    x_te = [np.load(Path(a.test_dir) / f"{r}.npy").astype(np.float32) for r in rids]
    x_tr = [xs[r] for r in meta.recording_id]
    acts, P, choice = predict_actions(d, x_tr, list(meta.action), x_te, len(rids) // 8)
    _, preds = regress(d, x_tr, [ys[r] for r in meta.recording_id], list(meta.action), x_te, acts)

    cols = [f"angle_{i:02d}" for i in range(20)]
    out = template.copy()
    for r, x, p in zip(rids, x_te, preds):
        full = np.clip(upsample(p, len(x)), -np.pi, np.pi)
        rows = (template.recording_id == r).to_numpy()
        idx = template.loc[rows, "sample_index"].to_numpy()
        assert idx.min() >= CONTEXT and idx.max() < len(full)
        out.loc[rows, cols] = full[idx]
    assert np.isfinite(out[cols].to_numpy(np.float64)).all()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False, float_format="%.6f")
    info = pd.DataFrame({"recording_id": rids, "action": acts, "p_assigned": P[np.arange(len(rids)), choice],
                         "argmax": [ACTIONS[i] for i in P.argmax(1)]})
    info.to_csv(Path(a.out).with_suffix(".actions.csv"), index=False)
    print(info.to_string())
    print("saved", a.out, out.shape)


def find_data(given):
    """Папка, где лежит train/metadata.csv: --data или типичные места (на 1–2 уровня глубже тоже)."""
    here = Path(__file__).resolve().parent
    roots = [Path(given)] if given else [Path.cwd() / "data", here / "data", Path.cwd(), here, Path("/data"),
                                         Path("/mnt/data"), Path("/input"), Path("/kaggle/input")]
    for pattern in ["train/metadata.csv", "*/train/metadata.csv", "*/*/train/metadata.csv"]:
        for root in roots:
            hits = sorted(root.glob(pattern)) if root.is_dir() else []
            if hits:
                return hits[0].parent.parent
    raise SystemExit("Не нашёл train/metadata.csv. Укажите --data /путь (внутри train/metadata.csv, signals/, targets/)")


def main(argv=None):
    global DEADLINE
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["cv", "report", "predict"])
    p.add_argument("--data", help="папка с train/ (по умолчанию ищется автоматически)")
    p.add_argument("--test-dir", help="по умолчанию <data>/test")
    p.add_argument("--template", help="по умолчанию <data>/phase2_sample_submission.csv")
    p.add_argument("--out", default="outputs/lite.csv")
    p.add_argument("--work", default="work_lite", help="куда сохранять готовые шаги")
    p.add_argument("--fold", type=int, nargs="+", choices=range(7), help="cv: только эти фолды (0..6)")
    p.add_argument("--time-limit", type=float, help="минуты: после них не начинать новый шаг")
    p.add_argument("--quick", action="store_true", help="проверка кода: 20 шагов на сеть, в cv — один фолд")
    a = p.parse_args(argv)
    work = Path(a.work) / ("quick" if a.quick else "full")  # шаги разных режимов не смешиваются
    if a.quick:
        STEPS.update(clf=20, teacher=20, student=20)
        a.fold = a.fold or [5]
    # Один процесс обучает одну маленькую сеть: больше половины ядер не ускоряет (на 1 CPU — 1 поток).
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    DEADLINE = time.time() + a.time_limit * 60 if a.time_limit else float("inf")
    common.DATA = find_data(a.data)
    print("потоков CPU:", torch.get_num_threads(), "| data:", common.DATA, "| work:", work, flush=True)
    if a.mode == "predict":
        a.test_dir = a.test_dir or str(common.DATA / "test")
        a.template = a.template or str(common.DATA / "phase2_sample_submission.csv")
        for path in (a.test_dir, a.template):
            if not Path(path).exists():
                raise SystemExit(f"Нет {path}. Укажите --test-dir и --template явно")
    try:
        {"cv": run_cv, "report": run_report, "predict": run_predict}[a.mode](a, work)
    except Pause:
        print(f"\nЛимит времени. Готовые шаги в {work}. Запустите ту же команду ещё раз — продолжит с этого места.")


if __name__ == "__main__":
    main()
