"""Validated EMG input and CPU inference. No training or global feature cache."""
import csv
import io
import re
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax

from clf_features import rec_feats
from common import ACTIONS, FINGERS
from feats import DS, build
from lite import Classifier, Regressor, classifier_log_probs, normalize, reg_angles, upsample

MIN_SAMPLES = 5_000
MAX_SAMPLES = 60_000
COLS = [f"angle_{i:02d}" for i in range(20)]
N_FEATURES = {"env": 18, "envraw": 78}


class InputError(ValueError):
    pass


def check_count(n):
    if not MIN_SAMPLES <= n <= MAX_SAMPLES:
        raise InputError(f"Нужно от {MIN_SAMPLES} до {MAX_SAMPLES} отсчётов на канал (500 Гц).")


def read_npy(body):
    with io.BytesIO(body) as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0,
                   (3, 0): np.lib.format.read_array_header_2_0}  # numeric headers are ASCII
        if version not in readers:
            raise InputError("Неподдерживаемая версия NPY.")
        shape, fortran, dtype = readers[version](stream, max_header_size=10_000)
        if len(shape) != 2 or shape[1] != 6 or fortran or dtype.str not in ("<f4", "<f8"):
            raise InputError("NPY: ожидается массив (T, 6), little-endian float32/64, C-order.")
        check_count(shape[0])
        offset = stream.tell()
        if len(body) - offset != shape[0] * 6 * dtype.itemsize:
            raise InputError("Длина данных не совпадает с заголовком NPY.")
        return np.frombuffer(body, dtype=dtype, offset=offset).reshape(shape)


def read_csv(body):
    with io.TextIOWrapper(io.BytesIO(body), encoding="utf-8-sig", newline="") as handle:
        first = next((line for line in handle if line.strip()), "")
        handle.seek(0)
        reader = csv.reader(handle, delimiter=";" if ";" in first and "," not in first else ",", strict=True)
        rows = []
        first_row = True
        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            if len(row) != 6:
                raise InputError("CSV: ожидается ровно 6 столбцов.")
            try:
                values = [float(v) for v in row]
            except ValueError as exc:
                header = [v.strip().casefold() for v in row]
                if (first_row and len(set(header)) == 6
                        and all(re.fullmatch(r"[^\W\d]\w*", v) and v not in {"nan", "inf", "infinity"}
                                for v in header)):
                    first_row = False
                    continue
                raise InputError("CSV: ожидаются числовые значения с десятичной точкой.") from exc
            first_row = False
            if not all(np.isfinite(values)):
                raise InputError("В ЭМГ есть NaN или Infinity.")
            rows.append(values)
            if len(rows) > MAX_SAMPLES:
                check_count(len(rows))
    check_count(len(rows))
    return np.asarray(rows)


def read_emg(body, filename):
    try:
        ext = Path(filename).suffix.lower()
        if ext not in {".npy", ".csv"}:
            raise InputError("Укажите имя файла с расширением .npy или .csv.")
        x = read_npy(body) if ext == ".npy" else read_csv(body)
        with np.errstate(over="ignore", invalid="ignore"):
            x = x.astype(np.float32, copy=False)
        if not np.isfinite(x).all():
            raise InputError("В ЭМГ есть NaN, Infinity или число вне диапазона float32.")
        return x
    except InputError:
        raise
    except (ValueError, EOFError, SyntaxError, csv.Error, OverflowError) as exc:
        raise InputError("Не удалось прочитать входной файл ЭМГ.") from exc


def record_features(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = rec_feats(x)
    if not np.isfinite(result).all():
        raise InputError("Не удалось вычислить признаки: проверьте, что каналы не постоянные и сигнал не нулевой.")
    return result


class PulseModel:
    def __init__(self, path):
        data = torch.load(path, map_location="cpu", weights_only=True)
        if (data.get("format") != 1 or data["actions"] != list(ACTIONS)
                or data["fingers"] != list(FINGERS) or data.get("sample_rate") != 500):
            raise ValueError("Несовместимый формат весов, порядок углов/движений или частота.")

        def tensor_array(t, shape):
            a = t.detach().numpy()
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError("Некорректная форма или значения в весах.")
            return a

        def network(net, group, features):
            if not all(t.is_floating_point() and torch.isfinite(t).all() for t in group["state"].values()):
                raise ValueError("Некорректный state_dict.")
            net.load_state_dict(group["state"], strict=True)
            mu, sd = (tensor_array(group[k], (features,)) for k in ("mu", "sd"))
            if np.any(sd <= 0):
                raise ValueError("Некорректная нормировка модели.")
            return net.eval(), mu, sd

        self.classifiers = {
            k: network(Classifier(n), data["cnn"][k], n) for k, n in N_FEATURES.items()
        }
        n = N_FEATURES["envraw"]
        self.student = network(Regressor(n, 96, 0.3), data["student"], n)
        rng = np.random.default_rng(42)
        warmup = rng.normal(0, 1e-5, size=(MIN_SAMPLES, 6)).astype(np.float32)
        width = record_features(warmup).size
        self.lr = {k: tensor_array(data["logreg"][k], shape) for k, shape in
                   {"mean": (width,), "scale": (width,), "coef": (len(ACTIONS), width),
                    "bias": (len(ACTIONS),)}.items()}
        if np.any(self.lr["scale"] <= 0):
            raise ValueError("Некорректная нормировка LogReg.")
        # Verify the full loaded graph before advertising readiness. No synthetic output is served.
        self.predict(warmup)

    @torch.inference_mode()
    def predict(self, x):
        rec = record_features(x)
        feats = {k: build(x, k) for k in N_FEATURES}
        if not all(np.isfinite(f).all() for f in feats.values()):
            raise InputError("В признаках ЭМГ есть NaN или Infinity. Проверьте входной файл.")
        logps = []
        for k, (net, mu, sd) in self.classifiers.items():
            logps.append(classifier_log_probs(net, normalize(feats[k], mu, sd)))
        z = (rec - self.lr["mean"]) / self.lr["scale"]
        logps.append(np.log(softmax(self.lr["coef"] @ z + self.lr["bias"]) + 1e-9))
        scores = np.mean(logps, axis=0)
        if not np.isfinite(scores).all():
            raise RuntimeError("Некорректный результат классификации")
        action = ACTIONS[int(scores.argmax())]
        net, mu, sd = self.student
        pred = reg_angles(net, normalize(feats["envraw"], mu, sd), action)
        if pred.shape != ((len(x) + DS - 1) // DS, len(COLS)) or not np.isfinite(pred).all():
            raise RuntimeError("Некорректный результат регрессии")
        return np.clip(upsample(pred, len(x)), -np.pi, np.pi)


def predict_emg_file(model, file_bytes, filename):
    angles = model.predict(read_emg(file_bytes, filename))
    output = io.StringIO()
    output.write(",".join(COLS) + "\n")
    np.savetxt(output, angles, delimiter=",", fmt="%.6f")
    return output.getvalue().encode("utf-8")
