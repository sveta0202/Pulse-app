import numpy as np
from scipy.signal import butter, sosfiltfilt
DS=10
def envelopes(x, cut=(2,5,15)):
    x=np.asarray(x,np.float64)/1e-5
    sos=butter(4,[20,240],btype='band',fs=500,output='sos'); xb=sosfiltfilt(sos,x,axis=0)
    out=[]
    r=xb**2
    for c in cut:
        s=butter(2,c,fs=500,output='sos'); e=np.sqrt(np.maximum(sosfiltfilt(s,r,axis=0),1e-6))
        out.append(np.log(e))
    return np.concatenate(out,1)[::DS].astype(np.float32)
def norm_rec(f):
    return (f-np.median(f,0))/(f.std(0)+1e-6)
def lagstack(f, lags):
    T=len(f); out=[]
    for l in lags:
        idx=np.clip(np.arange(T)+l,0,T-1); out.append(f[idx])
    return np.concatenate(out,1)


def frame_feats(x, win=100):
    x = np.asarray(x, np.float64) / 1e-5
    sos = butter(4, [10, 190], btype="band", fs=500, output="sos")
    xb = sosfiltfilt(sos, x, axis=0)
    T = len(xb)
    pad = np.pad(xb, ((win // 2, win // 2), (0, 0)), mode="reflect")
    centers = np.arange(0, T, DS)
    idx = centers[:, None] + np.arange(win)[None, :]
    fr = pad[idx]
    rms = np.log(np.sqrt((fr ** 2).mean(1)) + 1e-3)
    wl = np.log(np.abs(np.diff(fr, axis=1)).mean(1) + 1e-3)
    zc = (np.diff(np.sign(fr), axis=1) != 0).mean(1)
    spec = np.abs(np.fft.rfft(fr * np.hanning(win)[None, :, None], axis=1)) ** 2
    freqs = np.fft.rfftfreq(win, 1 / 500)
    tot = spec.sum(1) + 1e-9
    mnf = (spec * freqs[None, :, None]).sum(1) / tot / 100
    cum = np.cumsum(spec, 1) / tot[:, None, :]
    mdf = freqs[np.argmax(cum >= 0.5, axis=1)] / 100
    return np.concatenate([rms, wl, zc, mnf, mdf], 1).astype(np.float32)


def corr_feats(x, win=150, bands=((10, 40), (40, 80), (80, 150))):
    x = np.asarray(x, np.float64) / 1e-5
    iu = np.triu_indices(6, 1)
    out = []
    for lo, hi in bands:
        xb = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=500, output="sos"), x, axis=0)
        prod = xb[:, :, None] * xb[:, None, :]
        s = butter(2, 500 / win / 2, fs=500, output="sos")
        cov = sosfiltfilt(s, prod.reshape(len(x), -1), axis=0)[::DS].reshape(-1, 6, 6)
        d = np.sqrt(np.clip(np.einsum("tii->ti", cov), 1e-9, None))
        c = cov / (d[:, :, None] * d[:, None, :])
        out.append(np.arctanh(np.clip(c[:, iu[0], iu[1]], -0.999, 0.999)))
    return np.concatenate(out, 1).astype(np.float32)


def raw_stack(x):
    x = np.asarray(x, np.float64) / 1e-5
    xb = sosfiltfilt(butter(4, [10, 190], btype="band", fs=500, output="sos"), x, axis=0)
    T = len(xb)
    n = (T + DS - 1) // DS
    pad = np.pad(xb, ((0, n * DS - T), (0, 0)))
    return (np.sign(pad) * np.log1p(np.abs(pad))).reshape(n, DS * 6).astype(np.float32)


def _spd_fn(C, fn):
    w, v = np.linalg.eigh(C)
    return (v * fn(np.clip(w, 1e-10, None))[..., None, :]) @ np.swapaxes(v, -1, -2)


def tangent_feats(x, win=150, bands=((5, 40), (40, 80), (80, 150)), recenter=True):
    x = np.asarray(x, np.float64) / 1e-5
    iu = np.triu_indices(6)
    coef = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))
    out = []
    for lo, hi in bands:
        xb = sosfiltfilt(butter(4, [lo, hi], btype="band", fs=500, output="sos"), x, axis=0)
        prod = (xb[:, :, None] * xb[:, None, :]).reshape(len(x), -1)
        s = butter(2, 500 / win / 2, fs=500, output="sos")
        C = sosfiltfilt(s, prod, axis=0)[::DS].reshape(-1, 6, 6)
        C = 0.5 * (C + np.swapaxes(C, 1, 2)) + 1e-6 * np.eye(6)
        L = _spd_fn(C, np.log)
        if recenter:
            R = _spd_fn(L.mean(0), np.exp)
            Ri = _spd_fn(R, lambda w: w ** -0.5)
            L = _spd_fn(Ri @ C @ Ri, np.log)
        out.append(L[:, iu[0], iu[1]] * coef)
    return np.concatenate(out, 1).astype(np.float32)


def build(x, kind):
    if kind == "allfeat":
        e = envelopes(x)
        return np.concatenate([e, corr_feats(x), tangent_feats(x), raw_stack(x)], 1)
    if kind == "envtan":
        return np.concatenate([envelopes(x), tangent_feats(x)], 1)
    if kind == "envd":
        e = envelopes(x)
        d = np.gradient(e, axis=0) * 5
        return np.concatenate([e, d], 1)
    if kind == "envraw":
        return np.concatenate([envelopes(x), raw_stack(x)], 1)
    if kind == "envcorr":
        return np.concatenate([envelopes(x), corr_feats(x)], 1)
    if kind == "env":
        return envelopes(x)
    if kind == "frame":
        return frame_feats(x)
    if kind == "envframe":
        return np.concatenate([envelopes(x), frame_feats(x)], 1)
    raise ValueError(kind)
