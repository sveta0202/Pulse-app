import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d


def rec_feats(x):
    x = np.asarray(x, np.float64) / 1e-5
    env = uniform_filter1d(np.abs(x), 100, axis=0)
    le = np.log(env + 1e-3)
    m = le.mean(0)
    f = [m - m.mean(), [m.mean()], le.std(0), np.percentile(le, 90, 0) - np.percentile(le, 10, 0)]
    f.append(np.corrcoef(le.T)[np.triu_indices(6, 1)])
    s = le.mean(1)
    s = s - s.mean()
    sd = s[::10]
    ac = np.correlate(sd, sd, "full")[len(sd) - 1:]
    ac /= ac[0]
    lags = np.array([25, 50, 65, 80, 100, 130, 200])
    f.append(ac[np.minimum(lags, len(ac) - 1)])
    fr, p = welch(x, fs=500, nperseg=min(512, len(x)), axis=0)
    lp = np.log(p + 1e-12)
    bands = [(5, 30), (30, 60), (60, 100), (100, 150), (150, 250)]
    bp = np.stack([lp[(fr >= a) & (fr < b)].mean(0) for a, b in bands])
    f.append((bp - bp.mean(0)).ravel())
    f.append(np.corrcoef(x.T)[np.triu_indices(6, 1)])
    return np.concatenate([np.ravel(v) for v in f])
