import io
import tempfile
import unittest
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from common import ACTIONS
from export_model import export
from model_adapter import COLS, MAX_SAMPLES, MIN_SAMPLES
from server import MAX_BYTES, create_app

HEADERS = {"Content-Type": "application/octet-stream", "X-Filename": quote("запись.npy")}


def npy(x):
    buf = io.BytesIO()
    np.save(buf, x)
    return buf.getvalue()


def fixture(root):
    """Create private synthetic records for integration testing, not production."""
    train = root / "train"
    extra = root / "extra"
    (train / "signals").mkdir(parents=True)
    (train / "targets").mkdir()
    extra.mkdir()
    rng = np.random.default_rng(44)
    n = MIN_SAMPLES + 31
    t = np.arange(n) / 500
    for i, action in enumerate(ACTIONS):
        activity = 0.9 + 0.6 * np.sin((1 + i / 10) * t)
        x = (rng.normal(0, 1e-5, size=(n, 6)) * activity[:, None]).astype(np.float32)
        y = (np.sin(t[:, None] + np.arange(20)[None, :] / 5) * (0.1 + 0.05 * i)).astype(np.float32)
        np.save(train / "signals" / f"{action}.npy", x)
        np.save(train / "targets" / f"{action}.npy", y)
    pd.DataFrame({"recording_id": ACTIONS, "action": ACTIONS}).to_csv(train / "metadata.csv", index=False)
    x = rng.normal(0, 1e-5, size=(MIN_SAMPLES + 3, 6)).astype(np.float32)
    np.save(extra / "synthetic.npy", x)
    return x, extra


class ServedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(tmp.cleanup)
        cls.root = Path(tmp.name)
        cls.signal, extra = fixture(cls.root)
        cls.weights = cls.root / "model.pt"
        export(cls.root, cls.weights, extra, threads=1, steps=dict(clf=2, teacher=2, student=2))
        # The API process must not have access to training records.
        (cls.root / "train").rename(cls.root / "not-train")
        cls.client = TestClient(create_app(cls.weights))
        cls.client.__enter__()
        cls.addClassCleanup(cls.client.__exit__, None, None, None)

    def request(self, body, name="запись.npy"):
        return self.client.post("/api/predict", headers={**HEADERS, "X-Filename": quote(name)}, content=body)

    def parse_success(self, response):
        self.assertEqual(response.status_code, 200, response.text[:300])
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        df = pd.read_csv(io.BytesIO(response.content))
        self.assertEqual(list(df.columns), COLS)
        self.assertEqual(df.shape, (len(self.signal), 20))
        arr = df.to_numpy()
        self.assertTrue(np.isfinite(arr).all())
        self.assertLessEqual(abs(arr).max(), np.pi + 1e-6)
        self.assertGreater(np.ptp(arr[:, 0]), 0.0)
        return arr

    def test_html_and_checkpoint_ready(self):
        self.assertEqual(self.client.get("/healthz").json(), {"status": "ready"})
        self.assertIn("fetch('/api/predict'", self.client.get("/").text)

    def test_npy_and_csv_agree(self):
        a = self.parse_success(self.request(npy(self.signal)))
        buf = io.StringIO()
        np.savetxt(buf, self.signal, delimiter=";", header="ch1;ch2;ch3;ch4;ch5;ch6", comments="")
        b = self.parse_success(self.request(buf.getvalue().encode("utf-8-sig"), "запись.csv"))
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.abs(a - self.parse_success(self.request(npy(self.signal[:, ::-1].copy())))).max(), 0)

    def test_bad_inputs(self):
        bad = [npy(self.signal[:, :5]), npy(self.signal[:100]), npy(np.full_like(self.signal, np.nan)),
               npy(np.zeros_like(self.signal)), npy(self.signal.astype(">f4")), npy(self.signal)[:-1],
               npy(np.asfortranarray(self.signal)), npy(np.zeros((MAX_SAMPLES + 1, 6), dtype=np.float32)),
               b"not npy"]
        for body in bad:
            with self.subTest(length=len(body)):
                self.assertEqual(self.request(body).status_code, 422)
        self.assertEqual(self.request(b"NaN,1,2,3,4,5\n" + b"0,1,2,3,4,5\n" * MIN_SAMPLES,
                                      "bad.csv").status_code, 422)
        self.assertEqual(self.request(b"x", "bad.txt").status_code, 422)
        self.assertEqual(self.client.post("/api/predict", content=b"x").status_code, 415)

    def test_upload_limit(self):
        chunk = b"x" * (MAX_BYTES // 2)
        self.assertEqual(self.request(chunk * 2 + b"x").status_code, 413)
        self.assertEqual(self.client.post("/api/predict", headers=HEADERS,
                                         content=iter((chunk, chunk, b"x"))).status_code, 413)

    def test_missing_and_corrupt_weights(self):
        with TestClient(create_app(self.root / "missing.pt")) as client:
            self.assertEqual(client.get("/healthz").status_code, 503)
            self.assertEqual(client.post("/api/predict").status_code, 501)
        corrupt = self.root / "corrupt.pt"
        corrupt.write_bytes(b"corrupt")
        with TestClient(create_app(corrupt)) as client:
            self.assertEqual(client.get("/healthz").status_code, 503)
            self.assertEqual(client.post("/api/predict").status_code, 500)


if __name__ == "__main__":
    unittest.main()
