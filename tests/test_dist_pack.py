"""Numpy-dependent tests for the dist packing math. Skipped on the stdlib CI
matrix (no numpy); run in the build image / validate job."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(os.path.dirname(HERE), "src")
sys.path.insert(0, SRC)

try:
    import numpy as np
    import build_dist
    import reduce_layout
    HAVE = True
except Exception:  # noqa: BLE001
    HAVE = False


@unittest.skipUnless(HAVE, "numpy not installed")
class Quantize(unittest.TestCase):
    def test_int8_roundtrip_cosine(self):
        rng = np.random.default_rng(0)
        v = rng.standard_normal((50, 128)).astype(np.float32)
        v = reduce_layout.l2norm(v)
        q, scale = build_dist.quantize_i8(v)
        self.assertEqual(q.dtype, np.int8)
        # dequantized cosine ~= true cosine to ~1e-2
        a, b = 0, 1
        true = float(v[a] @ v[b])
        approx = float((q[a].astype(np.int32) @ q[b].astype(np.int32)) * scale[a] * scale[b])
        self.assertLess(abs(true - approx), 2e-2)

    def test_neighbor_pack_is_3_bytes(self):
        nidx = np.zeros((4, 15), dtype=np.uint16)
        nsim = np.zeros((4, 15), dtype=np.uint8)
        nidx[0, 0] = 513
        nsim[0, 0] = 200
        blob = build_dist.pack_neighbors(nidx, nsim)
        self.assertEqual(len(blob), 4 * 15 * 3)
        # first record little-endian u16 then u8
        self.assertEqual(blob[0], 513 & 0xFF)
        self.assertEqual(blob[1], 513 >> 8)
        self.assertEqual(blob[2], 200)


@unittest.skipUnless(HAVE, "numpy not installed")
class Layout(unittest.TestCase):
    def test_normalize_box_within_unit(self):
        rng = np.random.default_rng(1)
        P = rng.standard_normal((100, 2)) * 5 + 3
        Pn, c, s = reduce_layout.normalize_box(P)
        self.assertLessEqual(float(np.abs(Pn).max()), 1.0 + 1e-6)

    def test_procrustes_recovers_rotation(self):
        rng = np.random.default_rng(2)
        Y = rng.standard_normal((80, 2))
        theta = 0.7
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        X = Y @ R.T * 1.5 + 4.0
        Xa = reduce_layout.procrustes_onto(X, Y)
        self.assertLess(float(np.linalg.norm(Xa - Y) / np.linalg.norm(Y)), 1e-6)


if __name__ == "__main__":
    unittest.main()
