"""Smooth value-noise fBm on numpy arrays (rebuilt: the original helper was lost with the old container)."""
import numpy as np


def _hash(ix, iy, iz, seed):
    h = (ix * 374761393 + iy * 668265263 + iz * 2147483647 + seed * 144269504) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h & 0xFFFFFF) / float(0xFFFFFF)


def vnoise(x, y, z, seed):
    x = np.asarray(x, float); y = np.asarray(y, float) + np.zeros_like(x); z = np.asarray(z, float) + np.zeros_like(x)
    ix, iy, iz = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64), np.floor(z).astype(np.int64)
    fx, fy, fz = x - ix, y - iy, z - iz
    sx, sy, sz = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy), fz * fz * (3 - 2 * fz)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (sx if dx else 1 - sx) * (sy if dy else 1 - sy) * (sz if dz else 1 - sz)
                out = out + w * _hash(ix + dx, iy + dy, iz + dz, seed)
    return out


def fbm(x, y, z, scale, seed, octaves=4):
    x = np.asarray(x, float) / scale; y = np.asarray(y, float) / scale; z = np.asarray(z, float) / scale
    tot, amp, norm = 0.0, 1.0, 0.0
    for o in range(octaves):
        tot = tot + amp * vnoise(x * 2 ** o + 17.3 * o, y * 2 ** o, z * 2 ** o - 9.1 * o, seed * 31 + o)
        norm += amp; amp *= 0.5
    return tot / norm
