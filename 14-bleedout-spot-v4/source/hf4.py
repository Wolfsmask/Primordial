"""v4 world heightfield: a real valley at the end of golden hour.  Our mountain and the ledge as in v3 (a sheer drop
west of the ledge), a valley floor with a lake below and rivers fed by waterfalls, forested foothills, a jagged snowy
range across the valley whose hanging valleys pour huge waterfalls down its cliffs, a low gap where the sun sets, and
taller ranges far behind.  World coords: ledge at x = 10, z = 0; x = east, z = south."""
import math
import numpy as np
from lib import fbm

LEDGE_Y = 250
SEAT = (10.6, 1.45)
WATER_Y = 64
TREE_Y = 360.0
FLOOR = 70.0
CLOUD_LO, CLOUD_HI = 64.0, 118.0      # the valley mist layer (renderer)


def ss(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def nz(X, Z, scale, seed, oct=4):
    return fbm(X, 0.0, Z, scale, seed, oct)


def ridged(X, Z, scale, seed, oct=5):
    return 1 - np.abs(2 * nz(X, Z, scale, seed, oct) - 1)


def at(yaw, dist):
    y = math.radians(yaw)
    return SEAT[0] - math.sin(y) * dist, SEAT[1] - math.cos(y) * dist


def polar(X, Z):
    dx, dz = X - SEAT[0], Z - SEAT[1]
    yaw = np.degrees(np.arctan2(-dx, -dz)) % 360
    return np.where(yaw > 270, yaw - 360, yaw), np.hypot(dx, dz)


LAKE = at(93, 560)
LAKE_R = (300.0, 190.0)
# the big waterfalls: (yaw, distance, lip height, width) - each lip is the mouth of a hanging valley on the range's face
FALLS = [(84, 2250, 520, 80), (71, 2700, 600, 52), (100, 2050, 430, 44), (55, 2250, 470, 34), (124, 2350, 420, 28)]
# where the sun goes down: a low saddle in the range
SUN_GAP = 150.0


def _curve(pts, n=400):
    P = np.vstack([pts[:1], pts, pts[-1:]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, n // (len(P) - 3), endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2])
    return np.array(out)


VALLEY = _curve(np.array([(-1150, -4600), (-850, -2400), (-600, -1100), (-430, -250), LAKE, (-420, 700), (-700, 1900),
                          (-1150, 3300), (-1600, 4800)], float), n=3000)
from scipy.spatial import cKDTree
_VT = cKDTree(VALLEY)


def valley(X, Z):
    """distance from the valley's centre line, and the side (+ = west, the far wall)"""
    P = np.stack([np.ravel(X), np.ravel(Z)], 1)
    d, i = _VT.query(P)
    i0 = np.clip(i - 3, 0, len(VALLEY) - 1); i1 = np.clip(i + 3, 0, len(VALLEY) - 1)
    t = VALLEY[i1] - VALLEY[i0]; v = P - VALLEY[i]
    side = np.sign(t[:, 0] * v[:, 1] - t[:, 1] * v[:, 0])
    return d.reshape(np.shape(X)), side.reshape(np.shape(X))


def _rivers():
    rng = np.random.default_rng(3)
    out = [(VALLEY, 13.0)]                                  # the main river runs down the valley through the lake
    for yaw, dist, lip, w in FALLS:
        px, pz = at(yaw, dist)
        ux, uz = (SEAT[0] - px) / dist, (SEAT[1] - pz) / dist
        pool = np.array((px + ux * 40, pz + uz * 40))
        _, j = _VT.query(pool)
        j = min(len(VALLEY) - 1, j + int(rng.integers(40, 160)))          # join downstream
        tgt = VALLEY[j]
        mid = pool + (tgt - pool) * 0.5 + np.array([-(tgt - pool)[1], (tgt - pool)[0]]) * rng.uniform(-0.18, 0.18)
        out.append((_curve(np.array([pool, pool + (tgt - pool) * 0.2, mid, tgt], float), n=200), 5 + w * 0.16))
    return out


RIVERS = _rivers()
_RT = [(cKDTree(c), rw) for c, rw in RIVERS]


def river_carve(X, Z, H):
    P = np.stack([np.ravel(X), np.ravel(Z)], 1)
    for tree, rw in _RT:
        d, _ = tree.query(P, distance_upper_bound=rw * 1.4 + 40)
        d = d.reshape(np.shape(X))
        m = np.isfinite(d)
        if not m.any():
            continue
        dm = d[m]; w = rw * (0.8 + 0.4 * nz(X[m], Z[m], 90, 77, 2))
        h = H[m]
        h = np.where(dm < w, np.minimum(h, WATER_Y - 2 + dm / w), h)
        h = np.where((dm >= w) & (dm < w + 30), np.minimum(h, WATER_Y + 1.2 + (dm - w) * 0.35), h)
        H[m] = h
    return H


def falls_carve(X, Z, H):
    """each waterfall: a hanging valley behind the lip at lip height, a sheer cliff, an amphitheatre and a pool below"""
    for i, (yaw, dist, lip, w) in enumerate(FALLS):
        px, pz = at(yaw, dist)
        ux, uz = (SEAT[0] - px) / dist, (SEAT[1] - pz) / dist
        dx, dz = X - px, Z - pz
        along = dx * ux + dz * uz                     # + toward the seat (in front of the cliff)
        perp = -dx * uz + dz * ux
        wob = 30 * (nz(X, Z, 60, 300 + i, 2) - 0.5)
        behind = (along < 0) & (along > -380) & (np.abs(perp) < 170 + wob)
        plateau = lip + 6 + 0.06 * (-along) + 10 * (nz(X, Z, 40, 310 + i, 2) - 0.5)
        walls = plateau + np.clip(np.abs(perp) - 45, 0, None) * 1.6
        H = np.where(behind, np.clip(H, plateau, walls), H)
        chan = behind & (np.abs(perp + 25 * np.sin(-along / 70 + i)) < w * 0.5)
        H = np.where(chan, np.minimum(H, lip - 1.5), H)
        front = (along >= 0) & (along < 260) & (np.abs(perp) < 120 + along * 0.6 + wob)
        floor = FLOOR + 4 + along * 0.05 + 6 * (nz(X, Z, 30, 320 + i, 2) - 0.5)
        H = np.where(front, np.minimum(H, floor), H)
        # the plunge pool
        pd = np.hypot(dx - ux * 40, dz - uz * 40)
        H = np.where(pd < 28 + w * 0.4, np.minimum(H, WATER_Y - 3), H)
    return H


def height(X, Z):
    X = np.asarray(X, float); Z = np.asarray(Z, float)
    yaw, dist = polar(X, Z)
    dv, side = valley(X, Z)
    # the valley floor: meadows, low hills, foothills toward the walls
    H = FLOOR + 10 * (nz(X, Z, 220, 1) - 0.5) + 5 * (nz(X, Z, 50, 2) - 0.5)
    H += 120 * ss((dv - 280) / 500) * nz(X, Z, 300, 3, 4) ** 1.8
    # the walls: the far (west) one a jagged snowy range, the near (east) one our side of the valley
    r1 = ridged(X, Z, 780, 31, 5); r2 = ridged(X, Z, 290, 32, 4); r3 = ridged(X, Z, 105, 33, 3)
    rise_w = ss((dv - 620) / 1500) ** 1.2
    west = FLOOR + rise_w * (260 + 880 * r1 ** 2.0 * (0.55 + 0.45 * r2)) + 170 * np.clip(r3 - 0.6, 0, None) * r1 * rise_w
    rise_e = ss((dv - 700) / 1300)
    east = FLOOR + rise_e * (220 + 520 * ridged(X, Z, 600, 37, 4) ** 1.6)
    H = np.maximum(H, np.where(side > 0, west, east))
    # our mountain and the ledge (as v3)
    rr_ = np.hypot(X - 92, Z * 0.52) + 40 * (nz(X, Z, 120, 41) - 0.5)
    mm = np.clip(1 - rr_ / 340, 0, 1)
    m1 = 68 + 235 * mm ** 0.85 * (0.82 + 0.18 * ridged(X, Z, 60, 8))
    xc = -6 + 10 * (nz(0 * X, Z, 90, 7) - 0.5) + 4 * (nz(X, Z, 8, 51) - 0.5) * (1 - 0.8 * ss(1 - np.abs(Z) / 18))
    m1 += 22 * ss((X - xc) / 26) * ss((240 - np.abs(Z)) / 120) * (mm > 0.4)
    m1 += 7 * (nz(X, Z, 6, 52) - 0.5) * (mm > 0.2)
    edge = 7.5 + 3.0 * (nz(0 * X, Z, 40, 71) - 0.5) * ss(np.abs(Z) / 12) + 0.00004 * Z ** 2
    cliff = ss(np.abs(Z) / 500)
    m1 = np.where(X < edge, np.minimum(m1, 247 - np.clip(edge - X, 0, None) ** 1.15 * 5.0 * (1 - cliff)), m1)
    sad = (X > 6) & (X < 13) & (Z > 4) & (Z < 140)
    m1 = np.where(sad, np.minimum(m1, 248.5 - np.clip(Z - 4, 0, None) * 0.35), m1)
    H = np.maximum(H, m1)
    gap = 1 - 0.75 * np.exp(-((yaw - SUN_GAP) / 10.0) ** 2)
    # far ranges: taller, sharper, hazy
    far = FLOOR + (1100 * ridged(X, Z, 1500, 35, 5) ** 2.4 + 500 * np.clip(ridged(X, Z, 420, 36, 3) - 0.5, 0, None)) * ss((dist - 5200) / 2500) * gap
    H = np.maximum(H, far)
    # water: the lake under the ledge, the rivers, the falls
    H = falls_carve(X, Z, H)
    e = ((X - LAKE[0]) / LAKE_R[0]) ** 2 + ((Z - LAKE[1]) / LAKE_R[1]) ** 2 + 0.45 * (nz(X, Z, 40, 9) - 0.5)
    H = np.where(e < 1, np.minimum(H, WATER_Y - 2 - 9 * (1 - np.clip(e, 0, 1)) ** 0.6), H)
    H = np.where((e >= 1) & (e < 1.35), np.minimum(H, WATER_Y + (e - 1) * 16), H)
    H = river_carve(X, Z, H)
    return H


def forest(X, Z, H):
    f = 0.30 + 0.65 * ss((nz(X, Z, 90, 20) - 0.42) / 0.22)
    return f * (H < TREE_Y) * (1 - ss((H - (TREE_Y - 80)) / 80)) * (H > WATER_Y + 1)


def cherry_grove(X, Z):
    """pink groves: a few clumps near the lake and scattered clearings - never a ring"""
    lx, lz = LAKE
    e = np.hypot((X - lx) / LAKE_R[0], (Z - lz) / LAKE_R[1])
    near = np.exp(-((e - 1.6) / 0.6) ** 2)
    return np.clip(near * (nz(X, Z, 70, 21) > 0.62) + (nz(X, Z, 180, 22) > 0.74) * 0.8, 0, 1)


def channel_level(X, Z):
    """water level of the streams in the hanging valleys above the falls (-inf elsewhere)"""
    lvl = np.full(np.shape(X), -np.inf)
    for i, (yaw, dist, lip, w) in enumerate(FALLS):
        px, pz = at(yaw, dist)
        ux, uz = (SEAT[0] - px) / dist, (SEAT[1] - pz) / dist
        dx, dz = X - px, Z - pz
        along = dx * ux + dz * uz; perp = -dx * uz + dz * ux
        chan = (along < 2) & (along > -380) & (np.abs(perp + 25 * np.sin(-along / 70 + i)) < w * 0.5 - 1)
        lvl = np.where(chan, lip - 0.6, lvl)
    return lvl
