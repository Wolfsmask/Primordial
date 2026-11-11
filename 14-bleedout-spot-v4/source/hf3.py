"""v3 world heightfield: our mountain and the ledge as before, but the valley is a basin under a sea of cloud, with
stone pillars rising out of it and jagged ranges far off.  World coords: ledge at x = 10, z = 0; x = east, z = south.
The floating islands, the colossus and the arch are voxel objects (objects3.py), not part of this field."""
import math
import numpy as np
from lib import fbm

LEDGE_Y = 250
SEAT = (10.6, 1.45)                    # where he ends up sitting: the view is measured from here
CLOUD_LO, CLOUD_HI = 95.0, 188.0       # the cloud sea
TREE_Y = 330.0
FLOOR = 60.0


def ss(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def nz(X, Z, scale, seed, oct=4):
    return fbm(X, 0.0, Z, scale, seed, oct)


def ridged(X, Z, scale, seed, oct=5):
    return 1 - np.abs(2 * nz(X, Z, scale, seed, oct) - 1)


def at(yaw, dist):
    """world x, z of a point seen from the seat at this yaw (deg; 90 = west, + = to the left/south) and distance"""
    y = math.radians(yaw)
    return SEAT[0] - math.sin(y) * dist, SEAT[1] - math.cos(y) * dist


# ---- the pillars: (yaw, distance, radius, top) - a few placed by hand where the eye lands, the rest scattered
HERO_PILLARS = [(92, 175, 21, 233), (30, 250, 27, 287), (118, 300, 18, 214), (101, 430, 34, 312), (104, 640, 24, 262),
                (40, 560, 40, 372), (128, 520, 30, 300), (108, 820, 46, 404), (66, 1050, 38, 212), (116, 1250, 55, 452),
                (30, 950, 30, 270), (140, 1300, 34, 340), (96, 1500, 60, 380), (50, 1600, 44, 470), (76, 1250, 26, 196)]
CORRIDOR = (60.0, 94.0)                # keep the view of the colossus' floating underside clear


def _scatter():
    rng = np.random.default_rng(31)
    out = list(HERO_PILLARS)
    pts = [at(a, d) for a, d, _, _ in out]
    tries = 0
    while len(out) < 120 and tries < 20000:
        tries += 1
        yaw = rng.uniform(-10, 200); dist = 300 + 4200 * rng.random() ** 1.6
        x, z = at(yaw, dist)
        r = rng.uniform(14, 34) * (1 + dist / 2500)
        if any(math.hypot(x - px, z - pz) < r + 60 + 0.04 * dist for px, pz in pts):
            continue
        cx, cz = at(76, 2300)                                         # keep the colossus' footprint clear
        if math.hypot(x - cx, z - cz) < 700:
            continue
        top = rng.uniform(205, 330) + rng.random() ** 3 * 230 * min(1, dist / 1500)
        if CORRIDOR[0] < yaw < CORRIDOR[1]:
            top = min(top, 245 - 0.035 * dist)
        out.append((yaw, dist, r, top)); pts.append((x, z))
    return out


PILLARS = _scatter()


def pillars(X, Z):
    H = np.full(np.shape(X), -1e9)
    for i, (yaw, dist, r, top) in enumerate(PILLARS):
        cx, cz = at(yaw, dist)
        dx, dz = X - cx, Z - cz
        m = (np.abs(dx) < r * 1.9 + 40) & (np.abs(dz) < r * 1.9 + 40)
        if not m.any():
            continue
        ang = np.arctan2(dz[m], dx[m])
        d = np.hypot(dx[m], dz[m])
        rr = r * (1 + 0.16 * np.sin(3 * ang + i) + 0.10 * np.sin(5 * ang + 2 * i) + 0.12 * (nz(X[m], Z[m], 9, 100 + i, 2) - 0.5))
        crown = top + 6 * (nz(X[m], Z[m], 14, 200 + i, 3) - 0.5) - 3 * (d / rr) ** 2
        flare = rr * (1 + 0.55 * ss((CLOUD_HI - 40 - (crown - (d - rr) * 9)) / 120))     # a wider foot down in the cloud
        h = np.where(d < rr, crown, crown - (d - rr) * 11)
        h = np.where((d >= rr) & (d < flare), np.maximum(h, CLOUD_HI - 30 - (d - rr) * 1.5), h)
        H[m] = np.maximum(H[m], h)
    return H


def height(X, Z):
    X = np.asarray(X, float); Z = np.asarray(Z, float)
    H = FLOOR + 14 * (nz(X, Z, 160, 1) - 0.5) + 6 * (nz(X, Z, 40, 2) - 0.5)
    # our mountain: an elongated massif east of the ledge, its west face steep; the ledge sits just under the crest
    rr_ = np.hypot(X - 92, Z * 0.52) + 40 * (nz(X, Z, 120, 41) - 0.5)
    mm = np.clip(1 - rr_ / 340, 0, 1)
    m1 = 68 + 235 * mm ** 0.85 * (0.82 + 0.18 * ridged(X, Z, 60, 8))
    xc = -6 + 10 * (nz(0 * X, Z, 90, 7) - 0.5) + 4 * (nz(X, Z, 8, 51) - 0.5) * (1 - 0.8 * ss(1 - np.abs(Z) / 18))
    m1 += 22 * ss((X - xc) / 26) * ss((240 - np.abs(Z)) / 120) * (mm > 0.4)
    m1 += 7 * (nz(X, Z, 6, 52) - 0.5) * (mm > 0.2)
    # the ledge is the lip of a sheer west face: everything west of it drops straight into the cloud sea
    edge = 7.5 + 3.0 * (nz(0 * X, Z, 40, 71) - 0.5) * ss(np.abs(Z) / 12) + 0.00004 * Z ** 2
    cliff = ss(np.abs(Z) / 500)
    m1 = np.where(X < edge, np.minimum(m1, 247 - np.clip(edge - X, 0, None) ** 1.15 * 5.0 * (1 - cliff)), m1)
    # and a saddle south of the ledge, so the view to the south-west (the sun, the arch) is open
    sad = (X > 6) & (X < 13) & (Z > 4) & (Z < 140)
    m1 = np.where(sad, np.minimum(m1, 248.5 - np.clip(Z - 4, 0, None) * 0.35), m1)
    H = np.maximum(H, m1)
    # far ranges: sharp, unreal spires and walls all round, lower in the direction of the colossus
    d = np.hypot(X, Z)
    ang = np.arctan2(Z, X)
    ga = math.atan2(-555.0, -2220.0)
    dang = (((ang - ga + math.pi) % (2 * math.pi)) - math.pi)
    gap = 1 - 0.75 * np.exp(-(dang / 0.45) ** 2)
    rid = ridged(X, Z, 900, 31, 5)
    spikes = np.clip(ridged(X, Z, 260, 33, 3) - 0.55, 0, None) * 2.2
    mr = FLOOR + (900 * rid ** 3.0 + 700 * spikes * rid) * ss((d - 5200) / 2600) * gap
    H = np.maximum(H, mr)
    return H


def forest(X, Z, H):
    f = 0.25 + 0.7 * ss((nz(X, Z, 70, 20) - 0.45) / 0.2)
    return f * (H < TREE_Y) * (1 - ss((H - (TREE_Y - 60)) / 60))
WATER_Y = 0                            # no open water in v3 (the valley is under the clouds)
