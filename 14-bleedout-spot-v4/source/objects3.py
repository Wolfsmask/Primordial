"""The impossible things in the view, as coarse voxel objects (vertex-coloured): floating islands with groves and
waterfalls, the colossus (a whole mountain hanging in the air over the cloud sea), a giant stone arch standing out of
the clouds, and loose rocks drifting round them.  Writes web/obj_*.bin, web/waterfalls.json and adds to web/meta.json."""
import json, math, os, time
import numpy as np
import hf3
from lib import fbm
import mesher
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
at = hf3.at


def srgb(c):
    return (np.array(c, np.float32) / 255) ** 2.2


GRASS, DIRT, STONE_A, STONE_B, STONE_C, MOSS, SNOW, CHERRY, OAK, LOG, DARK, WATER, PALE = range(1, 14)
PAL = np.zeros((256, 3), np.float32)
for k, c in {GRASS: (98, 152, 62), DIRT: (122, 90, 62), STONE_A: (158, 148, 136), STONE_B: (126, 118, 110), STONE_C: (178, 152, 126),
             MOSS: (86, 124, 58), SNOW: (242, 246, 252), CHERRY: (244, 168, 206), OAK: (66, 124, 46), LOG: (92, 64, 46),
             DARK: (96, 88, 88), WATER: (150, 196, 230), PALE: (214, 200, 220)}.items():
    PAL[k] = srgb(c)
ROCK1, ROCK2, ROCK3, ROCK4, ROCK5, MOSS2, VINE = range(14, 21)
for k, c in {ROCK1: (178, 166, 150), ROCK2: (154, 144, 132), ROCK3: (198, 186, 168), ROCK4: (134, 126, 120), ROCK5: (168, 152, 140),
             MOSS2: (70, 100, 50), VINE: (60, 102, 46)}.items():
    PAL[k] = srgb(c)
ROCKS = np.array([ROCK4, ROCK2, ROCK5, ROCK1, ROCK3], np.uint8)


def rock_ids(X, Y, Z, seed):
    """natural mottled rock: soft patches from 3D noise, no bands"""
    n1 = fbm(X, Y * 0.7, Z, 16, seed + 900, 3)
    n2 = fbm(X, Y, Z, 4.5, seed + 901, 2)
    v = np.clip((n1 - 0.5) * 2.6 + 0.5 + 0.45 * (n2 - 0.5), 0, 0.999)
    return ROCKS[(v * len(ROCKS)).astype(int)]
WATERFALLS = []


def put_tree(vol, x, y, z, kind, s, rng):
    """a tree at voxel (x, y, z) sized for voxel scale s metres"""
    h = max(2, int(round((7 + 5 * rng.random()) / s)))
    leaf = CHERRY if kind == 'cherry' else OAK
    for k in range(h):
        if 0 <= y + k < vol.shape[1]:
            vol[x, y + k, z] = LOG
    r = max(1.2, (3.2 + 1.8 * rng.random()) / s)
    cy = y + h
    R = int(math.ceil(r)) + 1
    for dx in range(-R, R + 1):
        for dy in range(-R, R + 1):
            for dz in range(-R, R + 1):
                if (dx / r) ** 2 + (dy / (r * 0.75)) ** 2 + (dz / r) ** 2 < 1 and rng.random() > 0.08:
                    X, Y, Z = x + dx, cy + dy, z + dz
                    if 0 <= X < vol.shape[0] and 0 <= Y < vol.shape[1] and 0 <= Z < vol.shape[2] and vol[X, Y, Z] == 0:
                        vol[X, Y, Z] = leaf


def island(cx, cz, top, r, seed, s, depth=None, peak=0.0, cherry=0.5, falls=0, trees=0.035, snow_y=1e9, jagged=False):
    """returns (vol, origin) for a floating island: a hilly top, an inverted rocky cone below, trees, waterfalls"""
    rng = np.random.default_rng(seed)
    depth = depth if depth is not None else r * 1.4
    R = r * 1.25
    nx = int(2 * R / s) + 6; ny = int((depth + peak + 40 + 30) / s) + 6
    x0, z0 = cx - nx * s / 2, cz - nx * s / 2
    y0 = top - depth - 10
    xs = x0 + (np.arange(nx) + 0.5) * s; zs = z0 + (np.arange(nx) + 0.5) * s
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    dx, dz = X - cx, Z - cz
    ang = np.arctan2(dz, dx); d = np.hypot(dx, dz)
    rr = r * (1 + 0.14 * np.sin(3 * ang + seed) + 0.09 * np.sin(7 * ang + 2 * seed) + 0.18 * (fbm(X, 0.0, Z, r * 0.45, seed, 3) - 0.5))
    inside = d < rr
    u = np.clip(d / rr, 0, 1)
    if jagged:                          # a whole range on top: ridged peaks, the highest off-centre, cliffs and couloirs
        rid = 1 - np.abs(2 * fbm(X, 0.0, Z, r * 0.42, seed + 7, 5) - 1)
        rid2 = 1 - np.abs(2 * fbm(X, 0.0, Z, r * 0.16, seed + 8, 4) - 1)
        mass = np.clip(1 - u / 0.85, 0, 1) ** 0.9
        core = np.exp(-((dx + 0.18 * r) ** 2 + (dz - 0.1 * r) ** 2) / (2 * (0.28 * r) ** 2))
        prof = mass * (0.25 + 0.75 * rid ** 2.2) * (0.55 + 0.45 * rid2) + 0.9 * core * rid ** 1.2
        surf = top + 4 * (fbm(X, 0.0, Z, 30, seed + 1, 3) - 0.5) + peak * prof - 3 * u ** 3
    else:
        surf = top + 3.5 * (fbm(X, 0.0, Z, max(8, r * 0.3), seed + 1, 3) - 0.5) * (1 + r / 60) + peak * np.clip(1 - u / 0.55, 0, 1) ** 1.6 \
            * (0.75 + 0.5 * fbm(X, 0.0, Z, max(20, r * 0.25), seed + 7, 4)) - 3 * u ** 3
    under = top - depth * (1 - u ** 1.5) * (0.65 + 0.7 * fbm(X, 0.0, Z, max(6, r * 0.18), seed + 2, 4)) - 2
    under -= depth * 0.35 * np.clip(fbm(X, 0.0, Z, max(4, r * 0.07), seed + 3, 2) - 0.62, 0, 1) * 3 * (1 - u)   # stalactites
    vol = np.zeros((nx, ny, nx), np.uint8)
    ys = y0 + (np.arange(ny) + 0.5) * s
    Y = ys[None, :, None]
    solid = inside[:, None, :] & (Y <= surf[:, None, :]) & (Y >= under[:, None, :])
    depth_below = surf[:, None, :] - Y
    rock = rock_ids(X[:, None, :], np.broadcast_to(Y, (1, ny, 1)), Z[:, None, :], seed)
    vol[solid] = np.broadcast_to(rock, solid.shape)[solid]
    top_layer = solid & (depth_below < s * 1.01)
    dirt = solid & (depth_below >= s * 1.01) & (depth_below < 3 * s + 2)
    vol[dirt] = DIRT
    steep = np.zeros_like(surf, bool)
    gx = np.abs(np.gradient(surf, axis=0)) + np.abs(np.gradient(surf, axis=1))
    steep = gx > 2.2 * s
    tl = top_layer & ~steep[:, None, :]
    vol[tl] = GRASS
    vol[top_layer & steep[:, None, :]] = STONE_B
    vol[solid & (Y > snow_y + 25 * (fbm(X, 0.0, Z, 60, seed + 9, 3)[:, None, :] - 0.5)) & (depth_below < 3 * s)] = SNOW
    # moss and dark rock on the underside
    low = solid & (Y < under[:, None, :] + 2.5 * s)
    vol[low] = np.where(rng.random(low.sum()) < 0.35, MOSS, DARK)
    # trees on the top
    ii, kk = np.nonzero(inside & ~steep & (surf < snow_y - 30))
    for i, k in zip(ii, kk):
        if rng.random() < trees:
            yy = int((surf[i, k] - y0) / s) + 1
            if 0 < yy < ny - 6:
                put_tree(vol, i, yy, k, 'cherry' if rng.random() < cherry else 'oak', s, rng)
    # waterfalls: from rim points that face the ledge, with a channel cut back into the top
    if falls:
        sx, sz = hf3.SEAT
        to_seat = math.atan2(sz - cz, sx - cx)
        for f in range(falls):
            a = to_seat + (f - (falls - 1) / 2) * 0.55 + rng.uniform(-0.12, 0.12)
            rim = None
            for t in np.linspace(0.6, 1.3, 60):
                px, pz = cx + math.cos(a) * r * t, cz + math.sin(a) * r * t
                i, k = int((px - x0) / s), int((pz - z0) / s)
                if 0 <= i < nx and 0 <= k < nx and inside[i, k]:
                    rim = (i, k)
            if rim is None:
                continue
            i, k = rim
            w = max(1, int(round((5 + 0.12 * r) / s)))
            # a channel back from the lip, filled with water
            for t in range(0, max(3, int(r * 0.45 / s))):
                ci = int(round(i - math.cos(a) * t)); ck = int(round(k - math.sin(a) * t))
                for j in range(-w, w + 1):
                    qi = int(round(ci - math.sin(a) * j)); qk = int(round(ck + math.cos(a) * j))
                    if 0 <= qi < nx and 0 <= qk < nx and inside[qi, qk]:
                        ytop = int((surf[qi, qk] - y0) / s)
                        if 0 < ytop < ny:
                            vol[qi, ytop, qk] = WATER; vol[qi, ytop + 1:, qk] = np.where(vol[qi, ytop + 1:, qk] == LOG, 0, vol[qi, ytop + 1:, qk])
            lip_y = surf[i, k]
            WATERFALLS.append(dict(x=float(x0 + (i + 0.5) * s + math.cos(a) * s), z=float(z0 + (k + 0.5) * s + math.sin(a) * s),
                                   y=float(lip_y), bottom=float(hf3.CLOUD_HI - 12), width=float((2 * w + 1) * s * 0.9),
                                   dir=[math.cos(a), math.sin(a)], seed=int(seed * 10 + f)))
    return vol, (x0, y0, z0)


def arch(cx, cz, yaw_face, span, height, thick, deep, s, seed):
    """a natural stone arch standing out of the cloud sea, its opening facing the seat"""
    rng = np.random.default_rng(seed)
    a = math.radians(yaw_face)
    sdir = np.array([math.cos(a), -math.sin(a)])          # along the span (perpendicular to the line of sight)
    vdir = np.array([-math.sin(a), -math.cos(a)])         # along the line of sight
    nx = int((span + 2 * thick + 60) / s); ny = int((height + thick + 60) / s); nz = int((deep + 60) / s)
    base = hf3.CLOUD_LO
    vol = np.zeros((nx, ny, nz), np.uint8)
    lx = (np.arange(nx) - nx / 2 + 0.5) * s; ly = base + (np.arange(ny) + 0.5) * s; lz = (np.arange(nz) - nz / 2 + 0.5) * s
    LX, LY, LZ = np.meshgrid(lx, ly, lz, indexing='ij')
    # an elliptical band: inner ellipse span/2 x (height - thick), outer adds the thickness; wobbly with noise
    nzf = fbm(LX, LY, LZ, 30, seed, 3) - 0.5
    e_out = (LX / (span / 2 + thick)) ** 2 + ((LY - base) / (height + thick * 0.3)) ** 2
    e_in = (LX / (span / 2)) ** 2 + ((LY - base) / (height - thick)) ** 2
    dd = (LZ / (deep / 2 * (1 + 0.25 * np.sin(LY / 37 + seed)))) ** 2
    solid = (e_out < 1 + 0.25 * nzf) & (e_in > 1 + 0.2 * nzf) & (dd < 1 + 0.4 * nzf) & (LY > base)
    vol[solid] = rock_ids(LX + cx, LY, LZ + cz, seed)[solid]
    above = np.zeros_like(solid); above[:, :-1, :] = solid[:, 1:, :]
    topf = solid & ~above
    vol[topf & (LY > base + height * 0.55)] = GRASS
    hang = solid & (rng.random(solid.shape) < 0.04)
    vol[hang & (LY < base + height * 0.9)] = MOSS
    # place it in the world: rotate the local frame (x along span, z along sight) - voxelise by sampling
    return vol, sdir, vdir, (cx, base, cz)


def place_rotated(vol_local, sdir, vdir, centre, s):
    """re-voxelise a locally built object into a world-aligned grid (nearest sample)"""
    nx, ny, nz = vol_local.shape
    cx, base, cz = centre
    hx, hz = nx * s / 2, nz * s / 2
    R = math.hypot(hx, hz) + s
    n = int(2 * R / s) + 2
    x0, z0 = cx - n * s / 2, cz - n * s / 2
    gx = x0 + (np.arange(n) + 0.5) * s; gz = z0 + (np.arange(n) + 0.5) * s
    GX, GZ = np.meshgrid(gx, gz, indexing='ij')
    lx = (GX - cx) * sdir[0] + (GZ - cz) * sdir[1]; lz = (GX - cx) * vdir[0] + (GZ - cz) * vdir[1]
    i = np.round(lx / s + nx / 2 - 0.5).astype(int); k = np.round(lz / s + nz / 2 - 0.5).astype(int)
    ok = (i >= 0) & (i < nx) & (k >= 0) & (k < nz)
    tmp = np.zeros((n, n, ny), np.uint8)
    tmp[ok] = vol_local[i[ok], :, k[ok]]
    return np.ascontiguousarray(tmp.transpose(0, 2, 1)), (x0, base, z0)


def pillar(cx, cz, r, top, seed, s, trees=0.05):
    """a rock spire out of the cloud sea: tapering, leaning a little, lobed and fluted by erosion, with ledges,
    sometimes forking near the top; grass and groves on top and on the ledges, moss on the shade side, vines"""
    rng = np.random.default_rng(seed)
    base = hf3.CLOUD_LO + 2
    Hh = top - base
    lean = rng.uniform(0.0, 0.07) * Hh; la = rng.uniform(0, 2 * math.pi)
    fork = rng.random() < 0.3 and r > 16
    Rm = r * 1.4 + lean + 10
    nx = int(2 * Rm / s) + 4; ny = int((Hh + 34) / s) + 4
    x0, z0 = cx - nx * s / 2, cz - nx * s / 2
    xs = x0 + (np.arange(nx) + 0.5) * s; ys = base + (np.arange(ny) + 0.5) * s; zs = z0 + (np.arange(nx) + 0.5) * s
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    h = (Y - base) / Hh
    wob = r * 0.12
    ccx = cx + math.cos(la) * lean * h + wob * (fbm(0 * X + 3, Y, 0 * Z, 45, seed + 1, 2) - 0.5)
    ccz = cz + math.sin(la) * lean * h + wob * (fbm(0 * X, Y, 0 * Z + 3, 45, seed + 2, 2) - 0.5)
    dx, dz = X - ccx, Z - ccz
    d = np.hypot(dx, dz)
    taper = 1.18 - 0.30 * np.clip(h, 0, 1)
    bulge = 0.0
    for hl in rng.uniform(0.3, 0.88, int(rng.integers(1, 4))):
        bulge = bulge + 0.17 * np.exp(-((h - hl) / 0.022) ** 2) * (fbm(X, 0 * Y, Z, r * 0.5, seed + int(hl * 100), 2) > 0.45)
    lobes = 0.32 * (fbm(X, Y * 0.5, Z, max(6, r * 0.7), seed + 3, 3) - 0.5)
    flutes = 0.16 * (fbm(X, Y * 0.12, Z, max(2.5, r * 0.13), seed + 4, 2) - 0.5)
    Reff = r * (taper + lobes + flutes + bulge)
    topS = top + 3.5 * (fbm(X, 0 * Y, Z, 9, seed + 5, 2) - 0.5) - 6 * (d / r) ** 2
    if fork:
        hs = rng.uniform(0.55, 0.78); a2 = rng.uniform(0, 2 * math.pi); off = r * 0.42
        above = h > hs
        c1x, c1z = ccx + math.cos(a2) * off, ccz + math.sin(a2) * off
        c2x, c2z = ccx - math.cos(a2) * off, ccz - math.sin(a2) * off
        d1, d2 = np.hypot(X - c1x, Z - c1z), np.hypot(X - c2x, Z - c2z)
        t2 = top - rng.uniform(15, 45)
        r_sub = Reff * 0.55
        top1 = top + 3 * (fbm(X, 0 * Y, Z, 7, seed + 6, 2) - 0.5) - 6 * (d1 / (r * 0.55)) ** 2
        top2 = t2 + 3 * (fbm(X, 0 * Y, Z, 7, seed + 7, 2) - 0.5) - 6 * (d2 / (r * 0.55)) ** 2
        sub = ((d1 < r_sub) & (Y < top1)) | ((d2 < r_sub) & (Y < top2))
        solid = np.where(above, sub, (d < Reff) & (Y < topS))
    else:
        solid = (d < Reff) & (Y < topS)
    vol = np.zeros(solid.shape, np.uint8)
    vol[solid] = rock_ids(X, Y, Z, seed)[solid]
    up = np.zeros_like(solid); up[:, :-1, :] = solid[:, 1:, :]
    topf = solid & ~up
    vol[topf] = np.where(rng.random(int(topf.sum())) < 0.82, GRASS, MOSS)
    under = np.zeros_like(solid); under[:, 1:, :] = topf[:, :-1, :]
    vol[solid & under & ~topf] = DIRT
    # moss on the shaded north and east faces, in patches
    north = np.zeros_like(solid); north[:, :, 1:] = solid[:, :, :-1]
    shade = solid & ~topf & ~np.roll(solid, 1, axis=2) & (fbm(X, Y * 0.6, Z, 7, seed + 8, 2) > 0.52)
    vol[shade] = np.where(rng.random(int(shade.sum())) < 0.6, MOSS, MOSS2)
    # vines hanging from the rims of the top and the ledges
    ii, jj, kk = np.nonzero(topf)
    for i, j, k in zip(ii, jj, kk):
        if rng.random() > 0.10:
            continue
        for di, dk in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, c = i + di, k + dk
            if 0 <= a < nx and 0 <= c < nx and not solid[a, j, c]:
                L = int(rng.integers(2, int(14 / s) + 3))
                for q in range(1, L):
                    if j - q < 0 or solid[a, j - q, c]:
                        break
                    if rng.random() < 0.9:
                        vol[a, j - q, c] = VINE
                break
    # trees and bushes on the top and ledges
    for i, j, k in zip(ii, jj, kk):
        if rng.random() < trees and j < ny - 8:
            put_tree(vol, i, j + 1, k, 'cherry' if rng.random() < 0.45 else 'oak', s, rng)
    return vol, (x0, base, z0)


def rocks(cx, cy, cz, spread, count, seed, s):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(count):
        r = rng.uniform(4, 14)
        px, py, pz = cx + rng.normal(0, spread), cy + rng.normal(0, spread * 0.25), cz + rng.normal(0, spread)
        v, o = island(px, pz, py, r, int(rng.integers(1000)), s, depth=r * 1.1, trees=0.02 if r > 9 else 0.0, cherry=0.7)
        parts.append((v, o))
    return parts


if __name__ == '__main__':
    t0 = time.time()
    objs = []                          # (name, vol, origin, scale)
    cx, cz = at(76, 2300)
    v, o = island(cx, cz, 590, 560, 7, 6.0, depth=400, peak=620, cherry=0.6, falls=4, trees=0.07, snow_y=1010, jagged=True)
    objs.append(('colossus', v, o, 6.0)); print('colossus', v.shape, '%.0fs' % (time.time() - t0), flush=True)
    for name, (yaw, dist, top, r, seed, falls, s) in dict(
            i1=(48, 330, 332, 34, 11, 2, 2.0), i2=(99, 520, 364, 26, 12, 1, 2.0), i3=(68, 950, 300, 46, 13, 2, 3.0),
            i4=(33, 1250, 452, 62, 14, 1, 4.0), i5=(116, 1500, 520, 42, 15, 0, 4.0), i6=(86, 1700, 660, 30, 16, 1, 4.0),
            i7=(58, 2700, 720, 75, 17, 2, 6.0), i8=(132, 2400, 610, 55, 18, 1, 6.0)).items():
        x, z = at(yaw, dist)
        v, o = island(x, z, top, r, seed, s, falls=falls, peak=r * 0.5, cherry=0.55)
        objs.append((name, v, o, s))
    for k, (yaw, dist, r, top) in enumerate(hf3.PILLARS):
        x, z = at(yaw, dist)
        s = 1.25 if dist < 700 else (2.5 if dist < 1900 else 4.0)
        v, o = pillar(x, z, r, top, 300 + k, s, trees=0.05 if s < 2 else 0.035)
        objs.append(('pillar%d' % k, v, o, s))
    print('pillars %.0fs' % (time.time() - t0), flush=True)
    vl, sd, vd, c = arch(*at(148, 900), 148, 270, 250, 36, 48, 3.0, 21)
    v, o = place_rotated(vl, sd, vd, c, 3.0)
    objs.append(('arch', v, o, 3.0)); print('arch', v.shape, flush=True)
    for k, (v, o) in enumerate(rocks(*at(74, 2050)[:1], 470, at(74, 2050)[1], 260, 14, 40, 4.0)):
        objs.append(('rock%d' % k, v, o, 4.0))
    for k, (v, o) in enumerate(rocks(*at(55, 600)[:1], 380, at(55, 600)[1], 90, 5, 41, 2.0)):
        objs.append(('rockn%d' % k, v, o, 2.0))
    # mesh everything into one coloured group
    B = mesher.Builder()
    for name, v, o, s in objs:
        g = mesher.mesh_volume(v, o, scale=s, mode='colored', colors=PAL, chunk=256)['opaque']
        if g is None:
            continue
        P = g['pos'].reshape(-1, 4, 3); N = g['nrm'].reshape(-1, 4, 3)[:, 0]; C = g['col'].reshape(-1, 4, 3).astype(np.float32)
        U = np.zeros((len(P), 4, 2), np.float32)
        keys = np.full(len(P), hash(name) % 100000, np.int64)
        idx = g['idx'].reshape(-1, 6) % 4
        flip = idx[:, 0] == 1
        B.add(keys, P, N, U, C, flip)
    grp = B.build()
    mp = os.path.join(WEB, 'meta.json'); meta = json.load(open(mp))
    mesher.save_groups({'objects': grp}, WEB, 'obj', meta['groups'])
    json.dump(meta, open(mp, 'w'))
    json.dump(WATERFALLS, open(os.path.join(WEB, 'waterfalls.json'), 'w'))
    print('objects: %d verts, %d tris, %d waterfalls, %.0fs' % (len(grp['pos']), len(grp['idx']) // 3, len(WATERFALLS), time.time() - t0))
