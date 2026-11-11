"""v4: everything beyond the detailed blocks.  Water surfaces (lake, river, the streams above the falls), coarse terrain
in three rings (8 / 16 / 64 m cells, natural colours, no bands), instanced forests, the waterfalls, and one small floating
rock with a thread of water falling from it (the only openly impossible thing, and easy to miss).  -> web4/"""
import json, math, os, time
import numpy as np
import hf4, mesher
from lib import fbm
import objects3 as O
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web4')
SX, SZ = hf4.SEAT
DET = (-400, 160, -280, 280)
YAW_LO, YAW_HI = -40, 220


def srgb(c):
    return (np.asarray(c, np.float32) / 255) ** 2.2


def yaw_of(X, Z):
    y = np.degrees(np.arctan2(-(X - SX), -(Z - SZ))) % 360
    return np.where(y > 300, y - 360, y)


def grid(cell, half, inner):
    n = int(2 * half / cell)
    xs = SX - half + (np.arange(n) + 0.5) * cell; zs = SZ - half + (np.arange(n) + 0.5) * cell
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    keep = (yaw_of(X, Z) > YAW_LO) & (yaw_of(X, Z) < YAW_HI)
    if inner is not None:
        x0, x1, z0, z1 = inner
        keep &= ~((X + cell / 2 > x0) & (X - cell / 2 < x1) & (Z + cell / 2 > z0) & (Z - cell / 2 < z1))
    return X, Z, keep, n


def ring(cell, half, inner, B):
    X, Z, keep, n = grid(cell, half, inner)
    q = 1 if cell <= 8 else (2 if cell <= 16 else 4)
    H = np.round(hf4.height(X, Z) / q) * q
    keep &= H >= hf4.WATER_Y - 0.5
    pad = np.pad(H, 1, mode='edge')
    slope = np.max(np.abs(np.stack([pad[1:-1, 1:-1] - pad[:-2, 1:-1], pad[1:-1, 1:-1] - pad[2:, 1:-1],
                                     pad[1:-1, 1:-1] - pad[1:-1, :-2], pad[1:-1, 1:-1] - pad[1:-1, 2:]])), 0) / cell
    n1 = fbm(X, 0.0, Z, 140, 61, 3)[..., None]; n2 = fbm(X, 0.0, Z, 30, 62, 2)[..., None]; n3 = fbm(X, 0.0, Z, 9, 63, 2)[..., None]
    meadow = srgb((112, 158, 66)) * (0.85 + 0.3 * n2) * (0.92 + 0.16 * n3)
    forest_floor = srgb((62, 104, 44)) * (0.85 + 0.3 * n2)
    alpine = srgb((128, 146, 92)) * (0.85 + 0.3 * n3)
    rock = srgb((150, 142, 134)) * (0.78 + 0.35 * n1) * (0.9 + 0.2 * n3)
    snow = srgb((236, 240, 248)) * (0.95 + 0.08 * n3)
    shore = srgb((176, 164, 130)) * (0.9 + 0.2 * n3)
    F = hf4.forest(X, Z, H)
    top = np.where((F > 0.55)[..., None], forest_floor, meadow)
    top = np.where((H > 420 + 60 * n1[..., 0])[..., None], alpine, top)
    top = np.where((slope > 1.1)[..., None], rock, top)
    top = np.where(((H > 900 + 140 * n1[..., 0]) & (slope < 2.2))[..., None], snow, top)
    top = np.where((H < hf4.WATER_Y + 2)[..., None], shore, top)
    ii, kk = np.nonzero(keep)
    h = H[ii, kk]; x = X[ii, kk]; z = Z[ii, kk]; c = cell / 2
    keys = ((x // 512).astype(np.int64) * 100000 + (z // 512).astype(np.int64))
    pos = np.stack([np.stack([x - c, h, z - c], 1), np.stack([x - c, h, z + c], 1), np.stack([x + c, h, z + c], 1), np.stack([x + c, h, z - c], 1)], 1)
    B.add(keys, pos.astype(np.float32), np.broadcast_to(np.array([0, 127, 0], np.int8), (len(ii), 3)), np.zeros((len(ii), 4, 2), np.float32),
          np.repeat(top[ii, kk][:, None, :] * 255, 4, 1))
    # walls toward lower neighbours: rock, darker toward their feet by world height (no per-column banding)
    for (di, dk), nrm in (((1, 0), (1, 0, 0)), ((-1, 0), (-1, 0, 0)), ((0, 1), (0, 0, 1)), ((0, -1), (0, 0, -1))):
        ni, nk = ii + di, kk + dk
        ok = (ni >= 0) & (ni < n) & (nk >= 0) & (nk < n)
        hn = np.full(len(ii), hf4.WATER_Y - 3.0); hn[ok] = H[ni[ok], nk[ok]]
        m = h > hn + 0.5
        if not m.any():
            continue
        xm, zm, hm, lo = x[m], z[m], h[m], np.maximum(hn[m], hf4.WATER_Y - 3)
        ex, ez = xm + di * c, zm + dk * c; tx, tz = -dk, di
        a = np.stack([ex - tx * c, hm, ez - tz * c], 1); b = np.stack([ex + tx * c, hm, ez + tz * c], 1)
        a2 = np.stack([ex - tx * c, lo, ez - tz * c], 1); b2 = np.stack([ex + tx * c, lo, ez + tz * c], 1)
        quad = np.stack([a, b, b2, a2], 1)
        flip = np.dot(np.cross(quad[0, 1] - quad[0, 0], quad[0, 3] - quad[0, 0]), nrm) < 0
        nn = fbm(xm, hm * 0.5, zm, 60, 64, 3)[:, None]; nn2 = fbm(xm, hm, zm, 14, 65, 2)[:, None]
        base = srgb((150, 142, 134)) * (0.78 + 0.35 * nn) * (0.92 + 0.16 * nn2)
        mossy = (fbm(xm, hm * 0.3, zm, 45, 66, 2) > 0.58)[:, None] & (hm < 600)[:, None]
        base = np.where(mossy, srgb((84, 118, 58)) * (0.85 + 0.3 * nn2), base)
        snowy = (hm > 950 + 120 * nn[:, 0])[:, None]
        base = np.where(snowy, srgb((222, 228, 238)), base)

        def shade_at(y):
            return (0.62 + 0.38 * np.clip((y - 64) / 260, 0, 1))[:, None]
        col = np.stack([base * shade_at(hm), base * shade_at(hm), base * shade_at(lo), base * shade_at(lo)], 1) * 255
        if flip:
            quad = quad[:, [0, 3, 2, 1]]; col = col[:, [0, 3, 2, 1]]
        B.add(keys[m], quad.astype(np.float32), np.broadcast_to(np.array(nrm, np.int8) * 127, (m.sum(), 3)), np.zeros((m.sum(), 4, 2), np.float32), col)


def water(B):
    """flat water: the lake, the river, the pools, and the streams up in the hanging valleys"""
    for cell, half, inner in ((4, 1600, None), (12, 4800, (SX - 1600, SX + 1600, SZ - 1600, SZ + 1600))):
        X, Z, keep, n = grid(cell, half, inner)
        H = hf4.height(X, Z)
        lvl = np.where(H < hf4.WATER_Y + 0.5, hf4.WATER_Y + 0.88, -np.inf)
        ch = hf4.channel_level(X, Z)
        lvl = np.maximum(lvl, np.where(H < ch + 0.5, ch, -np.inf))
        ii, kk = np.nonzero(keep & np.isfinite(lvl))
        x = X[ii, kk]; z = Z[ii, kk]; y = lvl[ii, kk]; c = cell / 2 + 0.02
        keys = ((x // 512).astype(np.int64) * 100000 + (z // 512).astype(np.int64))
        pos = np.stack([np.stack([x - c, y, z - c], 1), np.stack([x - c, y, z + c], 1), np.stack([x + c, y, z + c], 1), np.stack([x + c, y, z - c], 1)], 1)
        B.add(keys, pos.astype(np.float32), np.broadcast_to(np.array([0, 127, 0], np.int8), (len(ii), 3)), np.zeros((len(ii), 4, 2), np.float32),
              np.full((len(ii), 4, 3), 255.0, np.float32))


def trees():
    """instanced forests outside the detailed blocks: (x, y, z, size, kind); kinds 0 oak, 1 spruce, 2 birch, 3 cherry"""
    rng = np.random.default_rng(8)
    out = []
    for cell, half, inner, base in ((7, 1300, DET, 1.0), (12, 3200, (SX - 1300, SX + 1300, SZ - 1300, SZ + 1300), 1.5)):
        X, Z, keep, n = grid(cell, half, inner)
        X = X + rng.uniform(-0.45, 0.45, X.shape) * cell; Z = Z + rng.uniform(-0.45, 0.45, Z.shape) * cell
        H = hf4.height(X, Z)
        Hx = hf4.height(X + 4, Z); Hz = hf4.height(X, Z + 4)
        slope = np.hypot(Hx - H, Hz - H) / 4
        F = hf4.forest(X, Z, H)
        G = hf4.cherry_grove(X, Z)
        m = keep & (rng.random(X.shape) < F * 0.95) & (slope < 0.85) & (H > hf4.WATER_Y + 1.5)
        ii, kk = np.nonzero(m)
        h = H[ii, kk]
        kind = np.where(h > 250, 1, np.where(rng.random(len(ii)) < 0.25, 2, np.where(rng.random(len(ii)) < 0.3, 1, 0)))
        kind = np.where((G[ii, kk] > 0.5) & (rng.random(len(ii)) < 0.7), 3, kind)
        size = base * rng.uniform(0.8, 1.3, len(ii))
        out.append(np.stack([X[ii, kk], h + 1, Z[ii, kk], size, kind], 1))
    T = np.concatenate(out).astype(np.float32)
    T.tofile(os.path.join(WEB, 'trees.bin'))
    return len(T)


if __name__ == '__main__':
    t0 = time.time()
    mp = os.path.join(WEB, 'meta.json'); meta = json.load(open(mp))
    meta['groups'] = {k: v for k, v in meta['groups'].items() if k.startswith('detail')}
    B = mesher.Builder()
    ring(8, 1300, DET, B); print('ring 8 %.0fs' % (time.time() - t0), flush=True)
    ring(16, 3400, (SX - 1300, SX + 1300, SZ - 1300, SZ + 1300), B); print('ring 16 %.0fs' % (time.time() - t0), flush=True)
    ring(64, 14336, (SX - 3400, SX + 3400, SZ - 3400, SZ + 3400), B); print('ring 64 %.0fs' % (time.time() - t0), flush=True)
    mesher.save_groups({'lod': B.build()}, WEB, 'lod', meta['groups'])
    W = mesher.Builder(); water(W)
    mesher.save_groups({'water': W.build()}, WEB, 'wat', meta['groups']); print('water %.0fs' % (time.time() - t0), flush=True)
    nt = trees(); print('trees', nt, '%.0fs' % (time.time() - t0), flush=True)
    # the one openly impossible thing: a small floating rock with a thread of water, far off over the slopes
    O.WATERFALLS.clear()
    x, z = hf4.at(64, 2100)
    v, o = O.island(x, z, 640, 22, 91, 2.0, falls=1, peak=8, cherry=0.6, trees=0.05)
    g = mesher.mesh_volume(v, o, scale=2.0, mode='colored', colors=O.PAL, chunk=256)['opaque']
    mesher.save_groups({'objects': g}, WEB, 'obj', meta['groups'])
    falls = []
    for f in O.WATERFALLS:                         # it never reaches the ground: it frays out into mist
        f = dict(f); f['bottom'] = f['y'] - 260; f['width'] = 4.0; f['fade'] = 1; falls.append(f)
    for i, (yaw, dist, lip, w) in enumerate(hf4.FALLS):
        px, pz = hf4.at(yaw, dist); ux, uz = (SX - px) / dist, (SZ - pz) / dist
        falls.append(dict(x=px, z=pz, y=lip - 0.6, bottom=hf4.WATER_Y + 0.5, width=float(w), dir=[ux, uz], seed=500 + i, fade=0))
    json.dump(falls, open(os.path.join(WEB, 'waterfalls.json'), 'w'))
    meta['spot'] = meta.get('spot', [10.0, 250.0, 0.0])
    meta['fall_bases'] = [[f['x'] + f['dir'][0] * 30, hf4.WATER_Y + 8, f['z'] + f['dir'][1] * 30, f['width']] for f in falls if not f['fade']]
    json.dump(meta, open(mp, 'w'))
    print({k: (v['verts'], v['tris']) for k, v in meta['groups'].items()}, '%.0fs' % (time.time() - t0))
