"""v5 distant terrain that reads like the detailed blocks: columns of 2 m cells out to 1.3 km, 4 m to 3.5 km and 16 m beyond,
each tagged with the block textures it would have (top / side / deep tile), drawn by the renderer with world-space
block-sized texture tiling, so the ground keeps its grass, dirt, stone and snow texture all the way out.  Flat runs are
merged.  Forests are placed exactly like the detailed blocks' (5 m grid, same density and species rules).
-> web4/lod5.bin, web4/trees5.bin, entries in web4/meta.json"""
import json, math, os, time
import numpy as np
import hf4, textures as TX
from lib import fbm
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web4')
SX, SZ = hf4.SEAT
DET = (-400, 160, -280, 280)
YAW_LO, YAW_HI = -18, 198
T = TX.IDX
STONE, ANDESITE, TUFF = T['stone'], T['andesite'], T['tuff']
ROCK_TILES = {STONE, ANDESITE, TUFF, T['gravel'], T['calcite']}


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


class LB:
    """quads with position, normal, colour (shading), tiles (top, side, deep) and the column top height"""
    def __init__(self):
        self.P, self.N, self.C, self.TL, self.TOP, self.K = [], [], [], [], [], []

    def add(self, keys, quads, nrm, col, tiles, top):
        if len(quads) == 0:
            return
        self.K.append(keys); self.P.append(quads.astype(np.float32)); self.N.append(np.broadcast_to(np.asarray(nrm, np.int8), (len(quads), 3)))
        col = np.asarray(col, np.float32)
        if col.ndim == 2:
            col = np.repeat(col[:, None, :], 4, 1)
        self.C.append(col); self.TL.append(tiles.astype(np.uint8)); self.TOP.append(top.astype(np.float32))

    def save(self, name, meta):
        K = np.concatenate(self.K); order = np.argsort(K, kind='stable'); K = K[order]
        P = np.concatenate(self.P)[order]; N = np.concatenate(self.N)[order]; C = np.concatenate(self.C)[order]
        TL = np.concatenate(self.TL)[order]; TOP = np.concatenate(self.TOP)[order]
        n = len(K)
        pos = P.reshape(-1, 3)
        nrm = np.repeat(N[:, None, :], 4, 1).reshape(-1, 3).astype(np.int8)
        col = np.clip(C.reshape(-1, 3) * 255, 0, 255).astype(np.uint8)
        tl = np.zeros((n * 4, 4), np.uint8); tl[:, :3] = np.repeat(TL[:, None, :], 4, 1).reshape(-1, 3)
        top = np.repeat(TOP[:, None], 4, 1).reshape(-1).astype(np.float32)
        starts = np.concatenate([[0], np.nonzero(np.diff(K))[0] + 1, [n]])
        idx, chunks, i0 = [], [], 0
        tri = np.array([0, 1, 2, 0, 2, 3])
        for a, b in zip(starts[:-1], starts[1:]):
            idx.append((tri[None, :] + (np.arange(b - a) * 4)[:, None]).reshape(-1).astype(np.uint32))
            chunks.append((int(K[a]), int(a * 4), int((b - a) * 4), int(i0), int((b - a) * 6))); i0 += (b - a) * 6
        idx = np.concatenate(idx)
        parts, buf, off = {}, bytearray(), 0
        for k, arr in (('pos', pos), ('top', top), ('idx', idx), ('nrm', nrm), ('col', col), ('tile', tl)):
            bb = arr.tobytes(); parts[k] = [off, int(arr.size)]; buf += bb; off += len(bb); pad = (-off) % 4; buf += b'\0' * pad; off += pad
        open(os.path.join(WEB, name + '.bin'), 'wb').write(bytes(buf))
        meta[name] = dict(file=name + '.bin', parts=parts, chunks=chunks, verts=int(n * 4), tris=int(n * 2), kind='lod5')
        return n


def tiles_for(X, Z, H, slope, cell):
    n1 = fbm(X, 0.0, Z, 90, 61, 3); n2 = fbm(X, 0.0, Z, 25, 62, 2); n3 = fbm(X, 0.0, Z, 7, 63, 2)
    F = hf4.forest(X, Z, H)
    top = np.full(H.shape, T['grass_top']); side = np.full(H.shape, T['grass_side']); deep = np.full(H.shape, T['dirt'])
    alpine = H > 420 + 60 * n1
    top = np.where(alpine & (n3 > 0.62), T['coarse_dirt'], top)
    top = np.where(alpine & (n3 < 0.3), T['gravel'], top)
    top = np.where((F > 0.6) & (n3 > 0.66) & (H > 140), T['podzol_top'], top); side = np.where(top == T['podzol_top'], T['podzol_side'], side)
    rock = slope > 1.05
    rk = np.where(n2 < 0.38, ANDESITE, np.where(n2 > 0.66, TUFF, STONE))
    top = np.where(rock, rk, top); side = np.where(rock, rk, side); deep = np.where(rock, rk, deep)
    snow = (H > 930 + 140 * n1) & (slope < 2.4)
    top = np.where(snow, T['snow'], top); side = np.where(snow, T['stone_snow_side'], side); deep = np.where(snow, rk, deep)
    shore = H < hf4.WATER_Y + 2
    top = np.where(shore, np.where(n3 > 0.5, T['sand'], T['gravel']), top); side = np.where(shore, top, side); deep = np.where(shore, top, deep)
    top = np.where(H < hf4.WATER_Y - 0.5, np.where(n2 > 0.6, T['clay'], T['sand']), top)
    return top, side, deep


def ring(cell, half, inner, B, merge=True):
    X, Z, keep, n = grid(cell, half, inner)
    q = 1 if cell <= 4 else 2
    H = np.round(hf4.height(X, Z) / q) * q
    pad = np.pad(H, 1, mode='edge')
    slope = np.max(np.abs(np.stack([pad[1:-1, 1:-1] - pad[:-2, 1:-1], pad[1:-1, 1:-1] - pad[2:, 1:-1],
                                     pad[1:-1, 1:-1] - pad[1:-1, :-2], pad[1:-1, 1:-1] - pad[1:-1, 2:]])), 0) / cell
    top, side, deep = tiles_for(X, Z, H, slope, cell)
    if cell >= 4:
        dist = np.hypot(X - SX, Z - SZ)
        F = hf4.forest(X, Z, H); G = hf4.cherry_grove(X, Z)
        rr = np.random.default_rng(cell).random(H.shape)
        can = (dist > 1500) & (rr < F * 0.92) & (slope < 1.0) & (H > hf4.WATER_Y + 1)
        ch = np.where(H > 140, 9 + 5 * rr, 6 + 3 * rr) * (0.85 if cell > 4 else 1.0)
        leaf = np.where(H > 140, T['spruce_leaves'], np.where(fbm(X, 0.0, Z, 40, 71, 2) > 0.62, T['birch_leaves'], T['oak_leaves']))
        leaf = np.where((G > 0.5) & (rr < 0.7), T['cherry_leaves'], leaf)
        H = np.where(can, np.round(H + ch), H)
        top = np.where(can, leaf, top); side = np.where(can, leaf, side); deep = np.where(can, leaf, deep)
        pad = np.pad(H, 1, mode='edge')
    hmax = np.max(np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]]), 0)
    occl = np.clip((hmax - H) / max(cell, 2), 0, 1)                   # a higher neighbour darkens the top a little (AO)
    shade = (1.0 - 0.28 * occl) * (0.93 + 0.07 * fbm(X, 0.0, Z, 11, 64, 2))
    c = cell / 2
    keepw = keep & (H >= hf4.WATER_Y - 3)
    # tops, merged along z where height, tiles and shading agree
    for i in range(n):
        row = keepw[i]
        if not row.any():
            continue
        h, t1, sh = H[i], top[i], np.round(shade[i] * 20) / 20
        ks = np.nonzero(row)[0]
        # split runs at gaps or changes
        brk = np.ones(len(ks), bool)
        brk[1:] = (np.diff(ks) != 1) | (h[ks[1:]] != h[ks[:-1]]) | (t1[ks[1:]] != t1[ks[:-1]]) | (sh[ks[1:]] != sh[ks[:-1]]) if merge else True
        st = np.nonzero(brk)[0]; en = np.concatenate([st[1:], [len(ks)]])
        k0 = ks[st]; k1 = ks[en - 1]
        x = X[i, 0]; z0 = Z[i, k0] - c; z1 = Z[i, k1] + c; y = h[k0]
        quads = np.stack([np.stack([np.full(len(k0), x - c), y, z0], 1), np.stack([np.full(len(k0), x - c), y, z1], 1),
                          np.stack([np.full(len(k0), x + c), y, z1], 1), np.stack([np.full(len(k0), x + c), y, z0], 1)], 1)
        keys = ((x // 512).astype(np.int64) * 100000 + (Z[i, k0] // 512).astype(np.int64)) if np.ndim(x) else (int(x // 512) * 100000 + (Z[i, k0] // 512).astype(np.int64))
        B.add(keys, quads, (0, 127, 0), np.repeat(sh[k0][:, None], 3, 1), np.stack([t1[k0], side[i, k0], deep[i, k0]], 1), y)
    # walls toward lower neighbours
    ii, kk = np.nonzero(keepw)
    h = H[ii, kk]; x = X[ii, kk]; z = Z[ii, kk]
    keys = ((x // 512).astype(np.int64) * 100000 + (z // 512).astype(np.int64))
    for (di, dk), nrm in (((1, 0), (1, 0, 0)), ((-1, 0), (-1, 0, 0)), ((0, 1), (0, 0, 1)), ((0, -1), (0, 0, -1))):
        ni, nk = ii + di, kk + dk
        ok = (ni >= 0) & (ni < n) & (nk >= 0) & (nk < n)
        hn = np.full(len(ii), hf4.WATER_Y - 4.0); hn[ok] = H[ni[ok], nk[ok]]
        m = h > hn + 0.5
        if not m.any():
            continue
        xm, zm, hm, lo = x[m], z[m], h[m], np.maximum(hn[m], hf4.WATER_Y - 4)
        ex, ez = xm + di * c, zm + dk * c; tx, tz = -dk, di
        a = np.stack([ex - tx * c, hm, ez - tz * c], 1); b = np.stack([ex + tx * c, hm, ez + tz * c], 1)
        a2 = np.stack([ex - tx * c, lo, ez - tz * c], 1); b2 = np.stack([ex + tx * c, lo, ez + tz * c], 1)
        quad = np.stack([a, b, b2, a2], 1)
        if np.dot(np.cross(quad[0, 1] - quad[0, 0], quad[0, 3] - quad[0, 0]), nrm) < 0:
            quad = quad[:, [0, 3, 2, 1]]
        # ambient occlusion down the wall: darker toward its foot
        ctop = np.ones(m.sum()) * 0.98; cbot = np.where(hm - lo > 3, 0.62, 0.8)
        col = np.stack([ctop, ctop, cbot, cbot], 1)
        if np.dot(np.cross(b[0] - a[0], a2[0] - a[0]), nrm) < 0:
            col = col[:, [0, 3, 2, 1]]
        sel = (ii[m], kk[m])
        B.add(keys[m], quad, np.array(nrm) * 127, np.repeat(col[:, :, None], 3, 2), np.stack([top[sel], side[sel], deep[sel]], 1), hm)


def smooth_ring(cell, half, inner, name, meta, tile_cells=64):
    """far terrain as a smooth shared-vertex surface (a column here is 2-3 px), same textures, canopy bumps in forests;
    where it tucks under the finer ring it sinks 2 m, so there are no cracks"""
    n = int(2 * half / cell) + 1
    xs = SX - half + np.arange(n) * cell; zs = SZ - half + np.arange(n) * cell
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    H = hf4.height(X, Z)
    gx, gz = np.gradient(H, cell)
    slope = np.hypot(gx, gz)
    top, side, deep = tiles_for(X, Z, H, slope * cell, cell)
    side = np.where(slope > 1.05, top, side)
    dist = np.hypot(X - SX, Z - SZ)
    F = hf4.forest(X, Z, H); G = hf4.cherry_grove(X, Z)
    rr = np.random.default_rng(cell + 7).random(H.shape)
    can = (dist > 1500) & (rr < F * 0.9) & (slope < 0.9) & (H > hf4.WATER_Y + 1)
    leaf = np.where(H > 140, T['spruce_leaves'], np.where(fbm(X, 0.0, Z, 40, 71, 2) > 0.62, T['birch_leaves'], T['oak_leaves']))
    leaf = np.where((G > 0.5) & (rr < 0.7), T['cherry_leaves'], leaf)
    H = np.where(can, H + np.where(H > 140, 8 + 5 * rr, 6 + 3 * rr), H)
    top = np.where(can, leaf, top); side = np.where(can, leaf, side)
    x0, x1, z0, z1 = inner
    insq = (X > x0 + cell) & (X < x1 - cell) & (Z > z0 + cell) & (Z < z1 - cell)
    H = np.where((X > x0 - cell) & (X < x1 + cell) & (Z > z0 - cell) & (Z < z1 + cell), H - 2.0, H)
    gx, gz = np.gradient(H, cell)
    nrm = np.stack([-gx, np.ones_like(H), -gz], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    shade = 0.93 + 0.07 * fbm(X, 0.0, Z, 11, 64, 2)
    # cells: drawn unless entirely inside the finer ring or behind him
    cx, cz = X[:-1, :-1] + cell / 2, Z[:-1, :-1] + cell / 2
    draw = (yaw_of(cx, cz) > YAW_LO) & (yaw_of(cx, cz) < YAW_HI) & ~(insq[:-1, :-1] & insq[1:, 1:] & insq[1:, :-1] & insq[:-1, 1:])
    pos, nn, col, tl, idx, chunks = [], [], [], [], [], []
    v_off = i_off = 0
    for ti in range(0, n - 1, tile_cells):
        for tk in range(0, n - 1, tile_cells):
            d = draw[ti:ti + tile_cells, tk:tk + tile_cells]
            if not d.any():
                continue
            a, b = d.shape
            sl = (slice(ti, ti + a + 1), slice(tk, tk + b + 1))
            P = np.stack([X[sl], H[sl], Z[sl]], -1).reshape(-1, 3)
            ii, kk = np.nonzero(d)
            v00 = ii * (b + 1) + kk; v01 = v00 + 1; v10 = v00 + (b + 1); v11 = v10 + 1
            tri = np.stack([v00, v01, v11, v00, v11, v10], 1).reshape(-1).astype(np.uint32)
            pos.append(P.astype(np.float32)); nn.append((nrm[sl].reshape(-1, 3) * 127).astype(np.int8))
            col.append(np.clip(np.repeat(shade[sl].reshape(-1, 1), 3, 1) * 255, 0, 255).astype(np.uint8))
            t4 = np.zeros((P.shape[0], 4), np.uint8); t4[:, 0] = top[sl].reshape(-1); t4[:, 1] = side[sl].reshape(-1); t4[:, 2] = side[sl].reshape(-1)
            tl.append(t4); idx.append(tri)
            chunks.append((int(ti * 1000 + tk), int(v_off), int(P.shape[0]), int(i_off), int(len(tri))))
            v_off += P.shape[0]; i_off += len(tri)
    pos = np.concatenate(pos); nn = np.concatenate(nn); col = np.concatenate(col); tl = np.concatenate(tl); idx = np.concatenate(idx)
    top_a = np.full(len(pos), -1e4, np.float32)
    parts, buf, off = {}, bytearray(), 0
    for k, arr in (('pos', pos), ('top', top_a), ('idx', idx), ('nrm', nn), ('col', col), ('tile', tl)):
        bb = arr.tobytes(); parts[k] = [off, int(arr.size)]; buf += bb; off += len(bb); pad = (-off) % 4; buf += b'\0' * pad; off += pad
    open(os.path.join(WEB, name + '.bin'), 'wb').write(bytes(buf))
    meta[name] = dict(file=name + '.bin', parts=parts, chunks=chunks, verts=int(len(pos)), tris=int(len(idx) // 3), kind='lod5')
    return len(idx) // 3


def trees(meta):
    """the same rules as the detailed blocks: a 5 m grid, density from the forest field, spruce up the slopes,
    birch and oak below, cherry in the groves.  (x, y, z, kind, height, canopy radius)"""
    rng = np.random.default_rng(42)
    out = []
    for half, inner in ((1500, DET),):
        X, Z, keep, n = grid(5, half, inner)
        X = X + rng.integers(-2, 3, X.shape); Z = Z + rng.integers(-2, 3, Z.shape)
        H = np.round(hf4.height(X, Z))
        slope = np.maximum(np.abs(np.round(hf4.height(X + 1, Z)) - H), np.abs(np.round(hf4.height(X, Z + 1)) - H))
        F = hf4.forest(X, Z, H); G = hf4.cherry_grove(X, Z)
        lx, lz = hf4.LAKE
        view = (np.hypot((X - SX), (Z - SZ)) < 90) & (X < SX - 3)
        m = keep & (slope <= 1) & (H > hf4.WATER_Y + 1) & (rng.random(X.shape) < F * np.where(view, 0.3, 1.0))
        ii, kk = np.nonzero(m)
        h = H[ii, kk]; r = rng.random(len(ii)); r2 = rng.random(len(ii)); r3 = rng.random(len(ii))
        kind = np.where(h > 140, 1, np.where(r < 0.3, 2, np.where(r2 < 0.35, 1, 0)))
        kind = np.where((G[ii, kk] > 0.5) & (r3 < 0.75), 3, kind)
        ht = np.select([kind == 0, kind == 1, kind == 2, kind == 3], [rng.integers(5, 8, len(ii)) + 3 * (rng.random(len(ii)) < 0.08),
                       (rng.integers(9, 15, len(ii)) * np.where(h < 230, 1.0, 0.75)).astype(int), rng.integers(6, 9, len(ii)),
                       (rng.integers(4, 6, len(ii)) * (0.9 + 0.5 * rng.random(len(ii)))).astype(int)])
        out.append(np.stack([X[ii, kk], h + 1, Z[ii, kk], kind, ht, rng.random(len(ii))], 1))
    Tr = np.concatenate(out).astype(np.float32)
    Tr.tofile(os.path.join(WEB, 'trees5.bin'))
    return len(Tr)


if __name__ == '__main__':
    t0 = time.time()
    mp = os.path.join(WEB, 'meta.json'); meta = json.load(open(mp))
    meta['groups'] = {k: v for k, v in meta['groups'].items() if k in ('detail_opaque', 'detail_leaves', 'detail_plants', 'wat_water', 'obj_objects')}
    B = LB()
    prev = DET
    for cell, half in ((2, 900), (4, 1800)):
        ring(cell, half, prev, B); prev = (SX - half, SX + half, SZ - half, SZ + half)
        print('ring %d %.0fs' % (cell, time.time() - t0), flush=True)
    for cell, half in ((8, 3600), (16, 7200), (32, 14400)):
        nt_ = smooth_ring(cell, half, prev, 'lod5s%d' % cell, meta['groups']); prev = (SX - half, SX + half, SZ - half, SZ + half)
        print('smooth ring %d: %d tris %.0fs' % (cell, nt_, time.time() - t0), flush=True)
    nq = B.save('lod5', meta['groups'])
    nt = trees(meta)
    json.dump(meta, open(mp, 'w'))
    print('lod5 quads %d (tris %d), trees %d, %.0fs' % (nq, nq * 2, nt, time.time() - t0))
