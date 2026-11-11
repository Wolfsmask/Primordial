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


def gully_depth(H, cell):
    """how far a point lies below its surroundings (m): + in gullies and hollows, - on ridges and spurs"""
    from scipy.ndimage import uniform_filter
    k = max(3, int(round(36 / cell)) | 1)
    return uniform_filter(H, size=k, mode='nearest') - H


def tiles_for(X, Z, H, slope, cell, gully=None):
    """block tiles per column: meadows low, alpine turf, scree and gravel fans higher up, bare rock on steep ground
    (patches of stone / andesite / tuff; the renderer adds rock layers), snow on the summits and down the gullies.
    Also returns a 0..1 snow weight that the smooth far rings hand to the shader."""
    n1 = fbm(X, 0.0, Z, 90, 61, 3); n2 = fbm(X, 0.0, Z, 25, 62, 2); n3 = fbm(X, 0.0, Z, 7, 63, 2)
    if gully is None:
        gully = np.zeros_like(H)
    F = hf4.forest(X, Z, H)
    top = np.full(H.shape, T['grass_top']); side = np.full(H.shape, T['grass_side']); deep = np.full(H.shape, T['dirt'])
    alpine = H > 420 + 60 * n1
    top = np.where(alpine & (n3 > 0.62), T['coarse_dirt'], top)
    top = np.where(alpine & (n3 < 0.3), T['gravel'], top)
    top = np.where(alpine & (H > 600 + 80 * n1) & (n3 > 0.45) & (n3 < 0.62), T['moss'], top)
    top = np.where((F > 0.6) & (n3 > 0.66) & (H > 140), T['podzol_top'], top); side = np.where(top == T['podzol_top'], T['podzol_side'], side)
    # scree: loose gravel fans on the steeper alpine slopes and at the feet of the cliffs
    scree = (H > 300 + 80 * n1) & (slope > 0.55) & (slope <= 1.05) & ((n2 > 0.4) | (gully > 2.0))
    top = np.where(scree, T['gravel'], top); side = np.where(scree, T['gravel'], side); deep = np.where(scree, T['gravel'], deep)
    rock = slope > 1.05
    rk = np.where(n2 < 0.38, ANDESITE, np.where(n2 > 0.66, TUFF, STONE))
    rk = np.where((H < 650) & (n3 > 0.72) & (gully > 0), T['mossy_stone'], rk)
    top = np.where(rock, rk, top); side = np.where(rock, rk, side); deep = np.where(rock, rk, deep)
    # snow: caps above a wavy snowline, tongues down the gullies, a dusting on the high steep faces
    sl = 880 + 120 * n1
    cap = np.clip((H - sl) / 60 + 0.5, 0, 1)
    tongue = np.clip((H - (640 + 100 * n1)) / 80, 0, 1) * np.clip((gully - 1.5) / 4, 0, 1)
    sw = np.maximum(cap, tongue)                    # the shader decides what is too steep to hold it
    sw = np.maximum(sw, 0.55 * np.clip((H - 1050) / 150, 0, 1) * (n3 > 0.5))
    snow = (sw > 0.5 + 0.25 * (n3 - 0.5)) & ((slope < 1.8) | (H > sl + 160) | (n3 > 0.55))
    top = np.where(snow, T['snow'], top); side = np.where(snow & (slope < 1.05), T['stone_snow_side'], side); deep = np.where(snow, rk, deep)
    shore = H < hf4.WATER_Y + 2
    top = np.where(shore, np.where(n3 > 0.5, T['sand'], T['gravel']), top); side = np.where(shore, top, side); deep = np.where(shore, top, deep)
    top = np.where(H < hf4.WATER_Y - 0.5, np.where(n2 > 0.6, T['clay'], T['sand']), top)
    return top, side, deep, np.clip(sw, 0, 1)


def ring(cell, half, inner, B, merge=True):
    X, Z, keep, n = grid(cell, half, inner)
    ring_cells(X, Z, X, Z, keep, cell, B, merge)


def ring_cells(X, Z, XL, ZL, keep, cell, B, merge=True, seed=None):
    """X, Z: world positions of the column centres (heights, tiles); XL, ZL: where the quads are laid out - the same
    grid for the rings, the waterfall's own frame for the cliff patches (rotated into place afterwards)"""
    n, n2 = X.shape
    q = 1 if cell <= 4 else 2
    H = np.round(hf4.height(X, Z) / q) * q
    pad = np.pad(H, 1, mode='edge')
    slope = np.max(np.abs(np.stack([pad[1:-1, 1:-1] - pad[:-2, 1:-1], pad[1:-1, 1:-1] - pad[2:, 1:-1],
                                     pad[1:-1, 1:-1] - pad[1:-1, :-2], pad[1:-1, 1:-1] - pad[1:-1, 2:]])), 0) / cell
    gul = gully_depth(H, cell)
    top, side, deep, _ = tiles_for(X, Z, H, slope, cell, gul)
    if cell >= 4:
        dist = np.hypot(X - SX, Z - SZ)
        F = hf4.forest(X, Z, H); G = hf4.cherry_grove(X, Z)
        rr = np.random.default_rng(cell if seed is None else seed).random(H.shape)
        can = (dist > 1500) & (rr < F * 0.92) & (slope < 1.0) & (H > hf4.WATER_Y + 1)
        ch = np.where(H > 140, 9 + 5 * rr, 6 + 3 * rr) * (0.85 if cell > 4 else 1.0)
        leaf = np.where(H > 140, T['spruce_leaves'], np.where(fbm(X, 0.0, Z, 40, 71, 2) > 0.62, T['birch_leaves'], T['oak_leaves']))
        leaf = np.where((G > 0.5) & (rr < 0.7), T['cherry_leaves'], leaf)
        H = np.where(can, np.round(H + ch), H)
        top = np.where(can, leaf, top); side = np.where(can, leaf, side); deep = np.where(can, leaf, deep)
        pad = np.pad(H, 1, mode='edge')
    hmax = np.max(np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]]), 0)
    occl = np.clip((hmax - H) / max(cell, 2), 0, 1)                   # a higher neighbour darkens the top a little (AO)
    shade = (1.0 - 0.28 * occl) * (0.92 + 0.08 * fbm(X, 0.0, Z, 11, 64, 2)) * (1.0 - 0.22 * np.clip(gul / 10, 0, 1) + 0.06 * np.clip(-gul / 10, 0, 1))
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
        x = XL[i, 0]; z0 = ZL[i, k0] - c; z1 = ZL[i, k1] + c; y = h[k0]
        quads = np.stack([np.stack([np.full(len(k0), x - c), y, z0], 1), np.stack([np.full(len(k0), x - c), y, z1], 1),
                          np.stack([np.full(len(k0), x + c), y, z1], 1), np.stack([np.full(len(k0), x + c), y, z0], 1)], 1)
        keys = ((x // 512).astype(np.int64) * 100000 + (ZL[i, k0] // 512).astype(np.int64)) if np.ndim(x) else (int(x // 512) * 100000 + (ZL[i, k0] // 512).astype(np.int64))
        B.add(keys, quads, (0, 127, 0), np.repeat(sh[k0][:, None], 3, 1), np.stack([t1[k0], side[i, k0], deep[i, k0]], 1), y)
    # walls toward lower neighbours
    ii, kk = np.nonzero(keepw)
    h = H[ii, kk]; x = XL[ii, kk]; z = ZL[ii, kk]
    keys = ((x // 512).astype(np.int64) * 100000 + (z // 512).astype(np.int64))
    for (di, dk), nrm in (((1, 0), (1, 0, 0)), ((-1, 0), (-1, 0, 0)), ((0, 1), (0, 0, 1)), ((0, -1), (0, 0, -1))):
        ni, nk = ii + di, kk + dk
        ok = (ni >= 0) & (ni < n) & (nk >= 0) & (nk < n2)
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


PATCH_A, PATCH_P = (-240.0, 180.0), 300.0         # each fall's cliff patch, in its own frame (along, perp)


def fall_frames():
    out = []
    for yaw, dist, lip, w in hf4.FALLS:
        px, pz = hf4.at(yaw, dist)
        out.append((px, pz, (SX - px) / dist, (SZ - pz) / dist))
    return out


def in_patch(X, Z, grow=0.0):
    m = np.zeros(np.shape(X), bool)
    for px, pz, ux, uz in fall_frames():
        a = (X - px) * ux + (Z - pz) * uz; p = -(X - px) * uz + (Z - pz) * ux
        m |= (a > PATCH_A[0] - grow) & (a < PATCH_A[1] + grow) & (np.abs(p) < PATCH_P + grow)
    return m


def fall_patches(B, cell=4):
    """the cliffs round the big falls as 4 m blocks laid out square to each fall, so the cliff face runs straight and the
    water leaves exactly from the notch (the 8 m far terrain made a staircase of the diagonal cliff line)"""
    ring4 = (SX - 1800, SX + 1800, SZ - 1800, SZ + 1800)
    tot = 0
    for i, (px, pz, ux, uz) in enumerate(fall_frames()):
        A = np.arange(PATCH_A[0] + cell / 2, PATCH_A[1], cell); P = np.arange(-PATCH_P + cell / 2, PATCH_P, cell)
        AL, PL = np.meshgrid(A, P, indexing='ij')
        X = px + ux * AL - uz * PL; Z = pz + uz * AL + ux * PL
        keep = ~((X > ring4[0]) & (X < ring4[1]) & (Z > ring4[2]) & (Z < ring4[3]))
        keep &= (yaw_of(X, Z) > YAW_LO) & (yaw_of(X, Z) < YAW_HI)
        L = LB()
        ring_cells(X, Z, AL, PL, keep, cell, L, True, seed=900 + i)
        if not L.K:
            continue
        for k in range(len(L.P)):
            Pq = L.P[k]; a, p = Pq[..., 0].copy(), Pq[..., 2].copy()
            Pq[..., 0] = px + ux * a - uz * p; Pq[..., 2] = pz + uz * a + ux * p
            Nn = L.N[k].astype(np.float32); na, npp = Nn[:, 0].copy(), Nn[:, 2].copy()
            Nn[:, 0] = ux * na - uz * npp; Nn[:, 2] = uz * na + ux * npp
            L.N[k] = np.round(Nn).astype(np.int8)
            L.K[k] = np.full(len(Pq), 7_000_000 + i, np.int64)
            tot += len(Pq)
        B.K += L.K; B.P += L.P; B.N += L.N; B.C += L.C; B.TL += L.TL; B.TOP += L.TOP
    return tot


def smooth_ring(cell, half, inner, name, meta, tile_cells=64):
    """far terrain as a smooth shared-vertex surface (a column here is 2-3 px), same textures, canopy bumps in forests;
    where it tucks under the finer ring it sinks 2 m, so there are no cracks"""
    n = int(2 * half / cell) + 1
    xs = SX - half + np.arange(n) * cell; zs = SZ - half + np.arange(n) * cell
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    H = hf4.height(X, Z)
    gx, gz = np.gradient(H, cell)
    slope = np.hypot(gx, gz)
    gul = gully_depth(H, cell)
    top, side, deep, sw = tiles_for(X, Z, H, slope, cell, gul)
    top = np.where(top == T['snow'], np.where(slope > 1.05, side, T['gravel']), top)   # the shader lays the snow here
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
    inp = in_patch(X, Z, -cell) if cell == 8 else np.zeros(X.shape, bool)
    H = np.where(in_patch(X, Z, cell) if cell == 8 else inp, H - 2.0, H)
    gx, gz = np.gradient(H, cell)
    nrm = np.stack([-gx, np.ones_like(H), -gz], -1); nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    shade = (0.9 + 0.1 * fbm(X, 0.0, Z, 11, 64, 2)) * (0.92 + 0.16 * fbm(X, 0.0, Z, 260, 66, 3)) \
        * (1.0 - 0.3 * np.clip(gul / 12, 0, 1) + 0.08 * np.clip(-gul / 12, 0, 1))
    sw = np.where(can, 0.0, sw)
    # cells: drawn unless entirely inside the finer ring or behind him
    cx, cz = X[:-1, :-1] + cell / 2, Z[:-1, :-1] + cell / 2
    draw = (yaw_of(cx, cz) > YAW_LO) & (yaw_of(cx, cz) < YAW_HI) & ~(insq[:-1, :-1] & insq[1:, 1:] & insq[1:, :-1] & insq[:-1, 1:])
    draw &= ~(inp[:-1, :-1] & inp[1:, 1:] & inp[1:, :-1] & inp[:-1, 1:])
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
            t4 = np.zeros((P.shape[0], 4), np.uint8); t4[:, 0] = top[sl].reshape(-1); t4[:, 1] = side[sl].reshape(-1); t4[:, 2] = side[sl].reshape(-1); t4[:, 3] = np.round(sw[sl].reshape(-1) * 255)
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
    print('fall patches: %d quads' % fall_patches(B), flush=True)
    for cell, half in ((8, 3600), (16, 7200), (32, 14400)):
        nt_ = smooth_ring(cell, half, prev, 'lod5s%d' % cell, meta['groups']); prev = (SX - half, SX + half, SZ - half, SZ + half)
        print('smooth ring %d: %d tris %.0fs' % (cell, nt_, time.time() - t0), flush=True)
    nq = B.save('lod5', meta['groups'])
    nt = trees(meta)
    json.dump(meta, open(mp, 'w'))
    print('lod5 quads %d (tris %d), trees %d, %.0fs' % (nq, nq * 2, nt, time.time() - t0))
