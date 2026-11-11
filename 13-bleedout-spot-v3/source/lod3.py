"""Coarse terrain beyond the detailed blocks: columns of 8-block cells out to ~3 km and 64-block cells to ~14 km,
vertex-coloured (grass / moss tops, banded rock walls, snow on the high spires, pink groves on some pillar tops).
Anything under the cloud sea and anything behind him (he never looks east) is left out."""
import json, math, os, time
import numpy as np
import hf3, mesher
from lib import fbm
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
SX, SZ = hf3.SEAT
DET = (-400, 160, -280, 280)             # the detailed block area (x0, x1, z0, z1)


def srgb(c):
    return (np.asarray(c, np.float32) / 255) ** 2.2


def ring(cell, half, inner, name, B, yaw_lo=-35, yaw_hi=215):
    n = int(2 * half / cell)
    xs = SX - half + (np.arange(n) + 0.5) * cell; zs = SZ - half + (np.arange(n) + 0.5) * cell
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    H = np.round(hf3.height(X, Z) / (1 if cell <= 8 else 4)) * (1 if cell <= 8 else 4)
    yaw = np.degrees(np.arctan2(-(X - SX), -(Z - SZ))) % 360
    yaw = np.where(yaw > 300, yaw - 360, yaw)
    keep = (yaw > yaw_lo) & (yaw < yaw_hi) & (H > hf3.CLOUD_LO + 5)
    if inner is not None:
        x0, x1, z0, z1 = inner
        keep &= ~((X + cell / 2 > x0) & (X - cell / 2 < x1) & (Z + cell / 2 > z0) & (Z - cell / 2 < z1))
    # colours
    pad = np.pad(H, 1, mode='edge')
    slope = np.max(np.abs(np.stack([pad[1:-1, 1:-1] - pad[:-2, 1:-1], pad[1:-1, 1:-1] - pad[2:, 1:-1],
                                     pad[1:-1, 1:-1] - pad[1:-1, :-2], pad[1:-1, 1:-1] - pad[1:-1, 2:]])), 0) / cell
    n1 = fbm(X, 0.0, Z, 90, 61, 3); n2 = fbm(X, 0.0, Z, 23, 62, 2)
    grass = srgb((96, 150, 60)) * (0.85 + 0.3 * n2[..., None]); forest = srgb((50, 96, 40)) * (0.8 + 0.4 * n2[..., None])
    pink = srgb((240, 166, 204)) * (0.9 + 0.2 * n2[..., None]); rock = srgb((160, 146, 132)) * (0.85 + 0.3 * n1[..., None])
    snow = srgb((240, 244, 252))
    top = np.where((n2 > 0.55)[..., None], forest, grass)
    pil = (hf3.pillars(X, Z) >= H - 2) & (H > hf3.CLOUD_HI)
    top = np.where((pil & (n1 > 0.58))[..., None], pink, top)
    top = np.where((slope > 1.4)[..., None], rock, top)
    top = np.where((H > 980 + 120 * n1)[..., None], snow, top)
    top = np.where((H > 640 + 80 * n1)[..., None] & (slope > 0.9)[..., None], srgb((150, 138, 156)) * (0.85 + 0.3 * n2[..., None]), top)
    ii, kk = np.nonzero(keep)
    if len(ii) == 0:
        return
    h = H[ii, kk]; x = X[ii, kk]; z = Z[ii, kk]
    c = cell / 2
    keys = ((x // 512).astype(np.int64) * 100000 + (z // 512).astype(np.int64))
    pos = np.stack([np.stack([x - c, h, z - c], 1), np.stack([x - c, h, z + c], 1), np.stack([x + c, h, z + c], 1), np.stack([x + c, h, z - c], 1)], 1)
    col = np.repeat(top[ii, kk][:, None, :] * 255, 4, 1)
    B.add(keys, pos.astype(np.float32), np.broadcast_to(np.array([0, 127, 0], np.int8), (len(ii), 3)), np.zeros((len(ii), 4, 2), np.float32), col)
    # walls toward lower neighbours, banded rock, darker toward the foot
    for (di, dk), nrm in (((1, 0), (1, 0, 0)), ((-1, 0), (-1, 0, 0)), ((0, 1), (0, 0, 1)), ((0, -1), (0, 0, -1))):
        ni, nk = ii + di, kk + dk
        ok = (ni >= 0) & (ni < n) & (nk >= 0) & (nk < n)
        hn = np.full(len(ii), -1e9); hn[ok] = H[ni[ok], nk[ok]]
        lo = np.maximum(hn, hf3.CLOUD_LO - 10)
        m = h > lo + 0.5
        if not m.any():
            continue
        xm, zm, hm, lom = x[m], z[m], h[m], lo[m]
        ex, ez = xm + di * c, zm + dk * c
        tx, tz = (-dk, di)
        a = np.stack([ex - tx * c, hm, ez - tz * c], 1); b = np.stack([ex + tx * c, hm, ez + tz * c], 1)
        a2 = np.stack([ex - tx * c, lom, ez - tz * c], 1); b2 = np.stack([ex + tx * c, lom, ez + tz * c], 1)
        quad = np.stack([a, b, b2, a2], 1)
        if np.dot(np.cross(quad[0, 1] - quad[0, 0], quad[0, 3] - quad[0, 0]), nrm) < 0:
            quad = quad[:, [0, 3, 2, 1]]
        band = 0.86 + 0.14 * np.sin(hm / 9.0 + xm * 0.01) + 0.08 * (fbm(xm, hm, zm, 50, 63, 2) - 0.5)
        base = srgb((166, 150, 134))
        ctop = base * band[:, None]; cbot = base * band[:, None] * 0.55
        mossy = (fbm(xm, 0.0, zm, 40, 64, 2) > 0.6)
        ctop = np.where(mossy[:, None], srgb((92, 126, 62)), ctop)
        col = np.stack([ctop, ctop, cbot, cbot], 1) * 255
        if quad.shape[1] == 4 and np.dot(np.cross(np.array(b[0]) - np.array(a[0]), np.array(a2[0]) - np.array(a[0])), nrm) < 0:
            col = col[:, [0, 3, 2, 1]]
        keys_m = keys[m]
        B.add(keys_m, quad.astype(np.float32), np.broadcast_to(np.array(nrm, np.int8) * 127, (m.sum(), 3)), np.zeros((m.sum(), 4, 2), np.float32), col)


if __name__ == '__main__':
    t0 = time.time()
    B = mesher.Builder()
    ring(8, 2944, DET, 'near', B)
    print('near %.0fs' % (time.time() - t0), flush=True)
    ring(64, 14336, (SX - 2944, SX + 2944, SZ - 2944, SZ + 2944), 'far', B)
    g = B.build()
    mp = os.path.join(WEB, 'meta.json'); meta = json.load(open(mp))
    mesher.save_groups({'lod': g}, WEB, 'lod', meta['groups'])
    json.dump(meta, open(mp, 'w'))
    print('lod: %d verts %d tris, %.0fs' % (len(g['pos']), len(g['idx']) // 3, time.time() - t0))
