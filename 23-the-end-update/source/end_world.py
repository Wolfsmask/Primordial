"""The End, rebuilt for "the End Update": the main island (obsidian pillars, crystals, the exit portal), and round it
floating isles - an ender grove of glowing trees, an end city, crystal spires, a chorus forest, a gateway, debris.
python3 end_world.py -> web/*.bin, web/meta.json, web/scene.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deps'))
from lib import fbm
import end_mesher as M

HERE = os.path.dirname(os.path.abspath(__file__))
B = M.B


class Vol:
    def __init__(self, lo, hi):
        self.lo = np.array(lo, int); self.hi = np.array(hi, int)
        self.v = np.zeros(tuple(self.hi - self.lo), np.uint8)

    def set(self, x, y, z, b):
        x = np.asarray(x) - self.lo[0]; y = np.asarray(y) - self.lo[1]; z = np.asarray(z) - self.lo[2]
        ok = (x >= 0) & (y >= 0) & (z >= 0) & (x < self.v.shape[0]) & (y < self.v.shape[1]) & (z < self.v.shape[2])
        self.v[np.asarray(x)[ok], np.asarray(y)[ok], np.asarray(z)[ok]] = b

    def get(self, x, y, z):
        x -= self.lo[0]; y -= self.lo[1]; z -= self.lo[2]
        if 0 <= x < self.v.shape[0] and 0 <= y < self.v.shape[1] and 0 <= z < self.v.shape[2]: return self.v[x, y, z]
        return 0


def island(V, cx, cy, cz, R, depth, seed, top_amp=3.0, surf=None, moss=0.55):
    """an end-stone island: a lumpy plate on top, a long ragged cone underneath.  returns the top height map (dict)"""
    xs = np.arange(int(cx - R - 8), int(cx + R + 9)); zs = np.arange(int(cz - R - 8), int(cz + R + 9))
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    d = np.hypot(X - cx, Z - cz) / R + 0.32 * (fbm(X, 0, Z, R * 0.5, seed) - 0.5) + 0.12 * (fbm(X, 7, Z, R * 0.15, seed + 1) - 0.5)
    inside = d < 1.0
    top = cy + top_amp * (1 - np.clip(d, 0, 1) ** 2) + 2.4 * (fbm(X, 3, Z, 9, seed + 2) - 0.5)
    bot = cy - 1 - depth * np.clip(1 - d, 0, 1) ** 1.25 * (0.55 + 0.9 * fbm(X, 11, Z, 7, seed + 3)) - 2.0 * fbm(X, 5, Z, 3, seed + 4)
    tops = {}
    for i in range(len(xs)):
        for k in range(len(zs)):
            if not inside[i, k]: continue
            y1 = int(np.floor(top[i, k])); y0 = int(np.floor(bot[i, k]))
            ys = np.arange(y0, y1 + 1)
            V.set(np.full(len(ys), xs[i]), ys, np.full(len(ys), zs[k]), B['end_stone'])
            tops[(int(xs[i]), int(zs[k]))] = y1
    # the surface: void moss in patches (or the given block), glowing tufts on the moss
    rng = np.random.default_rng(seed)
    for (x, z), y in tops.items():
        mp = fbm(x, 13, z, 11, seed + 5)
        if surf is not None: V.set(x, y, z, B[surf])
        elif mp > moss:
            V.set(x, y, z, B['void_moss'])
            if rng.random() < 0.10: V.set(x, y + 1, z, B['end_grass'])
    return tops


def cylinder(V, cx, cz, y0, y1, r, b):
    for x in range(int(cx - r - 1), int(cx + r + 2)):
        for z in range(int(cz - r - 1), int(cz + r + 2)):
            if (x + 0.5 - cx) ** 2 + (z + 0.5 - cz) ** 2 <= r * r:
                ys = np.arange(int(y0), int(y1) + 1); V.set(np.full(len(ys), x), ys, np.full(len(ys), z), b)


def box(V, x0, y0, z0, x1, y1, z1, b, hollow=False):
    X, Y, Z = np.mgrid[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]
    if hollow:
        m = (X == x0) | (X == x1) | (Y == y0) | (Y == y1) | (Z == z0) | (Z == z1)
        X, Y, Z = X[m], Y[m], Z[m]
    V.set(X.ravel(), Y.ravel(), Z.ravel(), b)


def mesh(V, prefix, meta):
    g = M.mesh_volume(V.v, tuple(V.lo))
    M.save_groups(g, os.path.join(HERE, 'web'), prefix, meta['groups'])
    print(prefix, V.v.shape, {k: (int(len(x['idx']) // 3) if x else 0) for k, x in g.items()})


def main():
    meta = dict(groups={}); S = dict(crystals=[], portal=None, gateway=None, falls=[], glows=[], lamps=[], islets=[])
    rng = np.random.default_rng(42)
    # ============================================================== the main island
    V = Vol((-72, -56, -72), (73, 62, 73))
    tops = island(V, 0, 0, 0, 58, 44, 1, top_amp=3.5, moss=0.62)
    gy = lambda x, z: tops.get((int(np.floor(x)), int(np.floor(z))), 0)
    for i in range(10):
        a = np.radians(i * 36 + 9); px, pz = 36 * np.cos(a), 36 * np.sin(a)
        H = 24 + [0, 8, 16, 4, 20, 12, 6, 18, 2, 10][i]; r = 2.6 + (i % 3) * 0.7
        y0 = gy(px, pz) - 3
        cylinder(V, px, pz, y0, H, r, B['obsidian'])
        if i % 3 == 0:                                       # new: some pillars split with glowing veins
            for y in range(int(y0) + 5, H - 2, 7): cylinder(V, px, pz, y, y + 1, r + 0.01, B['cracked_obsidian'])
        V.set(int(np.floor(px)), H + 1, int(np.floor(pz)), B['bedrock'])
        S['crystals'].append([float(np.floor(px) + 0.5), float(H + 3.2), float(np.floor(pz) + 0.5)])
    # the exit portal: a bedrock bowl, a pillar, four end rods
    g0 = gy(0, 0)
    for x in range(-5, 6):
        for z in range(-5, 6):
            rr = np.hypot(x + 0.5, z + 0.5)
            if rr <= 4.6: V.set(x, g0 - 1, z, B['bedrock'])
            if 3.6 <= rr <= 4.6: V.set(x, g0, z, B['bedrock'])
            if rr < 3.6: V.set(x, g0, z, 0)
    for y in range(g0, g0 + 5): V.set(0, y, 0, B['bedrock'])
    for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)): V.set(dx, g0 + 3, dz, B['end_rod'])
    S['portal'] = dict(c=[0.5, g0 + 0.85, 0.5], r=3.5)
    # the rim: where void-light pours off the edge (new: "void falls")
    rim = [(x, z, y) for (x, z), y in tops.items() if 50 <= np.hypot(x, z) <= 60]
    for k in rng.choice(len(rim), 14, replace=False):
        x, z, y = rim[k]; S['falls'].append([x + 0.5, y + 0.5, z + 0.5])
    mesh(V, 'main', meta)
    # ============================================================== the ender grove (new): glowing trees on a mossy isle
    gc = np.array([98, -14, -58])
    V = Vol(gc + [-36, -36, -36], gc + [37, 30, 37])
    tops = island(V, *gc, 26, 26, 7, top_amp=2.5, moss=0.25)
    for k in range(8):
        a = rng.uniform(0, 6.283); rr = rng.uniform(2, 17); tx, tz = int(gc[0] + rr * np.cos(a)), int(gc[2] + rr * np.sin(a))
        ty = tops.get((tx, tz))
        if ty is None: continue
        h = rng.integers(6, 11)
        for y in range(ty + 1, ty + h + 1): V.set(tx, y, tz, B['bloom_log'])
        cr = rng.uniform(3.0, 4.6); cyy = ty + h + 1
        for x in range(int(tx - cr - 1), int(tx + cr + 2)):
            for y in range(int(cyy - cr), int(cyy + cr * 0.8) + 1):
                for z in range(int(tz - cr - 1), int(tz + cr + 2)):
                    dd = ((x - tx) ** 2 + ((y - cyy) * 1.3) ** 2 + (z - tz) ** 2) ** 0.5
                    if dd < cr - 0.6 * fbm(x, y, z, 2.0, 21) and V.get(x, y, z) == 0: V.set(x, y, z, B['bloom_leaves'])
        for j in range(5):                                    # hanging glow fruit
            fx, fz = tx + rng.integers(-3, 4), tz + rng.integers(-3, 4)
            for y in range(int(cyy - cr), int(cyy - cr) - 3, -1):
                if V.get(fx, y, fz) == 0: V.set(fx, y, fz, B['chorus_flower']); break
        S['glows'].append([tx + 0.5, cyy + 0.5, tz + 0.5, cr])
    rim = [(x, z, y) for (x, z), y in tops.items() if np.hypot(x - gc[0], z - gc[2]) >= 21]
    for k in rng.choice(len(rim), 6, replace=False):
        x, z, y = rim[k]; S['falls'].append([x + 0.5, y + 0.5, z + 0.5])
    mesh(V, 'grove', meta)
    # ============================================================== the end city: purpur towers on their own isle
    cc = np.array([-108, 2, -46])
    V = Vol(cc + [-40, -36, -40], cc + [41, 72, 41])
    tops = island(V, *cc, 30, 30, 9, top_amp=2.0, moss=0.8)
    for (ox, oz, Ht) in ((-8, -4, 52), (9, 6, 38), (-2, 14, 30)):
        bx, bz = cc[0] + ox, cc[2] + oz; by = max(tops.get((bx, bz), cc[1]), cc[1]) + 1
        box(V, bx - 4, by, bz - 4, bx + 4, by + 3, bz + 4, B['end_bricks'])
        y = by + 4
        while y < by + Ht:
            box(V, bx - 2, y, bz - 2, bx + 2, y + 5, bz + 2, B['purpur'])
            for cx_, cz_ in ((-2, -2), (-2, 2), (2, -2), (2, 2)):
                for yy in range(y, y + 6): V.set(bx + cx_, yy, bz + cz_, B['purpur_pillar'])
            y += 6
            if (y - by) % 18 == 4 or y >= by + Ht - 6:                     # a wide room with gold trim and lanterns
                box(V, bx - 5, y, bz - 5, bx + 5, y + 5, bz + 5, B['purpur'], hollow=True)
                box(V, bx - 5, y + 5, bz - 5, bx + 5, y + 5, bz + 5, B['gold_trim'])
                for cx_, cz_ in ((-5, -5), (-5, 5), (5, -5), (5, 5)):
                    V.set(bx + cx_, y + 6, bz + cz_, B['end_rod']); S['lamps'].append([bx + cx_ + 0.5, y + 6.5, bz + cz_ + 0.5])
                for k in range(1, 5):                                  # windows
                    for s in (-5, 5): V.set(bx + s, y + 2, bz - 2 + k, 0); V.set(bx - 2 + k, y + 2, bz + s, 0)
                y += 6
        for k in range(4):                                             # the stepped roof
            box(V, bx - 4 + k, y + k, bz - 4 + k, bx + 4 - k, y + k, bz + 4 - k, B['purpur'])
        V.set(bx, y + 4, bz, B['end_rod']); S['lamps'].append([bx + 0.5, y + 5.0, bz + 0.5])
    mesh(V, 'city', meta)
    # ============================================================== crystal spires (new)
    kc = np.array([42, 20, 120])
    V = Vol(kc + [-34, -30, -34], kc + [35, 40, 35])
    tops = island(V, *kc, 20, 20, 13, top_amp=2.0, surf='shardstone')
    for k in range(7):
        a = rng.uniform(0, 6.283); rr = rng.uniform(0, 12); sx, sz = kc[0] + rr * np.cos(a), kc[2] + rr * np.sin(a)
        sy = tops.get((int(sx), int(sz)), kc[1]); L = rng.uniform(10, 26); w = rng.uniform(1.2, 2.4)
        d = np.array([np.cos(a) * 0.45, 1.0, np.sin(a) * 0.45]); d /= np.linalg.norm(d)
        for t in np.arange(0, L, 0.4):
            c = np.array([sx, sy, sz]) + d * t; ww = w * (1 - t / L) + 0.4
            for x in range(int(c[0] - ww - 1), int(c[0] + ww + 2)):
                for z in range(int(c[2] - ww - 1), int(c[2] + ww + 2)):
                    if abs(x + 0.5 - c[0]) + abs(z + 0.5 - c[2]) <= ww: V.set(x, int(c[1]), z, B['void_crystal'])
        S['glows'].append([float(sx + d[0] * L * 0.5), float(sy + L * 0.5), float(sz + d[2] * L * 0.5), 3.0])
    mesh(V, 'crystal', meta)
    # ============================================================== the chorus forest
    hc = np.array([-60, -26, 104])
    V = Vol(hc + [-34, -30, -34], hc + [35, 30, 35])
    tops = island(V, *hc, 22, 22, 17, top_amp=2.0, moss=0.7)
    def chorus(x, y, z, n, depth=0):
        for i in range(n):
            V.set(x, y + i, z, B['chorus_stem'])
        y += n
        if depth < 3:
            for _ in range(rng.integers(1, 4)):
                dx, dz = [(1, 0), (-1, 0), (0, 1), (0, -1)][rng.integers(0, 4)]
                L = rng.integers(1, 3)
                for j in range(1, L + 1): V.set(x + dx * j, y - 1, z + dz * j, B['chorus_stem'])
                chorus(x + dx * L, y, z + dz * L, rng.integers(2, 5), depth + 1)
        else:
            V.set(x, y, z, B['chorus_flower'])
    for k in range(16):
        a = rng.uniform(0, 6.283); rr = rng.uniform(0, 15); x, z = int(hc[0] + rr * np.cos(a)), int(hc[2] + rr * np.sin(a))
        if (x, z) in tops: chorus(x, tops[(x, z)] + 1, z, rng.integers(3, 7))
    mesh(V, 'chorus', meta)
    # ============================================================== debris islets, and the gateway
    V = Vol((-230, -110, -230), (231, 110, 231))
    for k in range(18):
        a = rng.uniform(0, 6.283); rr = rng.uniform(75, 210); y = rng.uniform(-70, 60)
        x, z = rr * np.cos(a), rr * np.sin(a)
        if any(np.hypot(x - c[0], z - c[2]) < 45 for c in (gc, cc, kc, hc)): continue
        R = rng.uniform(3, 8); island(V, x, y, z, R, R * 1.4, 100 + k, top_amp=0.8, moss=0.5); S['islets'].append([x, y, z, R])
    gw = np.array([0, 34, -82])
    box(V, gw[0] - 1, gw[1] - 1, gw[2] - 1, gw[0] + 1, gw[1] + 1, gw[2] + 1, B['bedrock']); V.set(*gw, 0)
    for dy in (-2, 2): V.set(gw[0], gw[1] + dy, gw[2], B['bedrock'])
    S['gateway'] = [float(gw[0] + 0.5), float(gw[1] + 0.5), float(gw[2] + 0.5)]
    mesh(V, 'far', meta)
    S['eclipse'] = dict(yaw=205.0, el=17.0)
    json.dump(meta, open(os.path.join(HERE, 'web', 'meta.json'), 'w'))
    json.dump(S, open(os.path.join(HERE, 'web', 'scene.json'), 'w'), indent=1)
    print('scene', {k: (len(v) if isinstance(v, list) else v) for k, v in S.items()})


if __name__ == '__main__':
    main()
