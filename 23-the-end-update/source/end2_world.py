"""The End, rebuilt for "the End Update" (v2).  Everything here is new:
  the main island: the Rift Gate (a colossal obsidian ring with gold inlay, runes and chevrons, over a stepped plinth),
      a lamp-lit causeway to it, guardian obelisks, rune pillars with crystals, floating debris, teal crystal growths
  the end city: pagoda towers of purpur with lit windows, eaves, banners, spires and bridges
  crystal spires: giant hexagonal crystals (violet and teal) on a dark shardstone isle, floating shards
  the elder grove: a giant glowing tree, hanging vines, mushrooms, a pool spilling into the void as light falls
  the far End: big islands, colossal pillars and broken towers out to 1.5 km (coarse LOD)
python3 end2_world.py -> web/*.bin, web/meta.json, web/scene.json"""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, 'deps'))
from lib import fbm
import end2_mesher as M

B = M.B


def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-12)


class Vol:
    def __init__(self, lo, hi):
        self.lo = np.array(lo, int); self.hi = np.array(hi, int)
        self.v = np.zeros(tuple(self.hi - self.lo), np.uint8)

    def set(self, x, y, z, b, only_air=False):
        x = np.asarray(x, int).ravel() - self.lo[0]; y = np.asarray(y, int).ravel() - self.lo[1]; z = np.asarray(z, int).ravel() - self.lo[2]
        n = max(len(x), len(y), len(z)); x, y, z = (np.broadcast_to(a, (n,)) for a in (x, y, z))
        ok = (x >= 0) & (y >= 0) & (z >= 0) & (x < self.v.shape[0]) & (y < self.v.shape[1]) & (z < self.v.shape[2])
        x, y, z = x[ok], y[ok], z[ok]
        if only_air:
            a = self.v[x, y, z] == 0; x, y, z = x[a], y[a], z[a]
        self.v[x, y, z] = b

    def get(self, x, y, z):
        x -= self.lo[0]; y -= self.lo[1]; z -= self.lo[2]
        if 0 <= x < self.v.shape[0] and 0 <= y < self.v.shape[1] and 0 <= z < self.v.shape[2]: return self.v[x, y, z]
        return 0

    def grid(self, lo, hi):
        lo = np.maximum(np.floor(lo).astype(int), self.lo); hi = np.minimum(np.ceil(hi).astype(int) + 1, self.hi)
        if np.any(hi <= lo): return None
        return np.mgrid[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]

    def put(self, G, m, b, only_air=False):
        X, Y, Z = G; self.set(X[m], Y[m], Z[m], b, only_air)


def box(V, x0, y0, z0, x1, y1, z1, b, only_air=False):
    x0, x1 = sorted((int(x0), int(x1))); y0, y1 = sorted((int(y0), int(y1))); z0, z1 = sorted((int(z0), int(z1)))
    X, Y, Z = np.mgrid[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1]; V.set(X, Y, Z, b, only_air)


def ring_xz(V, cx, y, cz, half, b, y1=None):
    """the square ring |x|==half or |z|==half (one block wide), from y to y1"""
    y1 = y if y1 is None else y1
    for yy in range(int(y), int(y1) + 1):
        for k in range(-half, half + 1):
            V.set([cx + k, cx + k, cx - half, cx + half], yy, [cz - half, cz + half, cz + k, cz + k], b)


def island(V, cx, cy, cz, R, depth, seed, top_amp=3.0, moss=0.55, block='end_stone', surf=None, deco=True):
    """an island: a lumpy plate on top, a long ragged cone underneath.  -> dict (x, z) -> top y"""
    xs = np.arange(int(cx - R - 8), int(cx + R + 9)); zs = np.arange(int(cz - R - 8), int(cz + R + 9))
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    d = np.hypot(X - cx, Z - cz) / R + 0.30 * (fbm(X, 0, Z, R * 0.5, seed) - 0.5) + 0.12 * (fbm(X, 7, Z, max(R * 0.15, 2), seed + 1) - 0.5)
    inside = d < 1.0
    top = cy + top_amp * (1 - np.clip(d, 0, 1) ** 2) + 2.0 * (fbm(X, 3, Z, 9, seed + 2) - 0.5)
    bot = cy - 1 - depth * np.clip(1 - d, 0, 1) ** 1.2 * (0.55 + 0.9 * fbm(X, 11, Z, 7, seed + 3)) - 2.0 * fbm(X, 5, Z, 3, seed + 4)
    top = np.floor(top).astype(int); bot = np.floor(bot).astype(int)
    ys = np.arange(bot[inside].min(), top[inside].max() + 1)
    m = inside[:, None, :] & (ys[None, :, None] >= bot[:, None, :]) & (ys[None, :, None] <= top[:, None, :])
    I, J, K = np.nonzero(m); V.set(xs[I], ys[J], zs[K], B[block])
    ii, kk = np.nonzero(inside); tops = {(int(xs[i]), int(zs[k])): int(top[i, k]) for i, k in zip(ii, kk)}
    rng = np.random.default_rng(seed)
    if surf is not None:
        V.set(xs[ii], top[ii, kk], zs[kk], B[surf])
    else:
        mp = fbm(X, 13, Z, 11, seed + 5)
        sel = inside & (mp > moss); a, c = np.nonzero(sel)
        V.set(xs[a], top[a, c], zs[c], B['void_moss'])
        if deco:
            g = rng.random(len(a)) < 0.12; V.set(xs[a[g]], top[a[g], c[g]] + 1, zs[c[g]], B['end_grass'])
            g = rng.random(len(a)) < 0.012; V.set(xs[a[g]], top[a[g], c[g]] + 1, zs[c[g]], B['glow_shroom'])
    return tops


def capsule(V, a, b, ra, rb, block, only_air=False):
    a = np.asarray(a, float); b = np.asarray(b, float); r = max(ra, rb) + 1
    G = V.grid(np.minimum(a, b) - r, np.maximum(a, b) + r)
    if G is None: return
    P = np.stack(G, -1) + 0.5; ab = b - a; t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    dist = np.linalg.norm(P - (a + t[..., None] * ab), axis=-1); V.put(G, dist <= ra + (rb - ra) * t, block, only_air)


def hex_crystal(V, base, d, L, w, block, tip=0.32, only_air=False):
    """a hexagonal prism from base along d, length L, radius w, with a pointed tip; returns the tip point"""
    base = np.asarray(base, float); d = unit(d); e1 = unit(np.cross(d, [0.3, 0.1, 1.0])); e2 = np.cross(d, e1)
    end = base + d * L; r = w + 1.5
    G = V.grid(np.minimum(base, end) - r, np.maximum(base, end) + r)
    if G is None: return end
    P = np.stack(G, -1) + 0.5 - base; t = P @ d; R = P - t[..., None] * d; a1 = R @ e1; a2 = R @ e2
    hx = np.maximum(np.abs(a1), np.maximum(np.abs(0.5 * a1 + 0.866 * a2), np.abs(0.5 * a1 - 0.866 * a2)))
    wt = np.where(t < L * (1 - tip), w, w * np.clip((L - t) / (L * tip), 0, 1))
    V.put(G, (t >= -w) & (t <= L) & (hx <= wt), block, only_air)
    return end


def mesh(V, prefix, meta, scale=1):
    g = M.mesh_volume(V.v, tuple(V.lo * scale), scale=scale)
    M.save_groups(g, os.path.join(HERE, 'web'), prefix, meta['groups'])
    print(prefix, V.v.shape, {k: (int(len(x['idx']) // 3) if x else 0) for k, x in g.items()}, flush=True)


# ============================================================================== the main island and the Rift Gate
def main_island(meta, S, rng):
    V = Vol((-78, -60, -78), (79, 100, 79))
    tops = island(V, 0, 0, 0, 62, 50, 1, top_amp=3.0, moss=0.60)
    gy = lambda x, z: tops.get((int(np.floor(x)), int(np.floor(z))), 0)
    G0 = 3                                                     # the plinth's floor
    # the plinth: a broad stepped platform under the gate
    box(V, -34, G0 - 4, -13, 34, G0, 13, B['end_stone_smooth'])
    ring_xz_rect = lambda x0, x1, z0, z1, y, b: (box(V, x0, y, z0, x1, y, z0, b), box(V, x0, y, z1, x1, y, z1, b), box(V, x0, y, z0, x0, y, z1, b), box(V, x1, y, z0, x1, y, z1, b))
    ring_xz_rect(-34, 34, -13, 13, G0, B['gold_trim'])
    box(V, -27, G0 + 1, -9, 27, G0 + 2, 9, B['purpur_tile']); ring_xz_rect(-27, 27, -9, 9, G0 + 2, B['chiseled_purpur'])
    for yy in range(G0 - 3, G0 + 1):                            # skirt down to the ground so it doesn't float
        box(V, -34, yy - 6, -13, 34, yy - 6, 13, B['end_stone'], only_air=True)
    # the ring
    C = np.array([0.0, G0 + 28.0, 0.0]); RIN, ROUT = 20.0, 26.0
    Gd = V.grid(C - [34, 34, 5], C + [34, 34, 5]); X, Y, Z = Gd
    qx = X + 0.5 - C[0]; qy = Y + 0.5 - C[1]; qz = Z + 0.5 - C[2]; rho = np.hypot(qx, qy); ph = np.degrees(np.arctan2(qy, qx)); az = np.abs(qz)
    hd = np.where(rho < 21.2, 1.6, np.where(rho < 25.0, 3.6, 2.6))
    body = (rho >= RIN) & (rho <= ROUT + 0.5) & (az <= hd)
    V.put(Gd, body, B['obsidian'])
    V.put(Gd, body & (rho < 21.2), B['cracked_obsidian'])
    face = body & (az > 2.6)
    V.put(Gd, face & (((rho >= 21.6) & (rho < 22.4)) | ((rho >= 24.0) & (rho < 24.9))), B['gold_trim'])
    rune_ang = (np.mod(ph + 15.0, 30.0) < 3.6)
    V.put(Gd, face & (rho >= 22.6) & (rho < 23.8) & rune_ang, B['rune_obsidian'])
    V.put(Gd, body & (rho > 25.0), B['purpur_tile'])
    chevs = []
    for a in np.arange(90, 90 + 360, 40):
        a = (a + 180) % 360 - 180
        if abs(((a + 90) + 180) % 360 - 180) < 55: continue    # not down in the plinth
        da = np.abs((ph - a + 180) % 360 - 180)
        ch = (da < 6.5 - (rho - 26) * 0.9) & (rho > 25.0) & (rho <= 31.0) & (az <= 2.6)
        V.put(Gd, ch, B['purpur'])
        V.put(Gd, ch & (rho > 29.6), B['gold_block'])
        V.put(Gd, ch & (rho > 26.6) & (rho < 28.6) & (da < 2.6), B['void_crystal'])
        ar = np.radians(a); chevs.append([float(C[0] + np.cos(ar) * 27.6), float(C[1] + np.sin(ar) * 27.6), float(C[2] - 2.7)])
    # buttresses holding the ring's foot
    for s in (-1, 1):
        for k in range(5):
            box(V, s * (12 + k * 2), G0 + 1, -6 + k, s * (24 - k), G0 + 6 + k * 2 - 2 * (k == 4), 6 - k, B['obsidian'])
        box(V, s * 16, G0 + 3, -6, s * 21, G0 + 3, 6, B['gold_trim'])
    # stairs up to the threshold (in front, -z), the threshold itself
    thr = int(C[1] - RIN)                                       # inner foot of the ring
    for k in range(thr - G0 - 2):
        y = G0 + 3 + k; z0 = -10 - (thr - G0 - 3 - k) * 2
        box(V, -6, y, z0, 6, y, z0 + 1, B['end_stone_smooth']); box(V, -6, y - 1, z0, 6, y - 1, -2, B['end_stone_smooth'], only_air=True)
        V.set([-7, 7], y, z0, B['purpur_pillar']); V.set([-7, 7], y, z0 + 1, B['purpur_pillar'])
    box(V, -8, G0 + 3, -9, 8, thr - 1, 9, B['obsidian'], only_air=True)
    # guardian obelisks either side of the stairs
    for s in (-1, 1):
        x = s * 15; z = -19
        box(V, x - 2, G0 + 1, z - 2, x + 2, G0 + 2, z + 2, B['purpur_tile'])
        box(V, x - 1, G0 + 3, z - 1, x + 1, G0 + 20, z + 1, B['obsidian'])
        for y in range(G0 + 6, G0 + 20, 5): box(V, x - 1, y, z - 1, x + 1, y, z + 1, B['rune_obsidian'])
        box(V, x - 2, G0 + 21, z - 2, x + 2, G0 + 21, z + 2, B['gold_trim'])
        box(V, x - 1, G0 + 22, z - 1, x + 1, G0 + 23, z + 1, B['chiseled_purpur']); V.set(x, G0 + 24, z, B['gold_block'])
        S['crystals'].append([x + 0.5, G0 + 27.5, z + 0.5])
    # the causeway: -z, from the stairs to the rim
    for z in range(-60, -21):
        g = gy(0, z)
        box(V, -5, G0 - 1, z, 5, G0, z, B['end_stone_smooth']); V.set([-5, 5], G0, z, B['purpur_tile']); V.set([-3, 3], G0, z, B['gold_trim'])
        for x in range(-5, 6):
            for yy in range(G0 - 2, min(G0 - 1, g + 1) - 12, -1):
                if V.get(x, yy, z) == 0: V.set(x, yy, z, B['end_stone'])
                else: break
        if z % 8 == 0:                                          # lamp posts
            for s in (-1, 1):
                x = s * 7
                box(V, x, G0 - 3, z, x, G0 + 5, z, B['purpur_pillar']); V.set(x, G0 + 6, z, B['chiseled_purpur']); V.set(x, G0 + 7, z, B['end_rod'])
                V.set(x - s, G0 + 5, z, B['purpur']); box(V, x - s, G0 + 2, z, x - s, G0 + 4, z, B['banner'])
                S['lamps'].append([x + 0.5, G0 + 7.6, z + 0.5])
    # the rune pillars and their crystals (clear of the causeway)
    for i, (ang, H, r) in enumerate(zip([-130, -50, -10, 30, 70, 110, 150, -170], [40, 46, 34, 52, 44, 50, 36, 42], [3.4, 3.8, 3.2, 4.2, 3.6, 4.0, 3.2, 3.6])):
        a = np.radians(ang); px, pz = 42 * np.cos(a), 42 * np.sin(a); y0 = gy(px, pz) - 4
        Gp = V.grid([px - r - 1, y0, pz - r - 1], [px + r + 1, H, pz + r + 1]); X, Y, Z = Gp
        dd = np.hypot(X + 0.5 - px, Z + 0.5 - pz); col = dd <= r
        V.put(Gp, col, B['obsidian'])
        V.put(Gp, col & (np.mod(Y - y0, 9) == 4), B['rune_obsidian'])
        V.put(Gp, col & (Y >= H - 3) & (Y <= H - 2), B['gold_trim'])
        V.put(Gp, col & (Y == H) & (dd > r - 1.2), B['chiseled_purpur'])
        V.set(int(np.floor(px)), H + 1, int(np.floor(pz)), B['bedrock'])
        S['crystals'].append([float(np.floor(px) + 0.5), float(H + 3.4), float(np.floor(pz) + 0.5)])
    # teal crystal growths scattered over the island
    for k in range(16):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(18, 54); x, z = rr * np.cos(a), rr * np.sin(a)
        if abs(x) < 12 and z < -8: continue
        if abs(z) < 16 and abs(x) < 36: continue
        y = gy(x, z) + 0.5
        for j in range(rng.integers(2, 5)):
            d = unit([rng.normal(0, 0.45), 1.0, rng.normal(0, 0.45)])
            hex_crystal(V, [x + rng.normal(0, 1.2), y - 1, z + rng.normal(0, 1.2)], d, rng.uniform(4, 11), rng.uniform(0.8, 1.6),
                        B['aether_crystal' if rng.random() < 0.7 else 'void_crystal'])
        S['glows'].append([x, y + 4, z, 2.2, 0.3, 1.0, 1.0])
    # light falls off the rim
    for ang in (-35, 205, 120):
        a = np.radians(ang); o = np.array([np.cos(a), 0, np.sin(a)])
        r_edge = max(rr for rr in np.arange(40, 75, 0.5) if (int(np.floor(o[0] * rr)), int(np.floor(o[2] * rr))) in tops)
        x, z = o[0] * (r_edge - 1), o[2] * (r_edge - 1); y = gy(x, z)
        for k in range(-2, 3):
            px, pz = x - o[2] * k, z + o[0] * k
            for t in range(0, 6): V.set(int(np.floor(px - o[0] * t)), y, int(np.floor(pz - o[2] * t)), 0)
        S['lightfalls'].append([float(x + o[0] * 1.2), float(y - 0.2), float(z + o[2] * 1.2), float(o[0]), float(o[2]), 5.0])
    mesh(V, 'main', meta)
    S['gate'] = dict(c=C.tolist(), rin=RIN, rout=ROUT, chevrons=chevs, floor=G0)
    # floating debris round the gate
    V = Vol((-90, 0, -90), (91, 110, 91))
    for k in range(14):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(32, 70); y = rng.uniform(C[1] + 4, C[1] + 55)
        x, z = rr * np.cos(a), rr * np.sin(a)
        if abs(x) < 10 and z < 0: continue
        R = rng.uniform(1.8, 4.5); island(V, x, y, z, R, R * 1.6, 200 + k, top_amp=0.6, moss=0.4, deco=False)
        if rng.random() < 0.4: hex_crystal(V, [x, y, z], unit([rng.normal(0, .3), 1, rng.normal(0, .3)]), R * 1.6, 0.8, B['aether_crystal'])
    mesh(V, 'debris', meta)


# ============================================================================== the end city
def tower(V, S, bx, by, bz, levels, seed, hs=6, hr=6):
    rng = np.random.default_rng(seed); y = by
    box(V, bx - 7, y - 8, bz - 7, bx + 7, y, bz + 7, B['end_bricks'], only_air=True); box(V, bx - 7, y, bz - 7, bx + 7, y, bz + 7, B['end_bricks'])
    box(V, bx - 6, y + 1, bz - 6, bx + 6, y + 1, bz + 6, B['purpur_tile'])
    y += 2; rooms = []
    for lv in range(levels):
        half = 5 if lv < levels - 1 else 4
        box(V, bx - 2, y, bz - 2, bx + 2, y + hs - 1, bz + 2, B['purpur'])
        for cx, cz in ((-2, -2), (-2, 2), (2, -2), (2, 2)): box(V, bx + cx, y, bz + cz, bx + cx, y + hs - 1, bz + cz, B['purpur_pillar'])
        for s in (-2, 2):
            box(V, bx + s, y + 1, bz, bx + s, y + hs - 2, bz, B['lit_window']); box(V, bx, y + 1, bz + s, bx, y + hs - 2, bz + s, B['lit_window'])
        y += hs
        # the room
        box(V, bx - half, y, bz - half, bx + half, y + hr - 1, bz + half, B['purpur'])
        ring_xz(V, bx, y, bz, half + 1, B['purpur_tile'])                       # ledge
        for cx, cz in ((-1, -1), (-1, 1), (1, -1), (1, 1)): box(V, bx + cx * half, y, bz + cz * half, bx + cx * half, y + hr - 1, bz + cz * half, B['purpur_pillar'])
        for k in (-3, 0, 3) if half >= 5 else (-2, 2):
            for s in (-half, half):
                box(V, bx + s, y + 2, bz + k, bx + s, y + 3, bz + k, B['lit_window']); box(V, bx + k, y + 2, bz + s, bx + k, y + 3, bz + s, B['lit_window'])
        ring_xz(V, bx, y + hr - 1, bz, half, B['gold_trim'])
        # eaves, the pagoda roof
        ring_xz(V, bx, y + hr, bz, half + 1, B['purpur_tile']); ring_xz(V, bx, y + hr, bz, half + 2, B['purpur_tile'])
        box(V, bx - half, y + hr, bz - half, bx + half, y + hr, bz + half, B['purpur'])
        box(V, bx - half + 1, y + hr + 1, bz - half + 1, bx + half - 1, y + hr + 1, bz + half - 1, B['purpur_top' if False else 'purpur'])
        ring_xz(V, bx, y + hr + 1, bz, half - 1, B['chiseled_purpur'])
        for cx, cz in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            V.set(bx + cx * (half + 2), y + hr + 1, bz + cz * (half + 2), B['end_rod']); S['lamps'].append([bx + cx * (half + 2) + 0.5, y + hr + 1.6, bz + cz * (half + 2) + 0.5])
        for s in (-1, 1):                                                      # banners under the eaves
            box(V, bx + s * (half + 1), y + hr - 4, bz, bx + s * (half + 1), y + hr - 1, bz, B['banner'])
            box(V, bx, y + hr - 4, bz + s * (half + 1), bx, y + hr - 1, bz + s * (half + 1), B['banner'])
        rooms.append((y, half)); y += hr + 2
    # the spire
    box(V, bx - 2, y, bz - 2, bx + 2, y + 3, bz + 2, B['purpur'])
    for k in range(7):
        hh = max(0, 2 - k // 3); box(V, bx - hh, y + 4 + k, bz - hh, bx + hh, y + 4 + k, bz + hh, B['purpur' if k % 3 else 'gold_trim'])
    V.set(bx, y + 11, bz, B['gold_block']); top = hex_crystal(V, [bx + 0.5, y + 11.5, bz + 0.5], [0, 1, 0], 7, 1.1, B['aether_crystal'])
    S['glows'].append([bx + 0.5, y + 15, bz + 0.5, 3.0, 0.3, 1.0, 1.0])
    return rooms, y + 11


def bridge(V, S, a, b, y):
    a = np.array(a, float); b = np.array(b, float); d = b - a; L = int(np.linalg.norm(d)); u = d / max(L, 1); n = np.array([-u[1], u[0]])
    for t in range(L + 1):
        p = a + u * t
        for k in range(-1, 2): V.set(int(np.floor(p[0] + n[0] * k)), y, int(np.floor(p[1] + n[1] * k)), B['purpur_tile'])
        V.set(int(np.floor(p[0])), y - 1, int(np.floor(p[1])), B['purpur'])
        for s in (-2, 2):
            q = p + n * s; V.set(int(np.floor(q[0])), y + 1, int(np.floor(q[1])), B['purpur_pillar' if t % 4 == 0 else 'purpur'])
            if t % 8 == 4: V.set(int(np.floor(q[0])), y + 2, int(np.floor(q[1])), B['end_rod']); S['lamps'].append([float(q[0]), y + 2.6, float(q[1])])


def end_city(meta, S, rng):
    cc = np.array([-175, 0, 30])
    V = Vol(cc + [-50, -45, -50], cc + [51, 130, 51])
    tops = island(V, *cc, 42, 40, 9, top_amp=2.0, moss=0.72)
    gy = lambda x, z: tops.get((int(x), int(z)), cc[1])
    T = [(0, 0, 6), (24, -16, 3), (-20, -20, 4), (14, 24, 3), (-26, 16, 2)]
    info = []
    for i, (ox, oz, lv) in enumerate(T):
        bx, bz = cc[0] + ox, cc[2] + oz; by = max(gy(bx, bz), cc[1]) + 1
        rooms, top = tower(V, S, bx, by, bz, lv, 30 + i); info.append((bx, bz, rooms, top))
    for i, j, lvl in ((0, 1, 1), (0, 2, 2), (0, 3, 1), (2, 4, 1)):
        bi, bj = info[i], info[j]; y = min(bi[2][min(lvl, len(bi[2]) - 1)][0], bj[2][min(lvl, len(bj[2]) - 1)][0]) + 0
        bridge(V, S, (bi[0] + 0.5, bi[1] + 0.5), (bj[0] + 0.5, bj[1] + 0.5), y)
    # plazas
    for bx, bz, _, _ in info:
        g = gy(bx, bz)
        for x in range(bx - 10, bx + 11):
            for z in range(bz - 10, bz + 11):
                if (x, z) in tops and abs(tops[(x, z)] - g) <= 2 and (x - bx) ** 2 + (z - bz) ** 2 < 100: V.set(x, tops[(x, z)], z, B['end_stone_smooth'])
    S['city'] = dict(c=cc.tolist(), towers=[[float(b[0]), float(b[3]), float(b[1])] for b in info])
    mesh(V, 'city', meta)


# ============================================================================== crystal spires
def crystal_spires(meta, S, rng):
    kc = np.array([175, 12, 40])
    V = Vol(kc + [-60, -50, -60], kc + [61, 110, 61])
    tops = island(V, *kc, 34, 34, 13, top_amp=2.5, block='shardstone', surf='shardstone')
    gy = lambda x, z: tops.get((int(np.floor(x)), int(np.floor(z))), kc[1])
    tip = hex_crystal(V, [kc[0], gy(*kc[[0, 2]]) - 4, kc[2]], [0.08, 1, -0.05], 78, 7.0, B['void_crystal'], tip=0.25)
    S['glows'].append([*(np.array([kc[0], kc[1] + 40, kc[2]], float)), 14.0, 1.0, 0.45, 1.0])
    S['spires'] = dict(c=kc.tolist(), tip=tip.tolist(), big=[])
    for k in range(9):
        a = k * 2 * np.pi / 9 + rng.uniform(-0.2, 0.2); rr = rng.uniform(9, 24); x, z = kc[0] + rr * np.cos(a), kc[2] + rr * np.sin(a)
        d = unit([np.cos(a) * rng.uniform(0.35, 0.7), 1, np.sin(a) * rng.uniform(0.35, 0.7)]); L = rng.uniform(22, 48); w = rng.uniform(2.6, 4.6)
        blk = B['aether_crystal'] if k % 2 else B['void_crystal']
        t = hex_crystal(V, [x, gy(x, z) - 3, z], d, L, w, blk)
        S['spires']['big'].append([x, gy(x, z) + L * 0.4, z])
        S['glows'].append([x + d[0] * L * 0.5, gy(x, z) + d[1] * L * 0.5, z + d[2] * L * 0.5, w * 2.2, *((0.3, 1.0, 1.0) if k % 2 else (1.0, 0.45, 1.0))])
    for k in range(40):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(4, 32); x, z = kc[0] + rr * np.cos(a), kc[2] + rr * np.sin(a)
        if (int(x), int(z)) not in tops: continue
        d = unit([rng.normal(0, 0.5), 1, rng.normal(0, 0.5)])
        hex_crystal(V, [x, gy(x, z) - 1, z], d, rng.uniform(3, 12), rng.uniform(0.7, 1.8), B['aether_crystal' if rng.random() < 0.5 else 'void_crystal'])
    # shards floating above
    for k in range(16):
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(14, 52); x, z = kc[0] + rr * np.cos(a), kc[2] + rr * np.sin(a); y = kc[1] + rng.uniform(12, 70)
        d = unit(rng.normal(0, 1, 3)); L = rng.uniform(3, 9)
        hex_crystal(V, [x, y, z], d, L, rng.uniform(0.7, 1.5), B['aether_crystal' if rng.random() < 0.5 else 'void_crystal'], tip=0.4)
        hex_crystal(V, [x, y, z], -d, L * 0.5, rng.uniform(0.7, 1.2), B['aether_crystal' if rng.random() < 0.5 else 'void_crystal'], tip=0.5)
    mesh(V, 'spires', meta)


# ============================================================================== the elder grove
def grove(meta, S, rng):
    gc = np.array([30, -28, -175])
    V = Vol(gc + [-55, -45, -55], gc + [56, 70, 56])
    tops = island(V, *gc, 40, 36, 7, top_amp=2.5, moss=0.18)
    gy = lambda x, z: tops.get((int(np.floor(x)), int(np.floor(z))), gc[1])
    def tree(x, z, H, R0, canopy, seed):
        rr = np.random.default_rng(seed); y0 = gy(x, z)
        base = np.array([x, y0 - 2, z], float); top = base + [rr.normal(0, 1.5), H, rr.normal(0, 1.5)]
        capsule(V, base, top, R0, R0 * 0.55, B['bloom_log'])
        for k in range(6):                                              # roots
            a = k * np.pi / 3 + rr.uniform(-0.3, 0.3); o = np.array([np.cos(a), 0, np.sin(a)])
            capsule(V, base + [0, R0 * 1.6, 0], base + o * R0 * 2.6 + [0, -1, 0], R0 * 0.5, R0 * 0.25, B['bloom_log'])
        ends = []
        for k in range(5):
            a = k * 2 * np.pi / 5 + rr.uniform(-0.3, 0.3); o = np.array([np.cos(a), 0, np.sin(a)]); h0 = H * rr.uniform(0.55, 0.8)
            p0 = base + [0, h0, 0]; p1 = p0 + o * canopy * 0.8 + [0, canopy * 0.45, 0]
            capsule(V, p0, p1, R0 * 0.45, R0 * 0.22, B['bloom_log']); ends.append(p1)
        ends.append(top)
        more = []
        for e in ends:                                                  # clumps round each branch end
            for j in range(3): more.append(e + [rr.normal(0, canopy * 0.28), rr.normal(0, canopy * 0.12), rr.normal(0, canopy * 0.28)])
        for e in ends + more:                                           # the canopy: flattened, ragged clumps of leaves
            rx = canopy * rr.uniform(0.28, 0.42); ry = canopy * rr.uniform(0.16, 0.22)
            Gc = V.grid(e - [rx + 1, ry + 1, rx + 1], e + [rx + 1, ry + 1, rx + 1]); X, Y, Z = Gc
            q = ((X + 0.5 - e[0]) / rx) ** 2 + ((Y + 0.5 - e[1]) / ry) ** 2 + ((Z + 0.5 - e[2]) / rx) ** 2
            n = fbm(X, Y, Z, 3.0, seed + 7)
            V.put(Gc, q < 1.0 - 0.9 * (n - 0.3), B['bloom_leaves'], only_air=True)
        # vines from the canopy's underside
        lv = (V.v == B['bloom_leaves']); xs, zs = np.nonzero(lv.any(1))
        for i in rr.choice(len(xs), min(len(xs), int(canopy * 2.5)), replace=False):
            col = np.nonzero(lv[xs[i], :, zs[i]])[0]; yb = col.min()
            if V.v[xs[i], yb - 1, zs[i]] != 0: continue
            n = rr.integers(3, 11)
            for j in range(1, n + 1):
                if V.v[xs[i], yb - j, zs[i]] != 0: break
                V.v[xs[i], yb - j, zs[i]] = B['glow_vine']
        S['glows'].append([float(top[0]), float(top[1] - 2), float(top[2]), canopy * 1.1, 1.0, 0.5, 1.0])
        return top
    top = tree(gc[0] + 4, gc[2] + 2, 38, 3.6, 17, 71)
    S['grove'] = dict(c=gc.tolist(), tree=top.tolist())
    for k, (dx, dz) in enumerate(((-24, 8), (20, -20), (-12, -26), (24, 18))):
        tree(gc[0] + dx, gc[2] + dz, rng.uniform(13, 19), 1.4, rng.uniform(6.5, 8.5), 90 + k)
    # the pool and its light fall
    a = np.radians(200); o = np.array([np.cos(a), 0, np.sin(a)])
    pc = gc + o * 24; py = gy(pc[0], pc[2])
    Gp = V.grid([pc[0] - 8, py - 3, pc[2] - 8], [pc[0] + 8, py + 4, pc[2] + 8]); X, Y, Z = Gp
    dd = np.hypot(X + 0.5 - pc[0], Z + 0.5 - pc[2])
    V.put(Gp, (dd < 6.5) & (Y >= py - 1), 0); V.put(Gp, (dd < 7.5) & (dd >= 6.5) & (Y == py - 1), B['end_stone_smooth'])
    V.put(Gp, (dd < 6.5) & (Y == py - 2), B['void_glass'])
    redge = max(rr for rr in np.arange(10, 60, 0.5) if (int(np.floor(gc[0] + o[0] * rr)), int(np.floor(gc[2] + o[2] * rr))) in tops)
    for t in np.arange(30, redge + 0.5, 0.5):
        for k in range(-2, 3):
            p = gc + o * t + np.array([-o[2], 0, o[0]]) * k
            for yy in range(py - 1, py + 4): V.set(int(np.floor(p[0])), yy, int(np.floor(p[2])), 0)
            V.set(int(np.floor(p[0])), py - 2, int(np.floor(p[2])), B['void_glass'])
    lip = gc + o * (redge + 0.6)
    S['pools'] = [[float(pc[0]), float(py - 1.35), float(pc[2]), 6.4]]
    S['channels'] = [[float(gc[0] + o[0] * 30), float(gc[2] + o[2] * 30), float(lip[0]), float(lip[2]), float(py - 1.35), 5.0]]
    S['lightfalls'].append([float(lip[0]), float(py - 1.4), float(lip[2]), float(o[0]), float(o[2]), 8.0])
    mesh(V, 'grove', meta)


# ============================================================================== the far End (coarse: one voxel = 4 blocks)
def far_end(meta, S, rng):
    SC = 4; n = 0
    def coarse(lo, hi): return Vol(np.floor(np.array(lo) / SC).astype(int), np.ceil(np.array(hi) / SC).astype(int))
    placed = [(0, 0, 90), (-175, 30, 60), (175, 40, 60), (30, -175, 60)]
    cnt = 0
    while cnt < 26:
        a = rng.uniform(0, 2 * np.pi); rr = rng.uniform(330, 1500); x, z = rr * np.cos(a), rr * np.sin(a)
        R = rng.uniform(24, 70) * (0.6 + rr / 1500); y = rng.uniform(-160, 140)
        if any(np.hypot(x - p[0], z - p[1]) < p[2] + R + 40 for p in placed): continue
        placed.append((x, z, R)); cnt += 1
        V = coarse([x - R - 40, y - R * 1.6 - 40, z - R - 40], [x + R + 40, y + R * 1.2 + 40, z + R + 40])
        tops = island(V, x / SC, y / SC, z / SC, R / SC, R * 1.3 / SC, 300 + cnt, top_amp=1.0, moss=0.55, deco=False)
        kind = rng.random(); cx, cz = int(x / SC), int(z / SC); ty = tops.get((cx, cz), int(y / SC))
        if kind < 0.35:                                     # a broken tower
            h = int(rng.uniform(6, 16)); box(V, cx - 1, ty, cz - 1, cx + 1, ty + h, cz + 1, B['purpur'])
            for yy in range(ty + 3, ty + h, 4): box(V, cx - 2, yy, cz - 2, cx + 2, yy, cz + 2, B['purpur_tile'])
            S['farlights'].append([x, (ty + h + 1) * SC, z, 1.0, 0.5, 1.0])
        elif kind < 0.6:                                    # a crystal
            hex_crystal(V, [cx + 0.5, ty, cz + 0.5], unit([rng.normal(0, .3), 1, rng.normal(0, .3)]), rng.uniform(5, 14), rng.uniform(1.2, 2.2),
                        B['aether_crystal' if rng.random() < 0.5 else 'void_crystal'])
            S['farlights'].append([x, (ty + 6) * SC, z, 0.4, 1.0, 1.0])
        elif kind < 0.8:                                    # pillars
            for k in range(3):
                px, pz = cx + rng.integers(-3, 4), cz + rng.integers(-3, 4); h = int(rng.uniform(4, 12))
                box(V, px, ty, pz, px + 1, ty + h, pz + 1, B['obsidian'])
        mesh(V, 'far%02d' % n, meta, scale=SC); n += 1
    # colossal pillars rising out of the void sea
    for k in range(7):
        a = np.radians(k * 360 / 7 + rng.uniform(-15, 15)); rr = rng.uniform(800, 1400); x, z = rr * np.cos(a), rr * np.sin(a)
        H = rng.uniform(180, 420); r = rng.uniform(22, 40)
        V = coarse([x - r - 8, -480, z - r - 8], [x + r + 8, H + 20, z + r + 8])
        Gp = V.grid(V.lo, V.hi - 1); X, Y, Z = Gp
        dd = np.hypot((X + 0.5) * SC - x, (Z + 0.5) * SC - z); rad = r * (1 - 0.15 * ((Y * SC + 480) / (H + 480)))
        col = (dd <= rad) & (Y * SC <= H - 8 * fbm(X, 0, Z, 2, k))
        V.put(Gp, col, B['obsidian']); V.put(Gp, col & (np.mod(Y, 9) == 0), B['cracked_obsidian'])
        S['farlights'].append([x, H + 30, z, 1.0, 0.4, 1.0]); S['colossi'].append([x, H, z, r])
        mesh(V, 'colossus%d' % k, meta, scale=SC)


def main():
    meta = dict(groups={}); rng = np.random.default_rng(42)
    S = dict(crystals=[], lamps=[], glows=[], lightfalls=[], farlights=[], colossi=[], pools=[], channels=[], bh=dict(yaw=180.0, el=19.0))
    main_island(meta, S, rng); end_city(meta, S, rng); crystal_spires(meta, S, rng); grove(meta, S, rng); far_end(meta, S, rng)
    json.dump(meta, open(os.path.join(HERE, 'web', 'meta.json'), 'w'))
    json.dump(S, open(os.path.join(HERE, 'web', 'scene.json'), 'w'), indent=1)
    print('scene', {k: (len(v) if isinstance(v, list) else '..') for k, v in S.items()})
    tot = sum(g['tris'] for g in meta['groups'].values()); print('total tris', tot)


if __name__ == '__main__':
    main()
