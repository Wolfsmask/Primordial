"""Original 16x16 block textures for the End Update (procedural pixel art), packed into four atlases:
  atlas.png     albedo (rgba)          atlas_em.png  emission (rgb)
  atlas_n.png   tangent-space normals  atlas_s.png   (unused, roughness, metalness) - three.js reads G and B
python3 end2_tex.py"""
import numpy as np
from PIL import Image

T = 16
COLS = 8
TILES = ['end_stone', 'end_bricks', 'obsidian', 'purpur', 'purpur_pillar', 'purpur_top', 'bedrock', 'chorus_stem',
         'chorus_flower', 'void_moss_top', 'void_moss_side', 'bloom_leaves', 'bloom_log', 'bloom_log_top', 'void_crystal', 'end_rod',
         'cracked_obsidian', 'gold_trim', 'end_grass', 'shardstone',
         'lit_window', 'banner', 'aether_crystal', 'glow_vine', 'glow_shroom', 'rune_obsidian', 'purpur_tile', 'end_stone_smooth',
         'chiseled_purpur', 'shardstone_top', 'gold_block', 'void_glass']
IDX = {n: i for i, n in enumerate(TILES)}
# roughness, metalness, normal strength
MAT = dict(end_stone=(0.86, 0, 1.6), end_bricks=(0.72, 0, 2.2), obsidian=(0.16, 0, 0.9), purpur=(0.5, 0, 1.8), purpur_pillar=(0.45, 0, 2.0),
           purpur_top=(0.45, 0, 1.6), bedrock=(0.9, 0, 2.5), chorus_stem=(0.6, 0, 1.5), chorus_flower=(0.5, 0, 1.0), void_moss_top=(0.9, 0, 1.4),
           void_moss_side=(0.85, 0, 1.4), bloom_leaves=(0.55, 0, 1.2), bloom_log=(0.8, 0, 2.2), bloom_log_top=(0.8, 0, 1.4), void_crystal=(0.06, 0, 0.8),
           end_rod=(0.2, 0, 0.2), cracked_obsidian=(0.2, 0, 1.4), gold_trim=(0.28, 1, 1.6), end_grass=(0.6, 0, 0.0), shardstone=(0.42, 0, 1.6),
           lit_window=(0.08, 0, 1.0), banner=(0.85, 0, 1.0), aether_crystal=(0.06, 0, 0.8), glow_vine=(0.6, 0, 0.0), glow_shroom=(0.5, 0, 0.0),
           rune_obsidian=(0.12, 0, 1.2), purpur_tile=(0.32, 0, 1.4), end_stone_smooth=(0.55, 0, 1.2), chiseled_purpur=(0.4, 0, 2.4),
           shardstone_top=(0.36, 0, 1.4), gold_block=(0.22, 1, 1.2), void_glass=(0.04, 0, 0.3))


def shades(base, n=4, spread=0.22):
    base = np.array(base, float)
    return [np.clip(base * (1 - spread + 2 * spread * k / (n - 1)), 0, 255) for k in range(n)]


def speckle(base, spread=0.22, n=4, seed=0, blotch=0.0):
    r = np.random.default_rng(seed); sh = shades(base, n, spread); v = r.random((T, T))
    if blotch:
        yy, xx = np.mgrid[0:T, 0:T]
        for _ in range(int(6 * blotch)):
            cx, cy, rr = r.integers(0, T), r.integers(0, T), r.integers(1, 3)
            v = np.clip(v + ((((xx - cx) % T) ** 2 + ((yy - cy) % T) ** 2) < rr * rr) * r.choice([-0.35, 0.35]), 0, 0.999)
    k = (v * n).astype(int); img = np.zeros((T, T, 4)); img[..., 3] = 255
    for i in range(n): img[k == i, :3] = sh[i]
    return img


def lum(img): return (img[..., 0] * 0.3 + img[..., 1] * 0.55 + img[..., 2] * 0.15) / 255.0
YY, XX = np.mgrid[0:T, 0:T]


def brick_mortar(rows=(0, 8), off=8, cols=None):
    """mask of mortar lines for a running-bond brick pattern"""
    m = np.zeros((T, T), bool)
    for y in rows: m[y, :] = True
    for y in range(T):
        band = 0 if y < 8 else 1
        for x in ((0,) if band == 0 else (off,)) if cols is None else cols[band]: m[y, x % T] = True
    return m


def tile(name):
    """-> (rgba, emission rgb 0..255, height 0..1)"""
    em = np.zeros((T, T, 3)); r = np.random.default_rng(sum(map(ord, name)) * 7 + 3); h = None
    if name == 'end_stone':
        img = speckle([221, 223, 165], 0.13, 4, seed=1, blotch=1.2)
        pits = r.random((T, T)) < 0.07; img[pits, :3] = [182, 178, 124]
        h = lum(img); h[pits] = 0.0
    elif name == 'end_stone_smooth':
        img = speckle([226, 226, 176], 0.05, 3, seed=31)
        edge = (XX == 0) | (YY == 0) | (XX == 15) | (YY == 15); img[edge, :3] *= 0.86
        h = np.where(edge, 0.3, 0.8) + lum(img) * 0.1
    elif name == 'end_bricks':
        img = speckle([226, 228, 172], 0.08, 3, seed=2)
        m = brick_mortar(rows=(0, 8), cols=((0,), (8,)))
        img[m, :3] *= 0.70; h = np.where(m, 0.0, 0.85) + 0.1 * lum(img)
    elif name in ('obsidian', 'cracked_obsidian', 'rune_obsidian'):
        img = speckle([20, 13, 32], 0.45, 5, seed=3, blotch=1.5)
        sp = r.random((T, T)) < 0.06; img[sp, :3] = [72, 42, 116]
        h = 0.5 + 0.5 * lum(img) * 3
        if name == 'cracked_obsidian':
            vein = (np.abs(np.sin(XX * 0.7 + YY * 0.35 + 1.0) * 6 - (YY - 8) * 0.6) < 0.8) | (np.abs(XX - 4 - (YY * 0.4)) < 0.6)
            img[vein, :3] = [200, 110, 255]; em[vein] = [210, 100, 255]; h[vein] = 0.0
        if name == 'rune_obsidian':                    # a carved glyph band, lit from inside
            g = np.zeros((T, T), bool)
            g[4:12, 7] = True; g[5, 5:10] = True; g[10, 6:9] = True; g[7:9, 4] = g[7:9, 10] = True; g[12, 5] = g[12, 9] = True
            img[g, :3] = [200, 140, 255]; em[g] = [150, 80, 230]; h[g] = 0.0
    elif name == 'purpur':
        img = speckle([169, 125, 169], 0.10, 3, seed=4)
        m = np.zeros((T, T), bool); m[0, :] = m[8, :] = True
        for y in range(T): m[y, 0 if y < 8 else 8] = True
        img[m, :3] *= 0.78; h = np.where(m, 0.1, 0.85)
    elif name == 'purpur_tile':
        img = speckle([176, 132, 178], 0.05, 3, seed=41)
        m = (XX == 0) | (YY == 0); img[m, :3] *= 0.8
        img[(XX == 1) | (YY == 1), :3] *= 1.08
        h = np.where(m, 0.1, 0.9)
    elif name == 'chiseled_purpur':
        img = speckle([171, 128, 171], 0.08, 3, seed=42)
        ring = np.maximum(np.abs(XX - 7.5), np.abs(YY - 7.5))
        m = (ring.astype(int) == 7) | (ring.astype(int) == 4); img[m, :3] *= 0.72
        c = ring < 2.5; img[c, :3] = [220, 175, 225]
        h = np.where(m, 0.05, 0.8); h[c] = 1.0
    elif name == 'purpur_pillar':
        img = speckle([171, 128, 171], 0.08, 3, seed=5)
        for x in (0, 3, 12, 15): img[:, x, :3] *= 0.80
        h = np.ones((T, T)) * 0.85; h[:, [0, 3, 12, 15]] = 0.2
    elif name == 'purpur_top':
        img = speckle([171, 128, 171], 0.08, 3, seed=6)
        ring = (np.maximum(np.abs(XX - 7.5), np.abs(YY - 7.5)).astype(int) % 4 == 0); img[ring, :3] *= 0.80
        h = np.where(ring, 0.2, 0.85)
    elif name == 'bedrock':
        img = speckle([84, 84, 84], 0.55, 5, seed=7, blotch=2); h = lum(img) * 2
    elif name == 'chorus_stem':
        img = speckle([96, 64, 110], 0.18, 4, seed=8)
        for y in range(0, T, 4): img[y, :, :3] *= 0.75
        h = lum(img) * 2
    elif name == 'chorus_flower':
        img = speckle([150, 110, 170], 0.15, 4, seed=9)
        c = ((XX - 7.5) ** 2 + (YY - 7.5) ** 2) < 14
        img[c, :3] = [235, 200, 255]; em[c] = [220, 170, 255]; h = np.where(c, 1.0, 0.5)
    elif name == 'void_moss_top':
        img = speckle([46, 28, 74], 0.30, 4, seed=10, blotch=1)
        sp = r.random((T, T)) < 0.08; img[sp, :3] = [170, 120, 255]; em[sp] = [140, 80, 255]; h = lum(img) * 3
    elif name == 'void_moss_side':
        img = speckle([221, 223, 165], 0.13, 4, seed=11, blotch=1.2)
        m = speckle([46, 28, 74], 0.3, 4, seed=12); fr = 4 + (r.random(T) < 0.5) + (r.random(T) < 0.3)
        for x in range(T): img[:fr[x], x] = m[:fr[x], x]
        sp = (r.random((T, T)) < 0.06); sp[6:] = False; img[sp, :3] = [170, 120, 255]; em[sp] = [140, 80, 255]
        h = lum(img)
    elif name == 'bloom_leaves':
        img = speckle([42, 26, 84], 0.16, 4, seed=13)
        holes = r.random((T, T)) < 0.10; img[holes, 3] = 0
        bl = np.zeros((T, T), bool)
        for cx_, cy_ in ((3, 4), (11, 9), (6, 12)): bl[cy_:cy_ + 2, cx_:cx_ + 2] = True
        bl &= ~holes; img[bl, :3] = [58, 34, 96]; em[bl] = [255, 110, 235]
        em[~bl & ~holes] = img[~bl & ~holes, :3] * 0.03; h = lum(img) * 2
    elif name == 'bloom_log':
        img = speckle([40, 30, 60], 0.3, 4, seed=14)
        for x in (2, 7, 12): img[:, x, :3] *= 0.6
        g = r.random((T, T)) < 0.008; img[g, :3] = [120, 90, 200]; em[g] = [90, 60, 200]
        h = np.ones((T, T)) * 0.8; h[:, [2, 7, 12]] = 0.1
    elif name == 'bloom_log_top':
        img = speckle([60, 45, 85], 0.2, 3, seed=15)
        rr = np.hypot(XX - 7.5, YY - 7.5); ring = (rr.astype(int) % 3 == 0); img[ring, :3] *= 0.8; h = np.where(ring, 0.3, 0.7)
    elif name in ('void_crystal', 'aether_crystal'):
        base = np.array([150, 82, 240] if name == 'void_crystal' else [62, 196, 214], float)
        g = 0.92 + 0.08 * np.cos((XX - 7.5) * 0.3) * np.cos((YY - 7.5) * 0.22)
        img = np.zeros((T, T, 4)); img[..., 3] = 255; img[..., :3] = base * g[..., None]
        em[:] = img[..., :3] * 0.55; h = np.full((T, T), 0.7)
    elif name == 'end_rod':
        img = np.zeros((T, T, 4)); img[..., 3] = 255; img[..., :3] = [250, 245, 255]; em[:] = [255, 245, 255]; h = np.ones((T, T)) * 0.5
    elif name == 'gold_trim':
        img = speckle([226, 180, 64], 0.16, 4, seed=17)
        m = (YY == 0) | (YY == 15) | (YY == 7); img[m, :3] *= 0.75; img[YY == 1, :3] = np.minimum(img[YY == 1, :3] * 1.2, 255)
        h = np.where(m, 0.2, 0.85)
    elif name == 'gold_block':
        img = speckle([236, 192, 78], 0.10, 4, seed=44)
        m = (XX == 0) | (YY == 0); img[m, :3] *= 0.78; h = np.where(m, 0.2, 0.9)
    elif name == 'end_grass':
        img = np.zeros((T, T, 4))
        for k in range(7):
            x = r.integers(1, 15); hh = r.integers(5, 14)
            img[T - hh:, x, :3] = [140, 96, 230]; img[T - hh:, x, 3] = 255
            img[T - hh, x, :3] = [240, 200, 255]; em[T - hh, x] = [230, 170, 255]
        h = np.zeros((T, T))
    elif name == 'glow_vine':                          # hanging vines with glowing berries
        img = np.zeros((T, T, 4))
        for x in (3, 7, 12):
            xs = (x + (np.sin(np.arange(T) * 0.8 + x) * 1.2).astype(int)) % T
            for y in range(T): img[y, xs[y], :3] = [70, 40, 120]; img[y, xs[y], 3] = 255
            for y in r.choice(T, 1, replace=False): img[y, xs[y], :3] = [150, 255, 240]; em[y, xs[y]] = [90, 255, 230]
        h = np.zeros((T, T))
    elif name == 'glow_shroom':
        img = np.zeros((T, T, 4))
        img[8:, 7:9, :3] = [200, 190, 220]; img[8:, 7:9, 3] = 255
        cap = ((XX - 7.5) ** 2 / 36 + (YY - 7) ** 2 / 9 < 1) & (YY <= 8)
        img[cap, :3] = [90, 230, 220]; img[cap, 3] = 255; em[cap] = [60, 230, 220]
        dots = cap & (r.random((T, T)) < 0.25); img[dots, :3] = [230, 255, 255]; em[dots] = [200, 255, 255]
        h = np.zeros((T, T))
    elif name == 'shardstone':
        img = speckle([52, 46, 70], 0.3, 4, seed=18, blotch=1)
        f = r.random((T, T)) < 0.05; img[f, :3] = [150, 100, 225]; em[f] = [70, 40, 120]; h = lum(img) * 3
    elif name == 'shardstone_top':
        img = speckle([58, 52, 78], 0.22, 4, seed=45)
        cr = (np.abs((XX * 0.9 + YY * 0.4) % 7 - 3.5) < 0.5); img[cr, :3] = [120, 230, 235]; em[cr] = [40, 160, 170]
        h = np.where(cr, 0.0, 0.7) + 0.1 * lum(img)
    elif name == 'lit_window':                         # purple panes, warm light behind them
        img = speckle([150, 90, 200], 0.08, 3, seed=46)
        fr = (XX == 0) | (XX == 15) | (YY == 0) | (YY == 15) | (XX == 7) | (XX == 8) | (YY == 7) | (YY == 8)
        glow = np.exp(-((XX - 7.5) ** 2 + (YY - 7.5) ** 2) / 60.0)
        img[..., 0] = 200 + 55 * glow; img[..., 1] = 140 + 80 * glow; img[..., 2] = 255
        em[..., 0] = 255 * (0.7 + 0.3 * glow); em[..., 1] = 160 + 70 * glow; em[..., 2] = 230
        img[fr, :3] = [70, 45, 90]; em[fr] = 0
        h = np.where(fr, 1.0, 0.3)
    elif name == 'banner':                             # magenta cloth, a gold eye woven in
        img = speckle([150, 40, 120], 0.10, 3, seed=47)
        eye = ((XX - 7.5) ** 2 / 20 + (YY - 7.5) ** 2 / 6 < 1); pupil = ((XX - 7.5) ** 2 + (YY - 7.5) ** 2) < 3
        img[eye, :3] = [235, 190, 80]; img[pupil, :3] = [30, 10, 40]
        img[(YY == 0) | (YY == 15), :3] = [230, 180, 70]
        h = np.where(eye, 0.8, 0.5) + 0.2 * lum(img)
    elif name == 'void_glass':
        img = np.zeros((T, T, 4)); img[..., :3] = [40, 20, 70]; img[..., 3] = 255
        img[(XX == 0) | (YY == 0) | (XX == 15) | (YY == 15), :3] = [90, 60, 130]; h = np.ones((T, T)) * 0.5
    else:
        img = speckle([255, 0, 255], 0.1, 2); h = np.zeros((T, T))
    if h is None: h = lum(img)
    return img, em, np.clip(h, 0, 1.5)


def normals(h, s):
    """tangent-space normal from a tiling height map (x right, y down the texture -> green = up)"""
    dx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5; dy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    n = np.stack([-dx * s, dy * s, np.ones_like(h)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return n * 0.5 + 0.5


def build(out='web'):
    rows = (len(TILES) + COLS - 1) // COLS
    A = np.zeros((rows * T, COLS * T, 4), np.uint8); E = np.zeros((rows * T, COLS * T, 3), np.uint8)
    N = np.zeros((rows * T, COLS * T, 3), np.uint8); S = np.zeros((rows * T, COLS * T, 3), np.uint8)
    N[...] = [128, 128, 255]; S[...] = [255, 230, 0]
    for i, n in enumerate(TILES):
        img, em, h = tile(n); r_, c_ = divmod(i, COLS); sl = (slice(r_ * T, (r_ + 1) * T), slice(c_ * T, (c_ + 1) * T))
        A[sl] = np.clip(img, 0, 255).astype(np.uint8); E[sl] = np.clip(em, 0, 255).astype(np.uint8)
        ro, me, ns = MAT[n]
        N[sl] = np.clip(normals(h, ns) * 255, 0, 255).astype(np.uint8)
        rough = np.clip(ro + (h - h.mean()) * -0.15 + np.random.default_rng(i).normal(0, 0.03, (T, T)), 0.03, 1.0)
        S[sl] = np.stack([np.full((T, T), 255), rough * 255, np.full((T, T), me * 255)], -1).astype(np.uint8)
    Image.fromarray(A, 'RGBA').save(f'{out}/atlas.png'); Image.fromarray(E, 'RGB').save(f'{out}/atlas_em.png')
    Image.fromarray(N, 'RGB').save(f'{out}/atlas_n.png'); Image.fromarray(S, 'RGB').save(f'{out}/atlas_s.png')
    return A, N


if __name__ == '__main__':
    import os
    os.makedirs('web', exist_ok=True)
    A, N = build()
    big = np.concatenate([A[..., :3], N], 0)
    Image.fromarray(big).resize((big.shape[1] * 6, big.shape[0] * 6), Image.NEAREST).save('atlas_big.png')
    print('atlas', A.shape, len(TILES), 'tiles')
