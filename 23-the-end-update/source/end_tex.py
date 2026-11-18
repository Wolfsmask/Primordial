"""Original 16x16 block textures for the End (procedural pixel art), packed into an atlas, plus an emission atlas.
python3 end_tex.py -> web/atlas.png, web/atlas_em.png"""
import numpy as np
from PIL import Image

T = 16
COLS = 8
TILES = ['end_stone', 'end_bricks', 'obsidian', 'purpur', 'purpur_pillar', 'purpur_top', 'bedrock', 'chorus_stem',
         'chorus_flower', 'void_moss_top', 'void_moss_side', 'bloom_leaves', 'bloom_log', 'bloom_log_top', 'void_crystal', 'end_rod',
         'cracked_obsidian', 'gold_trim', 'end_grass', 'shardstone']
IDX = {n: i for i, n in enumerate(TILES)}
EMISSIVE = {'chorus_flower', 'bloom_leaves', 'void_crystal', 'end_rod', 'cracked_obsidian', 'end_grass', 'void_moss_top', 'void_moss_side'}


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


def tile(name):
    """-> (rgba, emission rgb 0..255)"""
    em = np.zeros((T, T, 3))
    r = np.random.default_rng(abs(hash(name)) % 9973)
    if name == 'end_stone':                      # pale yellow-green stone, pitted
        img = speckle([221, 223, 165], 0.13, 4, seed=1, blotch=1.2)
        pits = r.random((T, T)) < 0.07; img[pits, :3] = [190, 186, 130]
    elif name == 'end_bricks':
        img = speckle([226, 228, 172], 0.08, 3, seed=2)
        for y in (0, 8): img[y, :, :3] *= 0.80
        for x in range(T):
            for y in range(T):
                if (x == 0 and y < 8) or (x == 8 and y >= 8): img[y, x, :3] *= 0.80
    elif name in ('obsidian', 'cracked_obsidian'):
        img = speckle([22, 14, 34], 0.45, 5, seed=3, blotch=1.5)
        sp = r.random((T, T)) < 0.06; img[sp, :3] = [70, 40, 110]
        if name == 'cracked_obsidian':                # glowing violet veins (new)
            yy, xx = np.mgrid[0:T, 0:T]
            vein = (np.abs(np.sin(xx * 0.7 + yy * 0.35 + 1.0) * 6 - (yy - 8) * 0.6) < 0.8) | (np.abs(xx - 4 - (yy * 0.4)) < 0.6)
            img[vein, :3] = [190, 90, 255]; em[vein] = [200, 90, 255]
    elif name == 'purpur':
        img = speckle([169, 125, 169], 0.10, 3, seed=4)
        for y in (0, 8):
            img[y, :, :3] *= 0.82
        for y in range(T):
            img[y, 0 if y < 8 else 8, :3] *= 0.82
    elif name == 'purpur_pillar':
        img = speckle([171, 128, 171], 0.08, 3, seed=5)
        for x in (0, 3, 12, 15): img[:, x, :3] *= 0.84
    elif name == 'purpur_top':
        img = speckle([171, 128, 171], 0.08, 3, seed=6)
        yy, xx = np.mgrid[0:T, 0:T]; ring = (np.maximum(np.abs(xx - 7.5), np.abs(yy - 7.5)).astype(int) % 4 == 0); img[ring, :3] *= 0.84
    elif name == 'bedrock':
        img = speckle([84, 84, 84], 0.55, 5, seed=7, blotch=2)
    elif name == 'chorus_stem':                   # purple, ribbed
        img = speckle([96, 64, 110], 0.18, 4, seed=8)
        for y in range(0, T, 4): img[y, :, :3] *= 0.75
    elif name == 'chorus_flower':
        img = speckle([150, 110, 170], 0.15, 4, seed=9)
        yy, xx = np.mgrid[0:T, 0:T]; c = ((xx - 7.5) ** 2 + (yy - 7.5) ** 2) < 14
        img[c, :3] = [235, 200, 255]; em[c] = [220, 170, 255]
    elif name == 'void_moss_top':                 # new: dark violet moss, glowing specks
        img = speckle([52, 30, 78], 0.30, 4, seed=10, blotch=1)
        sp = r.random((T, T)) < 0.08; img[sp, :3] = [180, 120, 255]; em[sp] = [150, 80, 255]
    elif name == 'void_moss_side':
        img = speckle([221, 223, 165], 0.13, 4, seed=11, blotch=1.2)
        m = speckle([52, 30, 78], 0.3, 4, seed=12); fr = 4 + (r.random(T) < 0.5) + (r.random(T) < 0.3)
        for x in range(T): img[:fr[x], x] = m[:fr[x], x]
        sp = (r.random((T, T)) < 0.06); sp[6:] = False; img[sp, :3] = [180, 120, 255]; em[sp] = [150, 80, 255]
    elif name == 'bloom_leaves':                  # new: the glowing canopy of the ender groves
        img = speckle([110, 60, 170], 0.30, 5, seed=13)
        holes = r.random((T, T)) < 0.12; img[holes, 3] = 0
        bl = r.random((T, T)) < 0.16; img[bl, :3] = [255, 190, 255]; em[bl] = [255, 150, 255]
        em[~bl & ~holes] = img[~bl & ~holes, :3] * 0.10
    elif name == 'bloom_log':
        img = speckle([40, 30, 60], 0.3, 4, seed=14)
        for x in (2, 7, 12): img[:, x, :3] *= 0.7
        g = r.random((T, T)) < 0.04; img[g, :3] = [160, 110, 255]; em[g] = [140, 90, 255]
    elif name == 'bloom_log_top':
        img = speckle([60, 45, 85], 0.2, 3, seed=15)
    elif name == 'void_crystal':                  # new: magenta-violet crystal, bright core
        img = speckle([190, 110, 255], 0.25, 4, seed=16)
        yy, xx = np.mgrid[0:T, 0:T]; facet = ((xx + yy) % 6 == 0) | ((xx - yy) % 7 == 0); img[facet, :3] = [250, 220, 255]
        em[:] = img[..., :3] * 0.7
    elif name == 'end_rod':
        img = np.zeros((T, T, 4)); img[..., 3] = 255; img[..., :3] = [250, 245, 255]; em[:] = [255, 245, 255]
    elif name == 'gold_trim':
        img = speckle([230, 190, 70], 0.18, 4, seed=17)
    elif name == 'end_grass':                     # new: short glowing tufts (crossed plant)
        img = np.zeros((T, T, 4))
        for k in range(7):
            x = r.integers(1, 15); h = r.integers(5, 14)
            img[T - h:, x, :3] = [150, 100, 230]; img[T - h:, x, 3] = 255
            img[T - h, x, :3] = [240, 200, 255]; em[T - h, x] = [220, 160, 255]
    elif name == 'shardstone':                    # new: dark slate with violet flecks (the crystal spires' bed)
        img = speckle([58, 52, 74], 0.3, 4, seed=18, blotch=1)
        f = r.random((T, T)) < 0.05; img[f, :3] = [160, 110, 230]
    else:
        img = speckle([255, 0, 255], 0.1, 2)
    return img, em


def build(path='web/atlas.png', em_path='web/atlas_em.png'):
    rows = (len(TILES) + COLS - 1) // COLS
    A = np.zeros((rows * T, COLS * T, 4), np.uint8); E = np.zeros((rows * T, COLS * T, 3), np.uint8)
    for i, n in enumerate(TILES):
        img, em = tile(n); r_, c_ = divmod(i, COLS)
        A[r_ * T:(r_ + 1) * T, c_ * T:(c_ + 1) * T] = np.clip(img, 0, 255).astype(np.uint8)
        E[r_ * T:(r_ + 1) * T, c_ * T:(c_ + 1) * T] = np.clip(em, 0, 255).astype(np.uint8)
    Image.fromarray(A, 'RGBA').save(path); Image.fromarray(E, 'RGB').save(em_path)
    return A


if __name__ == '__main__':
    import os
    os.makedirs('web', exist_ok=True)
    A = build()
    Image.fromarray(A, 'RGBA').resize((A.shape[1] * 8, A.shape[0] * 8), Image.NEAREST).save('atlas_big.png')
    print('atlas', A.shape)
