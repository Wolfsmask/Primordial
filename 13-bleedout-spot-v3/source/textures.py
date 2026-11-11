"""Original 16x16 pixel-art block textures in a blocky-game style, generated procedurally, packed into one atlas.
Each tile is built from a few quantised shades with per-pixel noise, the way hand-made 16px textures are."""
import numpy as np
from PIL import Image

T = 16
COLS = 8
TILES = ['grass_top', 'grass_side', 'dirt', 'stone', 'andesite', 'granite', 'tuff', 'gravel',
         'sand', 'snow', 'snow_side', 'water', 'water_flow', 'oak_log', 'log_top', 'oak_leaves',
         'birch_log', 'birch_leaves', 'spruce_log', 'spruce_leaves', 'cherry_log', 'cherry_leaves', 'coarse_dirt', 'moss',
         'short_grass', 'fern', 'poppy', 'dandelion', 'cornflower', 'allium', 'oxeye', 'pink_petals',
         'calcite', 'podzol_top', 'mossy_stone', 'podzol_side', 'clay', 'stone_snow_side', 'dark_oak_leaves', 'cherry_top']
IDX = {n: i for i, n in enumerate(TILES)}
rng = np.random.default_rng(7)


def shades(base, n=4, spread=0.22):
    base = np.array(base, float)
    return [np.clip(base * (1 - spread + 2 * spread * k / (n - 1)), 0, 255) for k in range(n)]


def speckle(base, spread=0.22, n=4, bias=None, blotch=0.0, seed=None):
    r = np.random.default_rng(seed)
    sh = shades(base, n, spread)
    v = r.random((T, T))
    if blotch > 0:                                       # a few larger darker/lighter clumps
        b = np.zeros((T, T))
        for _ in range(int(6 * blotch)):
            cx, cy, rr = r.integers(0, T), r.integers(0, T), r.integers(1, 3)
            yy, xx = np.mgrid[0:T, 0:T]
            b += (((xx - cx) % T) ** 2 + ((yy - cy) % T) ** 2 < rr * rr) * r.choice([-0.35, 0.35])
        v = np.clip(v + b, 0, 0.999)
    if bias is not None:
        v = np.clip(v + bias, 0, 0.999)
    k = (v * n).astype(int)
    img = np.zeros((T, T, 4))
    for i in range(n):
        img[k == i, :3] = sh[i]
    img[..., 3] = 255
    return img


def tile(name):
    if name == 'grass_top':
        return speckle([108, 168, 72], 0.20, 5, seed=1)
    if name == 'dirt':
        return speckle([134, 96, 67], 0.25, 4, blotch=1, seed=2)
    if name == 'grass_side':
        img = tile('dirt')
        g = speckle([108, 168, 72], 0.2, 4, seed=3)
        fringe = 3 + (rng.random(T) < 0.5).astype(int) + (rng.random(T) < 0.25).astype(int)
        for x in range(T):
            img[:fringe[x], x] = g[:fringe[x], x]
        return img
    if name == 'coarse_dirt':
        img = speckle([120, 86, 60], 0.3, 4, blotch=2, seed=4)
        st = rng.random((T, T)) < 0.12
        img[st, :3] = [112, 110, 108]
        return img
    if name == 'podzol_top':
        return speckle([110, 76, 40], 0.3, 4, blotch=1.5, seed=5)
    if name == 'podzol_side':
        img = tile('dirt'); p = tile('podzol_top')
        img[:3] = p[:3]
        return img
    if name == 'stone':
        return speckle([127, 127, 127], 0.18, 4, blotch=2, seed=6)
    if name == 'andesite':
        return speckle([136, 136, 138], 0.22, 4, blotch=3, seed=7)
    if name == 'granite':
        img = speckle([150, 104, 86], 0.22, 4, blotch=2, seed=8)
        img[rng.random((T, T)) < 0.1, :3] = [190, 150, 130]
        return img
    if name == 'tuff':
        return speckle([108, 109, 102], 0.2, 4, blotch=2, seed=9)
    if name == 'calcite':
        return speckle([222, 224, 220], 0.08, 3, blotch=1, seed=10)
    if name == 'mossy_stone':
        img = tile('stone'); m = speckle([88, 118, 52], 0.2, 3, seed=11)
        mask = rng.random((T, T)) < 0.35
        img[mask] = m[mask]
        return img
    if name == 'moss':
        return speckle([92, 124, 52], 0.22, 4, seed=12)
    if name == 'gravel':
        img = speckle([128, 122, 120], 0.35, 5, blotch=3, seed=13)
        return img
    if name == 'sand':
        return speckle([219, 207, 160], 0.08, 4, seed=14)
    if name == 'clay':
        return speckle([160, 166, 178], 0.08, 3, seed=15)
    if name == 'snow':
        return speckle([246, 250, 252], 0.04, 3, seed=16)
    if name == 'snow_side':
        img = tile('dirt'); s = tile('snow')
        fringe = 2 + (rng.random(T) < 0.5).astype(int) + (rng.random(T) < 0.2).astype(int)
        for x in range(T):
            img[:fringe[x], x] = s[:fringe[x], x]
        return img
    if name == 'stone_snow_side':
        img = tile('stone'); s = tile('snow')
        fringe = 3 + (rng.random(T) < 0.5).astype(int) + (rng.random(T) < 0.3).astype(int) * 2
        for x in range(T):
            img[:fringe[x], x] = s[:fringe[x], x]
        return img
    if name == 'water':
        img = speckle([70, 128, 200], 0.04, 3, seed=17)
        img[..., 3] = 200
        return img
    if name == 'water_flow':
        img = np.zeros((T, T, 4))
        sh = shades([60, 120, 210], 4, 0.18)
        for x in range(T):
            ph = rng.integers(0, 4)
            for y in range(T):
                img[y, x, :3] = sh[(y // 2 + ph + (x % 3 == 0)) % 4]
        img[..., 3] = 215
        return img
    if name in ('oak_log', 'spruce_log', 'cherry_log', 'birch_log'):
        base = {'oak_log': [104, 82, 50], 'spruce_log': [60, 42, 26], 'cherry_log': [92, 58, 64], 'birch_log': [220, 218, 210]}[name]
        img = np.zeros((T, T, 4)); sh = shades(base, 4, 0.25)
        for x in range(T):
            col = rng.integers(1, 3)
            for y in range(T):
                img[y, x, :3] = sh[(col + (rng.random() < 0.25) * rng.choice([-1, 1])) % 4]
        if name == 'birch_log':
            for _ in range(7):                            # the black marks
                y, x, w = rng.integers(0, T), rng.integers(0, T - 3), rng.integers(2, 5)
                img[y, x:x + w, :3] = [40, 40, 38]
        img[..., 3] = 255
        return img
    if name == 'log_top':
        img = np.zeros((T, T, 4)); yy, xx = np.mgrid[0:T, 0:T]
        r = np.hypot(xx - 7.5, yy - 7.5)
        ring = (r.astype(int) // 2) % 2
        sh = shades([170, 136, 88], 3, 0.15)
        img[..., :3] = np.where(ring[..., None] == 0, sh[2], sh[1])
        img[r > 6.6, :3] = [100, 78, 48]
        img[..., 3] = 255
        return img
    if name in ('oak_leaves', 'birch_leaves', 'spruce_leaves', 'cherry_leaves', 'dark_oak_leaves'):
        base = {'oak_leaves': [70, 128, 44], 'birch_leaves': [116, 156, 66], 'spruce_leaves': [66, 112, 70],
                'cherry_leaves': [236, 160, 196], 'dark_oak_leaves': [52, 96, 32]}[name]
        img = speckle(base, 0.28, 4, seed=hash(name) % 1000)
        holes = rng.random((T, T)) < (0.16 if name != 'spruce_leaves' else 0.12)
        img[holes, 3] = 0
        return img
    if name == 'cherry_top':
        return speckle([238, 168, 202], 0.18, 4, seed=21)
    if name == 'pink_petals':
        img = np.zeros((T, T, 4))
        for _ in range(22):
            x, y = rng.integers(0, T), rng.integers(0, T)
            img[y, x] = [248, 180, 214, 255]
        return img
    # plants: cross sprites on transparent background
    img = np.zeros((T, T, 4))
    if name in ('short_grass', 'fern'):
        sh = shades([100, 160, 66] if name == 'short_grass' else [72, 128, 56], 3, 0.2)
        for _ in range(9 if name == 'short_grass' else 6):
            x = rng.integers(1, T - 1); h = rng.integers(6, 14)
            for y in range(T - h, T):
                xx = int(np.clip(x + (rng.random() < 0.2) * rng.choice([-1, 1]), 0, T - 1))
                img[y, xx, :3] = sh[rng.integers(0, 3)]; img[y, xx, 3] = 255
                if name == 'fern' and y % 3 == 0:
                    for dx in (-2, -1, 1, 2):
                        if 0 <= xx + dx < T:
                            img[y, xx + dx, :3] = sh[1]; img[y, xx + dx, 3] = 255
        return img
    petal = {'poppy': [214, 40, 36], 'dandelion': [246, 214, 40], 'cornflower': [70, 110, 220], 'allium': [186, 110, 220],
             'oxeye': [240, 240, 236]}[name]
    stem = [70, 130, 44]
    for y in range(7, T):
        img[y, 7, :3] = stem; img[y, 7, 3] = 255
    img[11, 8, :3] = stem; img[11, 8, 3] = 255; img[12, 6, :3] = stem; img[12, 6, 3] = 255
    for dx, dy in [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1), (0, -2), (-2, 0), (2, 0)]:
        if abs(dx) + abs(dy) <= 2:
            x, y = 7 + dx, 5 + dy
            img[y, x, :3] = petal if (dx, dy) != (0, 0) else ([250, 210, 60] if name in ('oxeye', 'poppy') else [60, 40, 30])
            img[y, x, 3] = 255
    return img


def build(path='web/atlas.png'):
    rows = (len(TILES) + COLS - 1) // COLS
    atlas = np.zeros((rows * T, COLS * T, 4), np.uint8)
    for i, n in enumerate(TILES):
        r, c = divmod(i, COLS)
        atlas[r * T:(r + 1) * T, c * T:(c + 1) * T] = np.clip(tile(n), 0, 255).astype(np.uint8)
    Image.fromarray(atlas, 'RGBA').save(path)
    return atlas.shape


if __name__ == '__main__':
    print('atlas', build())
