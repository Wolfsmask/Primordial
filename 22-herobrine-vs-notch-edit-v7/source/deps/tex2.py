"""v2 block textures, original 16 px pixel art drawn to feel like the blocky game's classic (beta-era) textures:
clustered noise in a few shades per material, grass and leaves tinted with warm meadow colours, chunky stones,
tan planks.  -> web/atlas2.png (16 x 4 tiles), web/wolf2.png, web/sun2.png, web/sign.png, web/painting.png"""
import os, zlib
import numpy as np
from PIL import Image
import pixfont

HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
T = 16
TILES = ['grass_top', 'grass_side', 'dirt', 'stone', 'cobblestone', 'mossy_cobblestone', 'oak_planks', 'oak_log',
         'log_top', 'oak_leaves', 'glass', 'sand', 'gravel', 'water', 'tall_grass', 'rose',
         'dandelion', 'vine', 'cobweb', 'crafting_top', 'crafting_side', 'crafting_front', 'furnace_front', 'furnace_side',
         'furnace_top', 'chest_front', 'chest_side', 'chest_top', 'bed_head_top', 'bed_foot_top', 'bed_side_head', 'bed_side_foot',
         'bed_end', 'door_top', 'door_bottom', 'torch', 'water_flow', 'coal_ore', 'iron_ore', 'bookshelf',
         'birch_log', 'birch_leaves', 'sugar_cane', 'gravel2', 'white_wool', 'clay', 'sapling', 'lava']
IDX = {n: i for i, n in enumerate(TILES)}
COLS, ROWS = 16, 4
GRASS_TINT = np.array([124, 189, 80.0])          # warm meadow green
LEAF_TINT = np.array([86, 150, 46.0])


def R(name):
    return np.random.default_rng(zlib.crc32(name.encode()))


def clustered(r, levels, smooth=0.55):
    """per-pixel noise, smoothed a little with its neighbours (wrapping), quantised into len(levels) shades"""
    v = r.random((T, T))
    s = (np.roll(v, 1, 0) + np.roll(v, -1, 0) + np.roll(v, 1, 1) + np.roll(v, -1, 1)) / 4
    v = v * (1 - smooth) + s * smooth
    q = np.argsort(np.argsort(v.ravel())).reshape(T, T) / (T * T)       # equalise, then cut
    k = np.minimum((q * len(levels)).astype(int), len(levels) - 1)
    return np.array(levels, float)[k]


def img_from(rgb, a=255):
    out = np.zeros((T, T, 4)); out[..., :3] = rgb; out[..., 3] = a
    return out


def tinted(r, lv, tint):
    l = clustered(r, lv)
    return img_from(l[..., None] * tint[None, None, :])


def palette(r, cols, smooth=0.55):
    idx = clustered(r, list(range(len(cols))), smooth).astype(int)
    return img_from(np.array(cols, float)[idx])


def stones(r, n, base, hi, lo, mortar, deep):
    """chunky stones: voronoi cells, lit from the top-left, dark joints"""
    pts = r.random((n, 2)) * T
    yy, xx = np.mgrid[0:T, 0:T] + 0.5
    best = np.full((2, T, T), 1e9); lab = np.zeros((T, T), int)
    for i, (px, py) in enumerate(pts):
        for ox in (-T, 0, T):
            for oy in (-T, 0, T):
                d = np.hypot(xx - px - ox, yy - py - oy)
                closer = d < best[0]
                best[1] = np.where(closer, best[0], np.minimum(best[1], d)); best[0] = np.where(closer, d, best[0])
                lab = np.where(closer, i, lab)
    edge = (best[1] - best[0]) < 1.1
    out = np.zeros((T, T, 3))
    shade = 0.88 + 0.24 * r.random(n)
    col = np.array(base, float) * shade[lab][..., None]
    # light from the top-left inside each stone
    up = np.roll(lab, 1, 0) != lab; left = np.roll(lab, 1, 1) != lab
    dn = np.roll(lab, -1, 0) != lab; rt = np.roll(lab, -1, 1) != lab
    col = np.where((up | left)[..., None] & ~edge[..., None], np.array(hi, float), col)
    col = np.where((dn | rt)[..., None] & ~edge[..., None], np.array(lo, float), col)
    col = np.where(edge[..., None], np.array(mortar, float), col)
    deep_m = edge & (r.random((T, T)) < 0.35)
    col = np.where(deep_m[..., None], np.array(deep, float), col)
    col *= (0.95 + 0.1 * r.random((T, T)))[..., None]
    return img_from(col)


def tile(name):
    r = R(name)
    if name == 'grass_top':
        return tinted(r, [0.70, 0.78, 0.86, 0.93, 1.0], GRASS_TINT)
    if name == 'dirt':
        img = palette(r, [(104, 74, 50), (121, 85, 58), (134, 96, 67), (150, 108, 74)])
        for _ in range(7):
            x, y = r.integers(0, T, 2)
            img[y, x, :3] = (156, 124, 96) if r.random() < 0.6 else (96, 70, 48)
            if r.random() < 0.5:
                img[y, (x + 1) % T, :3] = img[y, x, :3] * 0.9
        return img
    if name == 'clay':
        return palette(r, [(150, 154, 168), (160, 165, 178), (170, 174, 186)])
    if name == 'grass_side':
        img = tile('dirt'); g = tinted(R('gs'), [0.72, 0.82, 0.92, 1.0], GRASS_TINT)
        for x in range(T):
            d = 3 + (r.random() < 0.6) + (r.random() < 0.3) + (r.random() < 0.12)
            img[:d, x] = g[:d, x]
        return img
    if name == 'stone':
        img = palette(r, [(108, 108, 108), (118, 118, 118), (126, 126, 126), (136, 136, 136)], 0.7)
        for _ in range(5):           # a few short dark cracks
            x, y = r.integers(0, T, 2); dx = r.choice([-1, 1])
            for k in range(r.integers(2, 5)):
                img[(y + k) % T, (x + dx * (k // 2)) % T, :3] = (94, 94, 94)
        return img
    if name in ('coal_ore', 'iron_ore'):
        img = tile('stone')
        c1, c2 = ((40, 40, 40), (66, 66, 66)) if name == 'coal_ore' else ((216, 175, 147), (175, 138, 112))
        for cx, cy in ((3, 3), (10, 4), (5, 10), (12, 11), (8, 7)):
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (2, 1)):
                if r.random() < 0.75:
                    img[(cy + dy) % T, (cx + dx) % T, :3] = c1 if r.random() < 0.65 else c2
        return img
    if name == 'cobblestone':
        return stones(r, 9, (128, 128, 128), (162, 162, 162), (100, 100, 100), (82, 82, 82), (58, 58, 58))
    if name == 'mossy_cobblestone':
        img = tile('cobblestone')
        m = tinted(R('moss'), [0.55, 0.68, 0.8], np.array([96, 140, 60.0]))
        yy, xx = np.mgrid[0:T, 0:T]
        mask = (np.sin(xx * 0.6 + 2) + np.cos(yy * 0.5 + 1) + r.random((T, T)) * 1.4) > 1.25
        img[mask] = m[mask]
        return img
    if name == 'oak_planks':
        img = palette(r, [(150, 120, 72), (160, 130, 80), (170, 139, 86)], 0.8)
        for y in range(T):
            if y % 4 == 3:
                img[y, :, :3] = (106, 82, 48)
            elif r.random() < 0.5:
                x0 = r.integers(0, T); img[y, x0:x0 + r.integers(3, 8), :3] = (136, 108, 64)
        for k, y0 in enumerate((0, 4, 8, 12)):
            sx = (3 + 7 * k) % T
            img[y0:y0 + 3, sx, :3] = (118, 92, 54)
        return img
    if name == 'oak_log':
        img = palette(r, [(78, 61, 36), (92, 72, 43), (104, 83, 50), (118, 94, 58)], 0.3)
        for x in range(T):
            if x % 3 == 0 or r.random() < 0.2:
                img[:, x, :3] *= 0.82
        return img
    if name == 'birch_log':
        img = palette(r, [(206, 206, 200), (216, 215, 210), (226, 226, 220)])
        for _ in range(8):
            x, y = r.integers(0, T, 2); w = r.integers(2, 5)
            img[y, x:x + w, :3] = (52, 52, 46)
        return img
    if name == 'log_top':
        yy, xx = np.mgrid[0:T, 0:T] + 0.5
        rr = np.maximum(np.abs(xx - 8), np.abs(yy - 8)) + 0.5 * r.random((T, T))
        img = img_from(np.where(((rr.astype(int)) % 2 == 0)[..., None], np.array([182, 146, 92.0]), np.array([158, 124, 76.0])))
        img[rr > 6.8] = tile('oak_log')[rr > 6.8]
        return img
    if name in ('oak_leaves', 'birch_leaves'):
        img = tinted(r, [0.48, 0.62, 0.76, 0.9, 1.0], LEAF_TINT if name == 'oak_leaves' else np.array([128, 167, 85.0]))
        img[r.random((T, T)) < 0.2, 3] = 0
        return img
    if name == 'glass':
        img = np.zeros((T, T, 4))
        for (y, x) in [(0, i) for i in range(T)] + [(T - 1, i) for i in range(T)] + [(i, 0) for i in range(T)] + [(i, T - 1) for i in range(T)]:
            img[y, x] = (214, 238, 242, 255)
        for k in range(4):
            img[2 + k, 3 + k] = (242, 252, 252, 220)
        for k in range(3):
            img[3 + k, 2 + k] = (242, 252, 252, 160)
        img[11, 12] = img[12, 11] = (242, 252, 252, 200)
        return img
    if name == 'sand':
        return palette(r, [(206, 197, 146), (215, 207, 154), (222, 214, 162), (230, 222, 172)], 0.4)
    if name in ('gravel', 'gravel2'):
        img = palette(r, [(98, 92, 90), (122, 116, 114), (140, 134, 130), (160, 152, 146), (128, 108, 96)], 0.75)
        return img
    if name == 'water':
        img = palette(r, [(38, 62, 200), (44, 72, 214), (52, 84, 228), (66, 100, 240)], 0.85)
        img[..., 3] = 168
        return img
    if name == 'water_flow':
        img = np.zeros((T, T, 4))
        for x in range(T):
            col = np.array([44, 72, 214.0]) * (0.85 + 0.3 * r.random())
            for y in range(T):
                img[y, x, :3] = col * (0.9 + 0.2 * ((y + x * 5) % 7 == 0)) + (40 if (y * 3 + x * 7) % 11 == 0 else 0)
        img[..., 3] = 190
        return img
    if name == 'lava':
        return palette(r, [(200, 70, 10), (230, 110, 20), (250, 160, 40), (255, 210, 90)], 0.8)
    if name == 'white_wool':
        return palette(r, [(214, 214, 214), (224, 224, 224), (234, 234, 234)])
    if name == 'bookshelf':
        img = tile('oak_planks')
        cols = [(110, 30, 30), (40, 58, 116), (56, 96, 40), (136, 112, 48), (86, 48, 104), (120, 76, 40)]
        for y0 in (1, 9):
            x = 1
            while x < 15:
                w = 1 + (r.random() < 0.45); c = np.array(cols[r.integers(0, len(cols))], float)
                h = 6 - (r.random() < 0.3)
                img[y0 + 6 - h:y0 + 6, x:x + w, :3] = c; img[y0 + 6 - h, x:x + w, :3] = c * 1.3
                x += w + (r.random() < 0.25)
        return img
    if name in ('crafting_top', 'crafting_side', 'crafting_front'):
        img = tile('oak_planks')
        img[0, :, :3] = img[-1, :, :3] = img[:, 0, :3] = img[:, -1, :3] = (92, 66, 38)
        if name == 'crafting_top':
            for k in (5, 10):
                img[k, 1:15, :3] = (92, 66, 38); img[1:15, k, :3] = (92, 66, 38)
        else:
            img[1:4, :, :3] = (122, 92, 54)
            if name == 'crafting_side':
                img[6:8, 3:11, :3] = (172, 172, 178); img[8, 3:11:2, :3] = (120, 120, 126); img[5:9, 11:13, :3] = (90, 62, 34)
            else:
                img[5:12, 5, :3] = (90, 62, 34); img[5:7, 3:8, :3] = (150, 150, 156)
                img[5:12, 10, :3] = (120, 120, 126); img[5:12, 12, :3] = (120, 120, 126)
        return img
    if name in ('furnace_front', 'furnace_side', 'furnace_top'):
        img = palette(r, [(104, 104, 104), (114, 114, 114), (122, 122, 122)], 0.7)
        img[0, :, :3] = img[-1, :, :3] = img[:, 0, :3] = img[:, -1, :3] = (84, 84, 84)
        if name == 'furnace_front':
            img[8:13, 4:12, :3] = (28, 28, 28); img[7, 3:13, :3] = (78, 78, 78); img[12, 5:11, :3] = (48, 40, 36)
            img[3:5, 4:12, :3] = (92, 92, 92)
        return img
    if name in ('chest_front', 'chest_side', 'chest_top'):
        img = palette(r, [(146, 100, 42), (156, 108, 46), (166, 116, 52)])
        img[0, :, :3] = img[-1, :, :3] = img[:, 0, :3] = img[:, -1, :3] = (58, 38, 16)
        if name != 'chest_top':
            img[5, :, :3] = (58, 38, 16)
        if name == 'chest_front':
            img[4:8, 7:9, :3] = (206, 206, 210); img[7, 7:9, :3] = (124, 124, 128)
        return img
    if name in ('bed_head_top', 'bed_foot_top'):
        img = palette(r, [(150, 26, 26), (164, 30, 30), (176, 36, 34)])
        img[:, 0, :3] = img[:, -1, :3] = (138, 102, 62)
        if name == 'bed_head_top':
            img[1:7, 2:14] = palette(R('pillow'), [(222, 222, 222), (232, 232, 232), (242, 242, 242)])[1:7, 2:14]
        return img
    if name in ('bed_side_head', 'bed_side_foot', 'bed_end'):
        img = np.zeros((T, T, 4))
        img[7:10, :] = tile('bed_foot_top')[7:10, :]
        img[10:16, :] = tile('oak_planks')[10:16, :]
        if name == 'bed_end':
            img[10:16, 3:13, 3] = 0
        if name == 'bed_side_head':
            img[7:10, 9:16, :3] = (232, 232, 232)
        return img
    if name in ('door_top', 'door_bottom'):
        img = tile('oak_planks')
        img[:, 0, :3] = img[:, -1, :3] = (98, 72, 40)
        if name == 'door_top':
            img[0, :, :3] = (98, 72, 40)
            for (y0, x0) in ((3, 3), (3, 9)):
                img[y0:y0 + 6, x0:x0 + 4] = 0
            img[3:9, 7:9, :3] = (98, 72, 40)
        else:
            img[-1, :, :3] = (98, 72, 40); img[2, 11:14, :3] = (56, 56, 56)
            img[5:7, 2:14, :3] = (126, 98, 58); img[11:13, 2:14, :3] = (126, 98, 58)
        return img
    if name == 'torch':
        img = np.zeros((T, T, 4))
        img[6:16, 7:9] = (104, 80, 46, 255); img[6:16, 8] = (84, 62, 34, 255)
        img[4:6, 7:9] = (255, 220, 96, 255); img[5, 7:9] = (255, 244, 170, 255); img[6, 7:9] = (214, 128, 44, 255)
        return img
    if name in ('tall_grass', 'sugar_cane', 'sapling'):
        img = np.zeros((T, T, 4))
        if name == 'sugar_cane':
            for x0 in (3, 8, 12):
                img[:, x0:x0 + 2] = (148, 192, 100, 255); img[::5, x0:x0 + 2, :3] = (110, 152, 74)
            return img
        if name == 'sapling':
            img[8:16, 7:9] = (104, 80, 46, 255)
            for (y, x) in ((3, 7), (4, 5), (4, 9), (5, 6), (5, 8), (6, 4), (6, 10), (2, 8), (7, 7)):
                img[y:y + 2, x:x + 2] = (*(LEAF_TINT * 0.95), 255)
            return img
        for k in range(10):
            x = r.integers(1, 15); h = r.integers(6, 15); lean = r.choice([-1, 0, 0, 1])
            for y in range(h):
                xx = int(np.clip(x + lean * (y // 5), 0, 15))
                img[T - 1 - y, xx, :3] = GRASS_TINT * (0.72 + 0.32 * r.random()); img[T - 1 - y, xx, 3] = 255
        return img
    if name in ('rose', 'dandelion'):
        img = np.zeros((T, T, 4))
        img[8:16, 7] = (70, 128, 40, 255); img[11, 8:10] = (70, 128, 40, 255); img[13, 5:7] = (70, 128, 40, 255)
        if name == 'rose':
            for (y, x, c) in ((4, 7, (200, 20, 20)), (4, 8, (230, 40, 34)), (5, 6, (180, 14, 14)), (5, 7, (240, 60, 50)), (5, 8, (200, 20, 20)),
                              (5, 9, (180, 14, 14)), (6, 7, (190, 18, 18)), (6, 8, (220, 30, 30)), (3, 8, (170, 10, 10))):
                img[y, x] = (*c, 255)
        else:
            for (y, x) in ((5, 6), (5, 7), (5, 8), (6, 7), (4, 7), (6, 6), (6, 8)):
                img[y, x] = (250, 220, 40, 255)
            img[5, 7] = (250, 170, 30, 255)
        return img
    if name == 'vine':
        img = np.zeros((T, T, 4))
        for x0 in (1, 5, 9, 13):
            y, x = 0, x0
            while y < T:
                img[y, x % T] = (*(LEAF_TINT * (0.75 + 0.35 * r.random())), 255)
                if r.random() < 0.45:
                    img[y, (x + 1) % T] = (*(LEAF_TINT * 0.95), 255)
                y += 1; x += r.choice([-1, 0, 0, 1])
        return img
    if name == 'cobweb':
        img = np.zeros((T, T, 4)); c = (232, 232, 236, 220)
        for k in range(T):
            img[k, k] = c; img[k, T - 1 - k] = c; img[k, 8] = c; img[8, k] = c
        for rr in (3, 6):
            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                img[int(8 + rr * np.sin(a)) % T, int(8 + rr * np.cos(a)) % T] = c
        return img
    raise KeyError(name)


def build():
    os.makedirs(WEB, exist_ok=True)
    atlas = np.zeros((ROWS * T, COLS * T, 4))
    for i, n in enumerate(TILES):
        atlas[(i // COLS) * T:(i // COLS + 1) * T, (i % COLS) * T:(i % COLS + 1) * T] = tile(n)
    Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'atlas2.png'))
    wolf(); sun(); sign(['', 'my house :D', '', '']); painting()


def box_uv(img, u, v, w, h, d, faces):
    reg = {'top': (u + d, v, w, d), 'bottom': (u + d + w, v, w, d), 'right': (u, v + d, d, h), 'front': (u + d, v + d, w, h),
           'left': (u + d + w, v + d, d, h), 'back': (u + 2 * d + w, v + d, w, h)}
    for k, (x, y, ww, hh) in reg.items():
        img[y:y + hh, x:x + ww] = faces[k](ww, hh)


def wolf():
    """an original tame-wolf skin in the classic look: pale grey-white fur, a darker grey saddle down the back,
    small dark eyes, a black nose, a red collar on the ruff (64 x 32, box UV)"""
    r = np.random.default_rng(21)
    img = np.zeros((32, 64, 4))
    pale, mid, saddle, belly = (214, 210, 202), (192, 186, 178), (150, 142, 134), (226, 222, 214)

    def fur(base, sp=0.07, dark_top=0):
        def f(w, h):
            a = np.zeros((h, w, 4)); v = r.random((h, w))
            a[..., :3] = np.array(base, float) * (1 - sp + 2 * sp * v)[..., None]; a[..., 3] = 255
            if dark_top:
                a[:dark_top, :, :3] = np.array(saddle) * (0.95 + 0.1 * r.random((dark_top, w, 1)))
            return a
        return f

    def face(w, h):                         # 6 x 6
        a = fur(pale)(w, h)
        a[0:2, :, :3] = np.array(mid) * (0.95 + 0.1 * r.random((2, w, 1)))
        a[2, 1, :3] = a[2, 4, :3] = (36, 30, 30)            # eyes
        a[2, 0, :3] = a[2, 5, :3] = (176, 170, 162)
        a[1, 1, :3] = a[1, 4, :3] = (130, 124, 118)          # brows
        return a
    box_uv(img, 0, 0, 6, 6, 4, dict(top=fur(mid), bottom=fur(belly), right=fur(mid), front=face, left=fur(mid), back=fur(saddle)))
    def snout_front(w, h):                  # 3 x 3
        a = fur(belly)(w, h); a[0, 1, :3] = (44, 38, 38); a[0, 0, :3] = a[0, 2, :3] = (150, 142, 136); return a
    def snout_top(w, h):
        a = fur(pale)(w, h); a[-1, 1, :3] = (52, 46, 46); return a
    box_uv(img, 0, 10, 3, 3, 4, dict(top=snout_top, bottom=fur(belly), right=fur(pale), front=snout_front, left=fur(pale), back=fur(pale)))
    def ear(w, h):
        a = fur(saddle)(w, h); return a
    def ear_front(w, h):
        a = fur((150, 120, 116))(w, h); return a
    box_uv(img, 16, 14, 2, 2, 1, dict(top=ear, bottom=ear, right=ear, front=ear_front, left=ear, back=ear))
    # body: 6 x 9 x 6 drawn upright (the model lays it down): its 'top' faces the head, 'back' faces up when laid down
    box_uv(img, 18, 14, 6, 9, 6, dict(top=fur(mid), bottom=fur(mid), right=fur(mid, dark_top=0), front=fur(belly), left=fur(mid), back=fur(saddle)))
    # ruff: 8 x 6 x 7 with the collar where it meets the head
    def ruff_side(w, h):
        a = fur(mid)(w, h); a[:, :2, :3] = (178, 28, 28); a[:, 0, :3] = (204, 40, 36); return a
    def ruff_top(w, h):
        a = fur(saddle)(w, h); a[:2, :, :3] = (178, 28, 28); return a
    def ruff_bottom(w, h):
        a = fur(belly)(w, h); a[:2, :, :3] = (178, 28, 28); return a
    def ruff_front(w, h):
        a = fur(pale)(w, h); a[h - 2:, :, :3] = (186, 30, 30); a[h - 2, :, :3] = (214, 44, 40); return a
    box_uv(img, 21, 0, 8, 6, 7, dict(top=ruff_top, bottom=ruff_bottom, right=ruff_side, front=ruff_front, left=ruff_side, back=fur(saddle)))
    def leg(w, h):
        a = fur(pale)(w, h); a[h - 1:, :, :3] = (176, 170, 162); return a
    box_uv(img, 0, 18, 2, 8, 2, dict(top=fur(mid), bottom=fur(mid), right=leg, front=leg, left=leg, back=leg))
    def tail(w, h):
        a = fur(mid)(w, h); a[h - 2:, :, :3] = (232, 228, 220); a[:2, :, :3] = (140, 132, 124); return a
    box_uv(img, 9, 18, 2, 8, 2, dict(top=fur(saddle), bottom=fur(belly), right=tail, front=tail, left=tail, back=tail))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'wolf2.png'))


def sun():
    img = np.zeros((32, 32, 4))
    yy, xx = np.mgrid[0:32, 0:32]
    d = np.maximum(np.abs(xx - 15.5), np.abs(yy - 15.5))
    img[d < 7] = (255, 246, 196, 255); img[(d >= 7) & (d < 9)] = (255, 220, 130, 170); img[(d >= 9) & (d < 12)] = (255, 196, 110, 70)
    Image.fromarray(img.astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'sun2.png'))


def sign(lines):
    S = 4
    base = np.tile(tile('oak_planks'), (1, 2, 1))[:12, :24]
    big = np.repeat(np.repeat(base, S, 0), S, 1)
    for i, s in enumerate(lines):
        if not s:
            continue
        m = pixfont.text_mask(s); h, w = m.shape
        y0 = 2 + i * 11; x0 = (big.shape[1] - w) // 2
        big[y0:y0 + h, x0:x0 + w][m, :3] = (20, 14, 10)
    Image.fromarray(np.clip(big, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'sign.png'))


def painting():
    """an original 32 x 16 painting: a sunset over hills, in a wooden frame"""
    r = np.random.default_rng(3)
    W_, H_ = 32, 16
    img = np.zeros((H_, W_, 4)); img[..., 3] = 255
    for y in range(H_):
        t = y / (H_ - 1)
        img[y, :, :3] = np.array([250, 170, 90]) * (1 - t) + np.array([120, 90, 160]) * t if y < 9 else (50, 80, 50)
    img[2:9, :, :3] = (np.array([[250, 196, 110]]) * np.linspace(1.0, 0.85, 7)[:, None])[:, None, :]
    img[4:7, 20:23, :3] = (255, 240, 190)
    hills = (9 + 2.2 * np.sin(np.arange(W_) * 0.35) + 1.4 * np.sin(np.arange(W_) * 0.9)).astype(int)
    for x in range(W_):
        img[hills[x]:, x, :3] = (62, 98, 56) if x % 5 else (54, 88, 50)
        img[min(H_ - 1, hills[x] + 3):, x, :3] = (46, 70, 44)
    img[0, :, :3] = img[-1, :, :3] = img[:, 0, :3] = img[:, -1, :3] = (110, 80, 44)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'painting.png'))


if __name__ == '__main__':
    build()
    print('atlas2', len(TILES), 'tiles')
