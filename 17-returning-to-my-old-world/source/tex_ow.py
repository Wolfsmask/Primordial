"""Original 16 px block textures in an older blocky-game style (flatter, a little more saturated), procedurally drawn.
-> web/atlas.png (16 x 4 tiles), web/wolf.png (64 x 32 box-UV skin), web/sun.png, web/sign.png"""
import os, zlib
import numpy as np
from PIL import Image
import pixfont

HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
T = 16
TILES = ['grass_top', 'grass_side', 'dirt', 'stone', 'cobblestone', 'mossy_cobblestone', 'oak_planks', 'oak_log',
         'log_top', 'oak_leaves', 'glass', 'sand', 'gravel', 'water', 'tall_grass', 'poppy',
         'dandelion', 'vine', 'cobweb', 'crafting_top', 'crafting_side', 'crafting_front', 'furnace_front', 'furnace_side',
         'furnace_top', 'chest_front', 'chest_side', 'chest_top', 'bed_head_top', 'bed_foot_top', 'bed_side_head', 'bed_side_foot',
         'bed_end', 'door_top', 'door_bottom', 'torch', 'obsidian', 'wheat', 'bookshelf', 'coal_ore',
         'birch_log', 'birch_leaves', 'sugar_cane', 'lily_pad', 'red_mushroom', 'brown_mushroom', 'white_wool', 'farmland']
IDX = {n: i for i, n in enumerate(TILES)}
COLS, ROWS = 16, 4


def rgba(c, a=255):
    return np.array(list(c) + [a], float)


def speckle(base, spread=0.2, n=4, seed=0, blotch=0):
    r = np.random.default_rng(seed)
    v = r.random((T, T))
    if blotch:
        yy, xx = np.mgrid[0:T, 0:T]
        for _ in range(blotch):
            cx, cy, rr = r.integers(0, T), r.integers(0, T), r.integers(1, 3)
            v += ((((xx - cx) % T) ** 2 + ((yy - cy) % T) ** 2) < rr * rr) * r.choice([-0.4, 0.4])
    k = np.clip((np.clip(v, 0, 0.999) * n).astype(int), 0, n - 1)
    img = np.zeros((T, T, 4))
    for i in range(n):
        img[k == i, :3] = np.clip(np.array(base, float) * (1 - spread + 2 * spread * i / (n - 1)), 0, 255)
    img[..., 3] = 255
    return img


def cells(seed, n, outline):
    """irregular stones: voronoi cells with dark joints"""
    r = np.random.default_rng(seed)
    pts = r.random((n, 2)) * T
    yy, xx = np.mgrid[0:T, 0:T] + 0.5
    d = []
    for px, py in pts:
        for ox in (-T, 0, T):
            for oy in (-T, 0, T):
                d.append(np.hypot(xx - px - ox, yy - py - oy))
    d = np.stack(d).reshape(n, 9, T, T).min(1)
    o = np.sort(d, 0)
    lab = np.argmin(d, 0)
    edge = (o[1] - o[0]) < outline
    return lab, edge, o[0]


def tile(name):
    r = np.random.default_rng(zlib.crc32(name.encode()))
    if name == 'grass_top':
        return speckle([104, 172, 66], 0.18, 5, 1)
    if name == 'dirt':
        return speckle([138, 98, 66], 0.24, 4, 2, blotch=6)
    if name == 'farmland':
        img = speckle([112, 78, 50], 0.22, 4, 22, blotch=4)
        img[::4, :, :3] *= 0.72
        return img
    if name == 'grass_side':
        img = tile('dirt'); g = speckle([104, 172, 66], 0.16, 4, 3)
        fr = 3 + (r.random(T) < 0.55) + (r.random(T) < 0.25)
        for x in range(T):
            img[:fr[x], x] = g[:fr[x], x]
        return img
    if name == 'stone':
        return speckle([125, 125, 125], 0.16, 4, 6, blotch=8)
    if name == 'coal_ore':
        img = tile('stone')
        for cx, cy in ((3, 3), (10, 5), (5, 11), (12, 12)):
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1), (-1, 0), (0, -1)):
                if r.random() < 0.85:
                    img[(cy + dy) % T, (cx + dx) % T, :3] = [38, 38, 38] if r.random() < 0.7 else [62, 62, 62]
        return img
    if name in ('cobblestone', 'mossy_cobblestone', 'furnace_side', 'furnace_front', 'furnace_top'):
        lab, edge, d0 = cells(11, 9, 0.9)
        sh = r.random(9) * 0.35 + 0.8
        base = np.array([122, 122, 122.0])
        img = np.zeros((T, T, 4)); img[..., 3] = 255
        img[..., :3] = base * sh[lab][..., None] * (1.12 - 0.05 * d0[..., None] / 3)
        img[..., :3] *= (0.94 + 0.12 * r.random((T, T)))[..., None]
        img[edge, :3] = [64, 64, 64]
        if name == 'mossy_cobblestone':
            m = speckle([78, 112, 48], 0.25, 3, 12)
            yy, xx = np.mgrid[0:T, 0:T]
            mask = (np.sin(xx * 0.7 + 1) + np.cos(yy * 0.55) + r.random((T, T)) * 1.2) > 1.0
            img[mask] = m[mask]
        if name in ('furnace_side', 'furnace_top', 'furnace_front'):
            img = speckle([118, 118, 118], 0.12, 3, 13)
            img[0, :, :3] = img[-1, :, :3] = [92, 92, 92]; img[:, 0, :3] = img[:, -1, :3] = [92, 92, 92]
            if name == 'furnace_front':
                img[8:13, 4:12, :3] = [30, 30, 30]; img[7, 3:13, :3] = [80, 80, 80]
                img[12, 5:11, :3] = [52, 44, 40]
                img[3:5, 4:12, :3] = [96, 96, 96]
        return img
    if name == 'oak_planks':
        img = speckle([162, 128, 80], 0.10, 4, 14)
        for y in range(T):
            img[y, :, :3] *= 0.97 + 0.06 * np.sin(np.arange(T) * 0.9 + y * 2.1)[:, None]
        for y in (3, 7, 11, 15):
            img[y, :, :3] = [104, 80, 48]
        for y0, sx in ((0, 5), (4, 12), (8, 2), (12, 9)):
            img[y0:y0 + 3, sx, :3] = [118, 92, 56]
        return img
    if name == 'sign_wood':
        return tile('oak_planks')
    if name == 'oak_log':
        img = speckle([104, 82, 50], 0.18, 4, 15)
        for x in range(T):
            if r.random() < 0.35:
                img[:, x, :3] *= 0.78
        img[r.random((T, T)) < 0.06, :3] = [70, 56, 34]
        return img
    if name == 'birch_log':
        img = speckle([222, 220, 212], 0.06, 3, 16)
        for _ in range(9):
            x, y, w = r.integers(0, T), r.integers(0, T), r.integers(2, 5)
            img[y, x:x + w, :3] = [48, 48, 44]
        return img
    if name == 'log_top':
        yy, xx = np.mgrid[0:T, 0:T] + 0.5
        rr = np.maximum(np.abs(xx - 8), np.abs(yy - 8)) + 0.4 * r.random((T, T))
        img = np.zeros((T, T, 4)); img[..., 3] = 255
        img[..., :3] = np.where(((rr.astype(int)) % 2 == 0)[..., None], [178, 144, 90], [150, 118, 72])
        img[(rr > 6.6)] = tile('oak_log')[(rr > 6.6)]
        return img
    if name in ('oak_leaves', 'birch_leaves'):
        base = [62, 128, 34] if name == 'oak_leaves' else [96, 140, 58]
        img = speckle(base, 0.32, 4, 17 if name == 'oak_leaves' else 18)
        img[r.random((T, T)) < 0.22, 3] = 0
        return img
    if name == 'glass':
        img = np.zeros((T, T, 4))
        img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = rgba([200, 226, 232])
        for k in range(3):
            img[2 + k, 2 + k] = img[3 + k, 2 + k] = rgba([238, 250, 252], 200)
        for k in range(2):
            img[9 + k, 11 - k] = rgba([238, 250, 252], 180)
        return img
    if name == 'sand':
        return speckle([220, 210, 158], 0.07, 4, 19)
    if name == 'gravel':
        img = speckle([132, 124, 122], 0.3, 5, 20, blotch=10)
        return img
    if name == 'water':
        img = speckle([46, 86, 206], 0.12, 3, 21)
        img[..., 3] = 176
        return img
    if name == 'white_wool':
        return speckle([226, 226, 226], 0.06, 3, 23)
    if name == 'obsidian':
        img = speckle([22, 16, 34], 0.35, 4, 24, blotch=6)
        img[r.random((T, T)) < 0.05, :3] = [70, 52, 104]
        return img
    if name == 'bookshelf':
        img = tile('oak_planks')
        cols = [[120, 30, 30], [40, 60, 120], [60, 100, 40], [140, 120, 50], [90, 50, 110]]
        for y0 in (1, 9):
            x = 1
            while x < 15:
                w = 1 + int(r.random() < 0.4); c = cols[r.integers(0, 5)]
                img[y0:y0 + 6, x:x + w, :3] = c; img[y0 + 1, x:x + w, :3] = np.array(c) * 1.3
                x += w + int(r.random() < 0.3)
        return img
    if name in ('crafting_top', 'crafting_side', 'crafting_front'):
        img = tile('oak_planks')
        img[0, :, :3] = img[-1, :, :3] = img[:, 0, :3] = img[:, -1, :3] = [96, 70, 40]
        if name == 'crafting_top':
            for k in (5, 10):
                img[k, 1:15, :3] = [96, 70, 40]; img[1:15, k, :3] = [96, 70, 40]
        else:
            img[1:4, :, :3] = [126, 96, 58]
            if name == 'crafting_side':      # a saw
                img[6:8, 3:11, :3] = [170, 170, 176]; img[8, 3:11:2, :3] = [120, 120, 126]; img[5:9, 11:13, :3] = [92, 64, 36]
            else:                            # hammer and tongs
                img[5:12, 5, :3] = [92, 64, 36]; img[5:7, 3:8, :3] = [150, 150, 156]
                img[5:12, 10, :3] = [120, 120, 126]; img[5:12, 12, :3] = [120, 120, 126]
        return img
    if name in ('chest_front', 'chest_side', 'chest_top'):
        img = speckle([150, 104, 46], 0.08, 3, 25)
        img[0, :, :3] = img[-1, :, :3] = img[:, 0, :3] = img[:, -1, :3] = [60, 40, 18]
        if name != 'chest_top':
            img[5, :, :3] = [60, 40, 18]
        if name == 'chest_front':
            img[4:8, 7:9, :3] = [200, 200, 204]; img[7, 7:9, :3] = [120, 120, 124]
        return img
    if name in ('bed_head_top', 'bed_foot_top'):
        img = speckle([172, 30, 30], 0.10, 3, 26)
        img[:, 0, :3] = img[:, -1, :3] = [140, 104, 64]
        if name == 'bed_head_top':
            img[1:7, 2:14, :3] = speckle([236, 236, 236], 0.05, 3, 27)[1:7, 2:14, :3]
        return img
    if name in ('bed_side_head', 'bed_side_foot', 'bed_end'):
        img = np.zeros((T, T, 4))
        img[7:10, :] = tile('bed_foot_top')[7:10, :]
        img[10:16, :] = tile('oak_planks')[10:16, :]
        img[10:16, 3:13, 3] = 0 if name == 'bed_end' else 255
        if name == 'bed_side_head':
            img[7:10, 9:16, :3] = [236, 236, 236]
        return img
    if name in ('door_top', 'door_bottom'):
        img = tile('oak_planks')
        img[:, 0, :3] = img[:, -1, :3] = [100, 74, 42]
        if name == 'door_top':
            img[0, :, :3] = [100, 74, 42]
            img[3:12, 3:13] = [0, 0, 0, 0]
            img[3:12, 7:9, :3] = [100, 74, 42]; img[3:12, 7:9, 3] = 255
            img[7, 3:13, :3] = [100, 74, 42]; img[7, 3:13, 3] = 255
        else:
            img[-1, :, :3] = [100, 74, 42]
            img[4, 11:14, :3] = [60, 60, 60]
        return img
    if name == 'torch':
        img = np.zeros((T, T, 4))
        img[6:16, 7:9] = rgba([104, 80, 46]); img[6:16, 8] = rgba([86, 64, 36])
        img[4:6, 7:9] = rgba([255, 214, 90]); img[5, 7:9] = rgba([255, 240, 160]); img[6, 7:9] = rgba([210, 120, 40])
        return img
    if name in ('tall_grass', 'wheat', 'sugar_cane'):
        img = np.zeros((T, T, 4))
        col = {'tall_grass': [92, 160, 56], 'wheat': [176, 162, 66], 'sugar_cane': [140, 196, 98]}[name]
        if name == 'sugar_cane':
            for x0 in (3, 8, 12):
                img[:, x0:x0 + 2] = rgba(col); img[::5, x0:x0 + 2, :3] = [100, 150, 70]
            return img
        for k in range(9 if name == 'tall_grass' else 6):
            x = r.integers(1, 15); h = r.integers(7, 16); lean = r.choice([-1, 0, 1])
            for y in range(h):
                xx = int(np.clip(x + lean * (y // 5), 0, 15))
                img[T - 1 - y, xx] = rgba(np.array(col) * (0.8 + 0.4 * r.random()))
            if name == 'wheat':
                img[T - h:T - h + 3, x] = rgba([206, 186, 92])
        return img
    if name in ('poppy', 'dandelion', 'red_mushroom', 'brown_mushroom'):
        img = np.zeros((T, T, 4))
        if 'mushroom' in name:
            img[10:16, 7:9] = rgba([220, 214, 200])
            cap = [200, 40, 36] if name == 'red_mushroom' else [150, 110, 80]
            img[7:10, 4:12] = rgba(cap); img[6, 5:11] = rgba(cap)
            if name == 'red_mushroom':
                img[7, 6] = img[8, 9] = rgba([240, 240, 240])
            return img
        img[8:16, 7] = rgba([70, 130, 40]); img[11, 8:10] = rgba([70, 130, 40])
        c = [210, 30, 28] if name == 'poppy' else [250, 220, 40]
        img[4:8, 6:10] = rgba(c); img[5:7, 5:11] = rgba(c)
        img[5:7, 7:9] = rgba([60, 20, 10] if name == 'poppy' else [250, 170, 30])
        return img
    if name == 'vine':
        img = np.zeros((T, T, 4))
        for x0 in (1, 5, 9, 13):
            y = 0; x = x0
            while y < T:
                img[y, x % T] = rgba(np.array([58, 112, 34]) * (0.8 + 0.4 * r.random()))
                if r.random() < 0.45:
                    img[y, (x + 1) % T] = rgba([74, 132, 44])
                y += 1; x += r.choice([-1, 0, 0, 1])
        return img
    if name == 'cobweb':
        img = np.zeros((T, T, 4)); c = rgba([232, 232, 236], 220)
        for k in range(T):
            img[k, k] = c; img[k, T - 1 - k] = c; img[k, 8] = c; img[8, k] = c
        for rr in (3, 6):
            for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                img[int(8 + rr * np.sin(a)) % T, int(8 + rr * np.cos(a)) % T] = c
        return img
    if name == 'lily_pad':
        img = np.zeros((T, T, 4))
        yy, xx = np.mgrid[0:T, 0:T]
        m = ((xx - 7.5) ** 2 + (yy - 7.5) ** 2 < 42) & ~((xx > 7) & (np.abs(yy - 7.5) < 1.2))
        img[m] = rgba([40, 104, 30]); img[m & (r.random((T, T)) < 0.3), :3] = [56, 128, 40]
        return img
    raise KeyError(name)


def build():
    os.makedirs(WEB, exist_ok=True)
    atlas = np.zeros((ROWS * T, COLS * T, 4))
    for i, n in enumerate(TILES):
        atlas[(i // COLS) * T:(i // COLS + 1) * T, (i % COLS) * T:(i % COLS + 1) * T] = tile(n)
    Image.fromarray(np.clip(atlas, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'atlas.png'))
    wolf(); sun(); sign(['', 'my house :D', '', ''])


def box_uv(img, u, v, w, h, d, faces):
    """paint a box-UV block: faces = dict(top, bottom, right, front, left, back) -> painter(width, height) arrays"""
    reg = {'top': (u + d, v, w, d), 'bottom': (u + d + w, v, w, d), 'right': (u, v + d, d, h), 'front': (u + d, v + d, w, h),
           'left': (u + d + w, v + d, d, h), 'back': (u + 2 * d + w, v + d, w, h)}
    for k, (x, y, ww, hh) in reg.items():
        img[y:y + hh, x:x + ww] = faces[k](ww, hh)


def wolf():
    """an original grey-and-cream wolf skin (64 x 32, box UV); red collar painted on the ruff"""
    r = np.random.default_rng(5)
    img = np.zeros((32, 64, 4))
    def fur(base, sp=0.12):
        def f(w, h):
            a = np.zeros((h, w, 4)); v = r.random((h, w))
            a[..., :3] = np.array(base) * (1 - sp + 2 * sp * v)[..., None]; a[..., 3] = 255
            return a
        return f
    grey, light, dark = [196, 190, 182], [232, 226, 214], [150, 144, 138]
    def face(w, h):
        a = fur(grey)(w, h)
        a[2, 1] = a[2, w - 2] = [24, 24, 24, 255]           # eyes
        a[2, 0] = a[2, w - 1] = [240, 240, 240, 255]
        a[h - 2:, 1:w - 1, :3] = light
        return a
    # head 6x6x4 at (0,0)
    box_uv(img, 0, 0, 6, 6, 4, dict(top=fur(grey), bottom=fur(light), right=fur(grey), front=face, left=fur(grey), back=fur(dark)))
    # snout 3x3x4 at (0,10)
    def snout(w, h):
        a = fur(light)(w, h); a[0, 1, :3] = [36, 30, 30]; return a
    box_uv(img, 0, 10, 3, 3, 4, dict(top=fur(grey), bottom=fur(light), right=fur(light), front=snout, left=fur(light), back=fur(light)))
    # ear 2x2x1 at (16,14)
    box_uv(img, 16, 14, 2, 2, 1, dict(top=fur(dark), bottom=fur(dark), right=fur(dark), front=fur([120, 100, 100]), left=fur(dark), back=fur(dark)))
    # body 6x9x6 at (18,14) (stands upright in the box and is laid down by the model)
    box_uv(img, 18, 14, 6, 9, 6, dict(top=fur(grey), bottom=fur(light), right=fur(grey), front=fur(grey), left=fur(grey), back=fur(dark)))
    # ruff (mane) 8x6x7 at (21,0), with the collar
    def ruff(w, h):
        a = fur(grey)(w, h); a[h - 2:h, :, :3] = [190, 30, 30]; a[h - 2, :, :3] = [220, 44, 40]; return a
    box_uv(img, 21, 0, 8, 6, 7, dict(top=fur(grey), bottom=fur(light), right=ruff, front=ruff, left=ruff, back=ruff))
    # leg 2x8x2 at (0,18)
    def leg(w, h):
        a = fur(grey)(w, h); a[h - 2:, :, :3] = light; return a
    box_uv(img, 0, 18, 2, 8, 2, dict(top=fur(grey), bottom=fur(light), right=leg, front=leg, left=leg, back=leg))
    # tail 2x8x2 at (9,18)
    def tail(w, h):
        a = fur(grey)(w, h); a[h - 2:, :, :3] = light; return a
    box_uv(img, 9, 18, 2, 8, 2, dict(top=fur(grey), bottom=fur(light), right=tail, front=tail, left=tail, back=tail))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'wolf.png'))


def sun():
    img = np.zeros((32, 32, 4))
    yy, xx = np.mgrid[0:32, 0:32]
    d = np.maximum(np.abs(xx - 15.5), np.abs(yy - 15.5))
    img[d < 8] = [255, 248, 200, 255]; img[(d >= 8) & (d < 10)] = [255, 214, 120, 150]; img[(d >= 10) & (d < 13)] = [255, 190, 90, 60]
    Image.fromarray(img.astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'sun.png'))


def sign(lines):
    """sign board texture (24 x 12 px block face drawn at 4x: 96 x 48) with the text"""
    S = 4
    base = np.tile(tile('oak_planks'), (1, 2, 1))[:12, :24]
    big = np.repeat(np.repeat(base, S, 0), S, 1)
    for i, s in enumerate(lines):
        if not s:
            continue
        m = pixfont.text_mask(s)
        h, w = m.shape
        y0 = 2 + i * 11; x0 = (big.shape[1] - w) // 2
        if x0 < 0:
            m = m[:, :big.shape[1]]; x0 = 0; w = m.shape[1]
        reg = big[y0:y0 + h, x0:x0 + w]
        reg[m, :3] = [20, 14, 10]
    Image.fromarray(np.clip(big, 0, 255).astype(np.uint8), 'RGBA').save(os.path.join(WEB, 'sign.png'))


if __name__ == '__main__':
    build()
    print('atlas', len(TILES), 'tiles')
