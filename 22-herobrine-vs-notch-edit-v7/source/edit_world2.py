"""The fight's world, grounded and full: a stone courtyard in a ruined castle at night.  North: the castle wall, towers,
battlements, a gatehouse with banners, lit windows and the keep behind.  West: a spruce forest climbing to snowy
mountains with a lava fall.  East: a burning village.  Round the courtyard: broken pillars and braziers.  Exported as
instance lists per block type (only blocks with an open face), plus fires, banners and the far mountains as big blocks.
-> web/world2.json, web/blocks2.png (+ blocks2_em.png)"""
import json, math, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
sys.path.insert(0, os.path.join(HERE, 'deps'))
from lib import fbm
import tex2, tex_ow

# ---- block types (index into the texture strip; grass gets its own top / side / bottom)
TYPES = ['stonebrick', 'crackedbrick', 'mossybrick', 'cobble', 'planks', 'log', 'leaves', 'grass', 'dirt', 'stone',
         'snow', 'obsidian', 'glow', 'lava', 'gravel', 'darkplanks', 'gold', 'path']
TI = {n: i for i, n in enumerate(TYPES)}
X0, X1, Z0, Z1, Y0, Y1 = -64, 64, -64, 64, -6, 60
G = np.zeros((X1 - X0, Y1 - Y0, Z1 - Z0), np.uint8)       # 0 = air, else TI + 1
rng = np.random.default_rng(12)


def put(x, y, z, t):
    i, j, k = x - X0, y - Y0, z - Z0
    if 0 <= i < G.shape[0] and 0 <= j < G.shape[1] and 0 <= k < G.shape[2]:
        G[i, j, k] = TI[t] + 1


def get(x, y, z):
    i, j, k = x - X0, y - Y0, z - Z0
    if 0 <= i < G.shape[0] and 0 <= j < G.shape[1] and 0 <= k < G.shape[2]:
        return G[i, j, k]
    return 0


def fill(x0, x1, y0, y1, z0, z1, t):
    for x in range(x0, x1):
        for z in range(z0, z1):
            for y in range(y0, y1):
                put(x, y, z, t)


# ------------------------------------------------------------------ ground
def ground_h(x, z):
    if -16 <= x <= 16 and -9 <= z <= 9:
        return 0
    d = max(abs(x) - 16, abs(z) - 9, 0)
    h = 1.2 * (fbm(x, 0.0, z, 18, 1, 3) - 0.5) * 4 + min(d, 8) * 0.12
    if x < -18:                                          # the forest climbs to the west
        h += (-18 - x) * 0.18 + 3 * (fbm(x, 0.0, z, 30, 2, 3) - 0.3)
    if z > 14:                                            # a low rise to the south
        h += (z - 14) * 0.08
    return int(round(h))


def ground():
    for x in range(X0, X1):
        for z in range(Z0, Z1):
            h = ground_h(x, z)
            fill(x, x + 1, Y0, h - 3, z, z + 1, 'stone'); fill(x, x + 1, h - 3, h, z, z + 1, 'dirt'); put(x, h - 1, z, 'grass')
    # the courtyard: stone bricks, some cracked, mossy at the edges, a worn path
    for x in range(-16, 17):
        for z in range(-9, 10):
            r = rng.random()
            edge = abs(x) > 13 or abs(z) > 7
            t = 'mossybrick' if (edge and r < 0.4) else ('crackedbrick' if r < 0.18 else ('stonebrick' if r < 0.9 else 'gravel'))
            if abs(z) <= 1 and abs(x) < 16 and rng.random() < 0.5:
                t = 'path'
            put(x, -1, z, t)
            for y in range(0, 6):
                put(x, y, z, None) if False else None


# ------------------------------------------------------------------ the castle to the north
def castle():
    zw = -16                                              # inner face of the wall
    for x in range(-46, 47):
        for y in range(-1, 14):
            for z in (zw - 2, zw - 1):
                r = rng.random()
                put(x, y, z, 'mossybrick' if (y < 3 and r < 0.4) else ('crackedbrick' if r < 0.15 else 'stonebrick'))
        if x % 2 == 0:                                     # battlements
            put(x, 14, zw - 1, 'stonebrick'); put(x, 14, zw - 2, 'stonebrick')
    # broken stretches of the wall
    for (bx, bw, bd) in ((-34, 5, 6), (27, 4, 5)):
        for x in range(bx, bx + bw):
            for y in range(14 - bd + int(rng.integers(0, 3)), 15):
                put(x, y, zw - 1, None if False else 'stonebrick') if False else None
                i, j = x - X0, y - Y0
                G[i, j, zw - 1 - Z0] = 0; G[i, j, zw - 2 - Z0] = 0
    # the gatehouse: two flanking towers and a dark arch
    for tx in (-7, 4):
        for x in range(tx, tx + 4):
            for z in range(zw - 4, zw + 1):
                for y in range(-1, 20):
                    if x in (tx, tx + 3) or z in (zw - 4, zw):
                        put(x, y, z, 'stonebrick' if rng.random() > 0.12 else 'crackedbrick')
        for x in range(tx - 1, tx + 5, 2):
            put(x, 20, zw - 4, 'stonebrick'); put(x, 20, zw, 'stonebrick')
        put(tx + 1, 12, zw + 1, 'glow'); put(tx + 2, 16, zw + 1, 'glow')
    for x in range(-3, 4):
        for y in range(-1, 8 - (1 if abs(x) == 3 else 0)):
            for z in (zw - 2, zw - 1):
                i, j, k = x - X0, y - Y0, z - Z0
                G[i, j, k] = 0
    # corner towers
    for tx in (-30, 24):
        cx, cz = tx + 3, zw - 4
        for x in range(cx - 4, cx + 5):
            for z in range(cz - 4, cz + 5):
                d = math.hypot(x - cx, z - cz)
                if 3.0 <= d < 4.5:
                    for y in range(-1, 26):
                        put(x, y, z, 'stonebrick' if rng.random() > 0.1 else 'mossybrick')
                if d < 4.6:
                    for y in range(26, 26 + int(max(0, 7 - d * 1.6))):
                        put(x, y, z, 'darkplanks')
        for y in (8, 14, 20):
            put(cx, y, cz + 4, 'glow')
    # the keep behind the wall, tall, with lit windows
    for x in range(-6, 7):
        for z in range(-34, -24):
            for y in range(-1, 38):
                if x in (-6, 6) or z in (-34, -25):
                    put(x, y, z, 'stonebrick' if rng.random() > 0.08 else 'crackedbrick')
    for x in range(-7, 8, 2):
        put(x, 38, -25, 'stonebrick'); put(x, 38, -34, 'stonebrick')
    for y in (10, 16, 22, 28, 33):
        for x in (-3, 0, 3):
            if rng.random() < 0.8:
                put(x, y, -24, 'glow'); put(x, y + 1, -24, 'glow')
    for x in (-38, -20, 14, 36):                          # wall-walk lights
        put(x, 10, zw, 'glow')


# ------------------------------------------------------------------ forest and mountains to the west, the village to the east
def spruce(x, z, h):
    y0 = ground_h(x, z)
    for y in range(y0, y0 + h):
        put(x, y, z, 'log')
    for k in range(h - 1):
        y = y0 + 2 + k
        r = max(0, int((h - k) * 0.35))
        if k % 2 == 1:
            r = max(0, r - 1)
        for dx in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if abs(dx) + abs(dz) <= r + 0.5 and (dx or dz):
                    put(x + dx, y, z + dz, 'leaves')
    put(x, y0 + h, z, 'leaves'); put(x, y0 + h + 1, z, 'leaves')


def forest():
    n = 0
    for _ in range(900):
        x = int(rng.integers(X0 + 2, -19)); z = int(rng.integers(Z0 + 2, Z1 - 2))
        if z > -14 and z < 12 and x > -24:
            continue
        if fbm(x, 0.0, z, 14, 9, 2) < 0.42:
            continue
        spruce(x, z, int(rng.integers(7, 14))); n += 1
    for _ in range(140):                                  # a scatter behind the castle and south too
        x = int(rng.integers(X0 + 2, X1 - 2)); z = int(rng.integers(Z0 + 2, Z1 - 2))
        if (-50 < x < 50 and -40 < z < -12) or (-20 < x < 22 and -12 <= z <= 14) or x > 18 or (-30 < x < 40 and z > 14):
            continue
        spruce(x, z, int(rng.integers(7, 12))); n += 1
    return n


FIRES = []


def house(x0, z0, w, d, burning, rot=False):
    y0 = max(ground_h(x, z) for x in range(x0, x0 + w) for z in range(z0, z0 + d))
    for x in range(x0, x0 + w):
        for z in range(z0, z0 + d):
            put(x, y0 - 1, z, 'cobble')
            edge = x in (x0, x0 + w - 1) or z in (z0, z0 + d - 1)
            corner = x in (x0, x0 + w - 1) and z in (z0, z0 + d - 1)
            for y in range(y0, y0 + 4):
                if corner:
                    put(x, y, z, 'log')
                elif edge:
                    put(x, y, z, 'cobble' if y == y0 else 'planks')
    for x in range(x0, x0 + w):
        for z in range(z0, z0 + d):
            if (x in (x0 + w // 2,) and z == z0) and True:
                pass
    # windows (lit) and a door gap facing west, toward the courtyard
    for z in range(z0 + 1, z0 + d - 1, 2):
        put(x0, y0 + 2, z, 'glow')
    put(x0, y0, z0 + d // 2, None) if False else None
    G[x0 - X0, y0 - Y0, z0 + d // 2 - Z0] = 0; G[x0 - X0, y0 + 1 - Y0, z0 + d // 2 - Z0] = 0
    # a pitched roof (dark planks), partly burnt away on the burning ones
    for k in range(d // 2 + 1):
        for x in range(x0 - 1, x0 + w + 1):
            for z in (z0 + k, z0 + d - 1 - k):
                if burning and rng.random() < 0.35:
                    continue
                put(x, y0 + 4 + k, z, 'darkplanks')
    if burning:
        for _ in range(3):
            FIRES.append([x0 + float(rng.uniform(0.5, w - 0.5)), y0 + 4 + float(rng.uniform(0, d / 2)), z0 + float(rng.uniform(0.5, d - 0.5)), float(rng.uniform(1.2, 2.2))])


def village():
    for (x0, z0, w, d, b) in ((22, -9, 7, 6, True), (24, 2, 6, 7, False), (31, -3, 8, 6, True), (33, 9, 6, 6, True), (21, 13, 6, 6, False), (40, -12, 7, 7, True),
                              (42, 5, 6, 8, False), (29, 18, 7, 6, True)):
        house(x0, z0, w, d, b)
    # a well
    for x in range(26, 29):
        for z in range(-2, 1):
            if x != 27 or z != -1:
                put(x, 0, z, 'cobble')
    # fences / burnt posts
    for z in range(-12, 22, 3):
        put(19, ground_h(19, z), z, 'log')


def courtyard_props():
    # broken pillars and braziers round the courtyard
    for (x, z, h) in ((-15, -8, 7), (-15, 8, 4), (15, -8, 6), (15, 8, 3), (-8, -9, 5), (8, -9, 8), (-8, 9, 2), (8, 9, 5)):
        for y in range(0, h):
            for (dx, dz) in ((0, 0), (1, 0), (0, 1), (1, 1)):
                if y >= h - 2 and rng.random() < 0.4:
                    continue
                put(x + dx, y, z + dz, 'stonebrick' if rng.random() > 0.3 else 'mossybrick')
    for (x, z) in ((-12, -8), (12, -8), (-12, 8), (12, 8), (0, -10)):
        put(x, 0, z, 'stonebrick'); put(x, 1, z, 'obsidian')
        FIRES.append([x + 0.5, 2.0, z + 0.5, 1.0])
    # a big rift portal in the gate arch: obsidian frame, its plane is drawn by the renderer
    for x in range(-3, 4):
        for y in (-1, 8):
            put(x, y, -17, 'obsidian')
    for y in range(-1, 9):
        put(-3, y, -17, 'obsidian'); put(3, y, -17, 'obsidian')


def mountains():
    """far mountains as 4 m blocks: west (with a lava fall) and round the back"""
    out = []
    S = 4
    for gx in range(-60, 61):
        for gz in range(-60, 61):
            x, z = gx * S, gz * S
            d = max(abs(x), abs(z))
            if d < 70:
                continue
            ridge = (1 - abs(2 * fbm(x, 0.0, z, 120, 21, 4) - 1)) ** 2
            h = (d - 70) * 0.35 * (0.4 + ridge) + 30 * ridge
            if x < -60:
                h += 20
            hh = int(h / S)
            for k in range(max(0, hh - 1), hh + 1):
                y = k * S
                out.append([x, y, z, 'snow' if y > 70 + 10 * fbm(x, 0.0, z, 40, 22, 2) else 'stone'])
    return out


def textures():
    T = 16
    def bricks(base, mortar, seed, crack=False, moss=False):
        r = np.random.default_rng(seed)
        a = tex2.palette(r, [tuple(np.array(base) * f) for f in (0.88, 0.95, 1.0, 1.06)], 0.6)
        for y in (3, 7, 11, 15):
            a[y, :, :3] = mortar
        for row, off in ((0, 0), (1, 4), (2, 0), (3, 4)):
            for x in (off, off + 8):
                a[row * 4:row * 4 + 3, x % 16, :3] = mortar
        if crack:
            for (y, x) in ((1, 3), (2, 4), (3, 4), (4, 5), (5, 6), (6, 6), (9, 10), (10, 11), (11, 11), (12, 12)):
                a[y, x, :3] = (52, 52, 54)
        if moss:
            m = r.random((T, T)) < 0.3
            a[m, :3] = (np.array([70, 104, 46]) * (0.8 + 0.4 * r.random((m.sum(), 1))))
        return a
    tiles = {
        'stonebrick': bricks((122, 122, 124), (78, 78, 80), 1), 'crackedbrick': bricks((116, 116, 118), (76, 76, 78), 2, crack=True),
        'mossybrick': bricks((118, 120, 116), (74, 76, 72), 3, moss=True), 'cobble': tex2.tile('cobblestone'), 'planks': tex2.tile('oak_planks'),
        'log': tex_ow.tile('oak_log') * np.array([0.6, 0.6, 0.6, 1]), 'leaves': tex2.tile('oak_leaves') * np.array([0.55, 0.75, 0.7, 1]),
        'grass': tex2.tile('grass_top') * np.array([0.7, 0.85, 0.75, 1]), 'dirt': tex2.tile('dirt'), 'stone': tex2.tile('stone'),
        'snow': tex_ow.tile('white_wool'), 'obsidian': tex_ow.tile('obsidian'), 'glow': tex2.palette(np.random.default_rng(9), [(255, 200, 110), (255, 220, 140), (240, 170, 80)]),
        'lava': tex2.tile('lava'), 'gravel': tex2.tile('gravel'), 'darkplanks': tex2.tile('oak_planks') * np.array([0.45, 0.4, 0.38, 1]),
        'gold': tex2.palette(np.random.default_rng(8), [(230, 180, 50), (250, 210, 80), (200, 150, 40)]), 'path': tex2.tile('gravel') * np.array([0.8, 0.75, 0.7, 1]),
    }
    grass_side = tex2.tile('grass_side'); grass_side[:5, :, :3] *= np.array([0.7, 0.85, 0.75])
    strip = np.concatenate([tiles[n] for n in TYPES] + [grass_side], 1)
    strip[..., 3] = np.where(strip[..., 3] > 0, 255, 0)
    leaf = TYPES.index('leaves'); strip[:, leaf * 16:(leaf + 1) * 16, 3] = tex2.tile('oak_leaves')[..., 3]
    Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8)).save(os.path.join(WEB, 'blocks2.png'))
    em = np.zeros_like(strip); em[..., 3] = 255
    for n in ('glow', 'lava'):
        i = TYPES.index(n); em[:, i * 16:(i + 1) * 16, :3] = 255
    Image.fromarray(em.astype(np.uint8)).save(os.path.join(WEB, 'blocks2_em.png'))


def export(mount):
    solid = G > 0
    pad = np.pad(solid, 1, constant_values=True)
    pad[:, -1, :] = False                                 # the open sky above
    leafy = np.pad(G == TI['leaves'] + 1, 1)
    open_face = np.zeros_like(solid)
    for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
        sl = tuple(slice(1 + k, pad.shape[a] - 1 + k) for a, k in enumerate(d))
        open_face |= ~pad[sl] | leafy[sl]
    out = {}
    for n in TYPES:
        xs, ys, zs = np.nonzero((G == TI[n] + 1) & open_face)
        out[n] = np.stack([xs + X0, ys + Y0, zs + Z0], 1).astype(int).tolist()
    out['mountains'] = mount
    out['fires'] = [[round(v, 2) for v in f] for f in FIRES]
    out['banners'] = [[-6.0, 17, -15.9, 'red'], [5.0, 17, -15.9, 'red'], [-20, 12, -15.9, 'gold'], [16, 12, -15.9, 'gold'], [-0.5, 7.5, -15.9, 'red']]
    out['portal'] = [0, 3.5, -16.5, 5, 8]
    out['lavafall'] = [-118, 70, -10, 8]
    json.dump(out, open(os.path.join(WEB, 'world2.json'), 'w'))
    print({k: len(v) for k, v in out.items() if isinstance(v, list)})


if __name__ == '__main__':
    textures()
    ground(); castle(); n = forest(); village(); courtyard_props()
    # keep the courtyard clear above the floor (trees and props never in the fight lane)
    for x in range(-16, 17):
        for z in range(-7, 8):
            for y in range(0, 30):
                if -14 <= x <= 14 and -6 <= z <= 6:
                    G[x - X0, y - Y0, z - Z0] = 0
    print('trees', n)
    export(mountains())
