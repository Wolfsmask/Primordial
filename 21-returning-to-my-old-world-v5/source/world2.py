"""v2 old world: old-generation terrain (3D noise: cliffs, overhangs, cave mouths, ore in the rock, two small floating
islands, a waterfall off the cliffs into the lake), a meadow by the lake at sunset, and ten years of overgrowth on the
first house (door to the west, toward the sunset), the path, the sign, the first-night dirt pillar.  Block light and
sky light are flood-filled like the classic game and baked with smooth lighting.  -> web/world2.bin + web/world2.json"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
sys.path.insert(0, os.path.join(HERE, '..', 'bleed3'))
from lib import fbm
import tex2 as TX

NX, NY, NZ = 384, 128, 384
SPAWN = (180, 191)
SEA = 63
HOUSE = (196, 202, 188, 194)            # x0, x1, z0, z1
# ---- block ids
AIR, GRASS, DIRT, STONE, COBBLE, MOSSY, PLANKS, LOG, LEAVES, GLASS, SAND, GRAVEL, WATER = range(13)
TALLGRASS, ROSE, DANDELION, CRAFT, FURNACE, CHEST, BED_HEAD, BED_FOOT, FALLW, SAPLING, BOOKS, COAL = range(13, 25)
BIRCH, BIRCH_LEAVES, CANE, IRON, CLAY, WOOL, FENCE = range(25, 32)
T = TX.IDX
FULL = {GRASS, DIRT, STONE, COBBLE, MOSSY, PLANKS, LOG, SAND, GRAVEL, CRAFT, FURNACE, CHEST, BOOKS, COAL, BIRCH, IRON, CLAY, WOOL}
CUTOUT_CUBE = {LEAVES, BIRCH_LEAVES, GLASS}
PLANTS = {TALLGRASS: 'tall_grass', ROSE: 'rose', DANDELION: 'dandelion', CANE: 'sugar_cane', SAPLING: 'sapling'}
CUBE_TILES = {GRASS: ('grass_top', 'grass_side', 'dirt'), DIRT: ('dirt',) * 3, STONE: ('stone',) * 3, COBBLE: ('cobblestone',) * 3,
              MOSSY: ('mossy_cobblestone',) * 3, PLANKS: ('oak_planks',) * 3, LOG: ('log_top', 'oak_log', 'log_top'),
              LEAVES: ('oak_leaves',) * 3, GLASS: ('glass',) * 3, SAND: ('sand',) * 3, GRAVEL: ('gravel',) * 3,
              CRAFT: ('crafting_top', 'crafting_side', 'oak_planks'), FURNACE: ('furnace_top', 'furnace_side', 'furnace_top'),
              CHEST: ('chest_top', 'chest_side', 'chest_top'), BOOKS: ('oak_planks', 'bookshelf', 'oak_planks'),
              COAL: ('coal_ore',) * 3, IRON: ('iron_ore',) * 3, BIRCH: ('log_top', 'birch_log', 'log_top'),
              BIRCH_LEAVES: ('birch_leaves',) * 3, CLAY: ('clay',) * 3, WOOL: ('white_wool',) * 3}
FRONT = {}

W = np.zeros((NX, NY, NZ), np.uint8)
EXTRA = dict(torches=[], vines=[], webs=[])


def ss(u):
    u = np.clip(u, 0, 1)
    return u * u * (3 - 2 * u)


def nz(x, z, s, seed, o=4):
    return fbm(x, 0.0, z, s, seed, o)


def fields():
    X, Z = np.meshgrid(np.arange(NX, dtype=float), np.arange(NZ, dtype=float), indexing='ij')
    hcx, hcz = (HOUSE[0] + HOUSE[1]) / 2, (HOUSE[2] + HOUSE[3]) / 2
    d = np.hypot(X - hcx, Z - hcz)
    L = ((X - 150) / 24) ** 2 + ((Z - 184) / 31) ** 2 + 0.35 * (nz(X, Z, 12, 5, 2) - 0.5)
    # the mountains: an arc round the lake's west and north
    M = np.maximum(ss((102 - X) / 36), ss((130 - Z) / 30) * ss((176 - X) / 50))
    M = np.maximum(M, 0.85 * ss((Z - 236) / 36) * ss((160 - X) / 46))
    shore = ss((L - 1.35) / 2.2)                     # the land rises slowly away from the water
    hills = (4 + 16 * nz(X, Z, 70, 3)) * ss((d - 34) / 50) * shore * (1 - M)
    base = 64.3 + hills + M * (32 + 16 * nz(X, Z, 55, 8))
    meadow = np.exp(-(d / 30) ** 2 * 1.6)
    base = base * (1 - meadow) + 64.3 * meadow
    # the lake bowl and its shore
    base = np.where(L < 1, np.minimum(base, 56.5 + 6.0 * L), base)
    base = np.where((L >= 1) & (L < 1.45), np.minimum(base, 62.6 + (L - 1) * 5), base)
    amp = (0.8 + 27 * M + 5 * ss((d - 45) / 40) * shore * (1 - M)) * ss((d - 24) / 16) * ss((L - 1.25) / 1.2)
    return X, Z, base, amp, M, L, d


def terrain():
    X, Z, base, amp, M, L, d = fields()
    Y0, Y1 = 36, 124
    W[:, :Y0, :] = STONE
    for x0 in range(0, NX, 24):
        xs = slice(x0, x0 + 24)
        nxs = min(24, NX - x0)
        Xc = np.broadcast_to(X[xs][:, None, :], (nxs, Y1 - Y0, NZ)); Zc = np.broadcast_to(Z[xs][:, None, :], (nxs, Y1 - Y0, NZ))
        Yc = np.broadcast_to(np.arange(Y0, Y1, dtype=float)[None, :, None], (nxs, Y1 - Y0, NZ))
        n3 = fbm(Xc, Yc * 1.75, Zc, 24, 31, 4)
        D = base[xs][:, None, :] - Yc + amp[xs][:, None, :] * (2 * n3 - 1)
        solid = D > 0
        # spaghetti caves through the mountains
        mm = (M[xs][:, None, :] > 0.25) & (Yc < base[xs][:, None, :] - 2) & (Yc > 48)
        if mm.any():
            c1 = fbm(Xc, Yc * 1.3, Zc, 20, 41, 2); c2 = fbm(Xc + 77.7, Yc * 1.3, Zc - 33.3, 20, 42, 2)
            solid &= ~(mm & (np.abs(c1 - 0.5) < 0.032) & (np.abs(c2 - 0.5) < 0.032))
        W[xs, Y0:Y1, :] = np.where(solid, STONE, AIR)
        print('slab', x0, flush=True) if x0 % 96 == 0 else None
    islands()
    # layers, column by column from the top: grass where it meets the air, three of dirt, then stone
    cnt = np.zeros((NX, NZ), np.int16)
    for y in range(NY - 1, -1, -1):
        s = W[:, y, :] == STONE
        cnt = np.where(s, cnt + 1, 0)
        W[:, y, :] = np.where(s & (cnt == 1) & (y >= SEA - 1), GRASS, np.where(s & (cnt >= 1) & (cnt <= 4), DIRT, W[:, y, :]))
    # the lake: water, sand and clay at the shore, gravel in patches
    lake = (L < 1.55)[:, None, :]
    ys = np.arange(NY)[None, :, None]
    near = lake & (ys >= SEA - 3) & (ys <= SEA + 1) & np.isin(W, [GRASS, DIRT])
    W[near] = SAND
    gr = (nz(X, Z, 9, 12, 2) > 0.62)[:, None, :] & near
    W[gr] = GRAVEL
    deep = lake & (ys < SEA - 3) & np.isin(W, [GRASS, DIRT])
    W[deep] = np.where(np.random.default_rng(1).random(deep.sum()) < 0.25, CLAY, SAND)
    W[(L < 1.6)[:, None, :] & (ys <= SEA) & (W == AIR)] = WATER
    # bare rock and gravel on the steep high ground, ores in the stone
    rng = np.random.default_rng(2)
    st = W == STONE
    r = rng.random(W.shape, dtype=np.float32)
    W[st & (r < 0.012) & (ys < 105)] = COAL
    W[st & (r > 0.9935) & (ys < 90)] = IRON
    hi = (M > 0.55)[:, None, :] & (W == GRASS) & (nz(X, Z, 7, 13, 2) > 0.66)[:, None, :]
    W[hi] = GRAVEL
    return base


def islands():
    for (cx, cy, cz, r) in ((118, 100, 150, 7.5), (104, 106, 212, 5.0), (140, 110, 132, 3.5)):
        x0, x1, z0, z1 = int(cx - r - 2), int(cx + r + 3), int(cz - r - 2), int(cz + r + 3)
        for x in range(x0, x1):
            for z in range(z0, z1):
                q = math.hypot(x - cx, z - cz) / r + 0.25 * (nz(x, z, 4, 77, 2) - 0.5)
                if q >= 1:
                    continue
                top = cy + 1.5 * (1 - q)
                bot = cy - r * 1.35 * (1 - q) ** 0.75
                W[x, int(bot):int(top) + 1, z] = STONE


def surface(x, z):
    col = W[x, :, z]
    ys = np.nonzero(np.isin(col, list(FULL | {WATER})))[0]
    return int(ys.max()) if len(ys) else 0


def tree(x, z, kind=LOG, leaves=LEAVES, h=None, rng=None, big=False, y0=None):
    y0 = y0 if y0 is not None else surface(x, z) + 1
    if W[x, y0 - 1, z] not in (GRASS, DIRT):
        return
    h = h or int(rng.integers(4, 7))
    if any(W[x, y, z] not in (AIR, TALLGRASS, ROSE, DANDELION) for y in range(y0, y0 + h + 2)):
        return
    for y in range(y0, y0 + h):
        W[x, y, z] = kind
    W[x, y0 - 1, z] = DIRT
    top = y0 + h
    rad = [2, 2, 1, 1] if not big else [3, 3, 2, 2, 1]
    for k, r in enumerate(rad):
        y = top - len(rad) + 1 + k + (1 if big else 0)
        for ax in range(-r, r + 1):
            for az in range(-r, r + 1):
                if abs(ax) == r and abs(az) == r and (rng.random() < 0.6 or k >= len(rad) - 2):
                    continue
                if 0 <= x + ax < NX and 0 <= z + az < NZ and W[x + ax, y, z + az] in (AIR, TALLGRASS):
                    W[x + ax, y, z + az] = leaves
    if big:
        for ax, az in ((1, 0), (0, 1), (-1, 0)):
            W[x + ax * 2, top - 2, z + az * 2] = kind


def plants(rng, H):
    """tall grass a little thicker than it was (ten years), roses and dandelions"""
    X, Z, base, amp, M, L, d = fields()
    for x in range(1, NX - 1):
        for z in range(1, NZ - 1):
            y = surface(x, z)
            if W[x, y, z] != GRASS or W[x, y + 1, z] != AIR:
                continue
            dd = d[x, z]
            dens = 0.5 if dd < 34 else (0.2 if M[x, z] < 0.5 else 0.08)
            r = rng.random()
            if r < dens:
                W[x, y + 1, z] = TALLGRASS
            elif r < dens + 0.03:
                W[x, y + 1, z] = ROSE if rng.random() < 0.5 else DANDELION


def house(rng):
    x0, x1, z0, z1 = HOUSE
    for x in range(x0 - 1, x1 + 2):
        for z in range(z0 - 1, z1 + 2):
            W[x, 64, z] = GRASS if (x < x0 or x > x1 or z < z0 or z > z1) else COBBLE
            W[x, 65:80, z] = AIR
            W[x, 58:64, z] = DIRT
    for x in range(x0, x1 + 1):
        for z in range(z0, z1 + 1):
            edge = x in (x0, x1) or z in (z0, z1)
            corner = x in (x0, x1) and z in (z0, z1)
            if corner:
                W[x, 64:69, z] = LOG
            elif edge:
                W[x, 64, z] = MOSSY if rng.random() < 0.55 else COBBLE
                W[x, 65:68, z] = PLANKS
                W[x, 68, z] = LOG
            else:
                W[x, 64, z] = PLANKS
                W[x, 68, z] = PLANKS
    W[x0 + 1:x1, 69, z0 + 1:z1] = PLANKS
    W[x0 + 2:x1 - 1, 70, z0 + 2:z1 - 1] = PLANKS
    door = (x0, 191)
    W[door[0], 65:67, door[1]] = AIR
    for (x, z) in ((x0, 189), (x0, 193), (199, z1)):
        W[x, 66, z] = GLASS
    W[x1, 66, 190] = GLASS; W[x1, 66, 192] = AIR
    EXTRA['webs'].append((x1, 66, 192))
    # inside: the bed along the east wall, the chest, the crafting table and furnace in the corner, a bookshelf
    W[201, 65, 189] = BED_HEAD; W[200, 65, 189] = BED_FOOT
    W[198, 65, 193] = CHEST; FRONT[(198, 65, 193)] = (5, 'chest_front')
    W[201, 65, 193] = CRAFT; FRONT[(201, 65, 193)] = (1, 'crafting_front')
    W[200, 65, 193] = FURNACE; FRONT[(200, 65, 193)] = (5, 'furnace_front')
    W[197, 65:67, 193] = BOOKS
    EXTRA['torches'] += [(201, 66, 190, 'e'), (197, 66, 189, 'n'), (199, 67, 193, 's')]
    EXTRA['webs'] += [(197, 67, 189), (201, 67, 193), (201, 67, 189)]
    EXTRA['torches'] += [(x0 - 1, 66, 190, 'e')]            # beside the door, outside
    # vines on the west and north walls, hanging off the roof edge
    for z in range(z0, z1 + 1):
        for y in range(64, 69):
            if rng.random() < 0.45 and W[x0 - 1, y, z] == AIR and not (z == 191 and y < 67) and not (z == 190 and y == 66):
                EXTRA['vines'].append((x0 - 1, y, z, 'e'))
    for x in range(x0, x1 + 1):
        for y in range(64, 69):
            if rng.random() < 0.5:
                EXTRA['vines'].append((x, y, z0 - 1, 's'))
            if rng.random() < 0.25:
                EXTRA['vines'].append((x, y, z1 + 1, 'n'))
    # the oak that grew at the north-west corner and leaned its crown over the roof
    tree(x0 - 3, z0 - 2, h=6, rng=rng, big=True, y0=65)
    for x in range(x0 - 2, x0 + 3):
        for z in range(z0 - 1, z0 + 4):
            if rng.random() < 0.7 and W[x, 71, z] == AIR:
                W[x, 71, z] = LEAVES
            if rng.random() < 0.3 and W[x, 72, z] == AIR:
                W[x, 72, z] = LEAVES
    for (sx, sz) in ((x1 + 2, z1 + 1), (x0 - 2, z1 + 2)):
        W[sx, 65, sz] = SAPLING


def path(rng):
    for x in range(178, HOUSE[0]):
        for z in (190, 191, 192):
            if z != 191 and rng.random() < 0.4:
                continue
            r = rng.random()
            W[x, 64, z] = GRASS if r < 0.2 else (MOSSY if r < 0.52 else (GRAVEL if r < 0.66 else COBBLE))
            if z == 191 or rng.random() < 0.6:
                W[x, 65, z] = AIR


def landmarks(rng):
    # the dirt pillar from the first night, torch still burning on top
    py0 = surface(186, 183) + 1
    for y in range(py0, py0 + 7):
        W[186, y, 183] = DIRT
    EXTRA['torches'].append((186, py0 + 7, 183, 'u'))
    # sugar cane round the lake
    X, Z, base, amp, M, L, d = fields()
    cands = [(x, z) for x in range(120, 185, 3) for z in range(150, 225, 3) if 1.0 < L[x, z] < 1.35]
    for i in rng.choice(len(cands), size=min(14, len(cands)), replace=False):
        x, z = cands[i]
        y = surface(x, z)
        if W[x, y, z] in (SAND, GRASS) and W[x, y + 1, z] == AIR:
            for k in range(1, 2 + int(rng.integers(1, 3))):
                W[x, y + k, z] = CANE


def waterfall():
    """water that broke out of the cliff face high above the lake and falls straight down into it"""
    best = None
    for x in range(70, 132):
        for z in range(120, 240):
            for y in range(100, 76, -1):
                if W[x, y, z] in (STONE, DIRT, GRASS, COAL, IRON) and W[x + 1, y, z] == AIR and W[x, y + 1, z] != AIR:
                    k = 0
                    while y - k - 1 > 30 and W[x + 1, y - k - 1, z] == AIR:
                        k += 1
                    below = W[x + 1, y - k - 1, z]
                    score = k + (10 if below == WATER else 0) - math.hypot(x - 150, z - 184) * 0.25
                    if k > 14 and (best is None or score > best[0]):
                        best = (score, x, y, z, k)
                    break
    if best is None:
        print('no waterfall site'); return None
    _, x, y, z, k = best
    W[x, y, z] = WATER
    for yy in range(y - k, y + 1):
        W[x + 1, yy, z] = FALLW
    print('waterfall at', x, y, z, 'drop', k)
    return (x + 1, y, z, k)


def forest(rng):
    X, Z, base, amp, M, L, d = fields()
    for k in range(1400):
        x, z = int(rng.integers(6, NX - 6)), int(rng.integers(6, NZ - 6))
        if d[x, z] < 26 or L[x, z] < 1.7 or (174 < x < 200 and 186 < z < 196):
            continue
        dens = nz(x, z, 34, 9, 2) * (1.15 if M[x, z] < 0.4 else 0.55)
        if rng.random() > dens * 0.9:
            continue
        if rng.random() < 0.22:
            tree(x, z, BIRCH, BIRCH_LEAVES, h=int(rng.integers(5, 7)), rng=rng)
        else:
            tree(x, z, rng=rng, big=rng.random() < 0.12)
    for (x, z) in ((118, 150), (104, 212)):          # one on each big floating island
        tree(x, z, rng=rng)
    for (x, z) in ((176, 200), (208, 180), (212, 199), (189, 205), (171, 181), (214, 188)):
        tree(x, z, rng=rng, big=(x == 208))


# ------------------------------------------------------------------ light
def opacity():
    op = np.zeros(W.shape, np.int8)
    op[np.isin(W, list(FULL))] = 15
    op[np.isin(W, [LEAVES, BIRCH_LEAVES])] = 1
    op[np.isin(W, [WATER, FALLW])] = 2
    op[np.isin(W, [BED_HEAD, BED_FOOT])] = 0
    return op


def shift(a, d, fill):
    out = np.full_like(a, fill)
    src = tuple(slice(k, None) if k > 0 else (slice(None, k) if k < 0 else slice(None)) for k in d)
    dst = tuple(slice(None, -k) if k > 0 else (slice(-k, None) if k < 0 else slice(None)) for k in d)
    out[dst] = a[src]
    return out


NB = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]


def spread(L, op, n):
    solid = op >= 15
    for _ in range(n):
        best = L.copy()
        for d in NB:
            best = np.maximum(best, shift(L, d, 0) - 1 - op)
        L = np.where(solid, 0, np.maximum(L, best)).astype(np.int8)
    return L


def light():
    op = opacity()
    sky = np.zeros(W.shape, np.int8)
    cur = np.full((NX, NZ), 15, np.int16)
    for y in range(NY - 1, -1, -1):
        cur = np.where(op[:, y, :] >= 15, 0, np.maximum(cur - op[:, y, :], 0))
        sky[:, y, :] = cur
    sky = spread(sky, op, 15)
    blk = np.zeros(W.shape, np.int8)
    for (x, y, z, _) in EXTRA['torches']:
        blk[x, y, z] = 14
    blk = spread(blk, op, 14)
    return sky, blk


GAMMA = 0.5                     # half way to the game's 'Bright' setting, as most people played it


def lm(level):
    """the classic light curve: level 0..15 -> brightness, with the brightness setting"""
    f = 1.0 - np.asarray(level, np.float32) / 15.0
    b = (1 - f) / (f * 3 + 1) * 0.95 + 0.05
    return b * (1 - GAMMA) + (1 - (1 - b) ** 4) * GAMMA


SKY_COL = np.array([1.0, 0.87, 0.74], np.float32) * 0.88       # sunset: the sky light is low and warm
TORCH_COL = np.array([1.0, 0.88, 0.7], np.float32)


# ------------------------------------------------------------------ meshing
class Geo:
    def __init__(self):
        self.g = {k: dict(pos=[], uv=[], col=[]) for k in ('opaque', 'cutout', 'water')}

    def quad(self, grp, P, UV, C):
        self.g[grp]['pos'].append(np.asarray(P, np.float32)); self.g[grp]['uv'].append(np.asarray(UV, np.float32))
        self.g[grp]['col'].append(np.asarray(C, np.float32))

    def quads(self, grp, P, UV, C):
        if len(P):
            self.g[grp]['pos'].append(np.asarray(P, np.float32).reshape(-1, 4, 3)); self.g[grp]['uv'].append(np.asarray(UV, np.float32).reshape(-1, 4, 3))
            self.g[grp]['col'].append(np.asarray(C, np.float32).reshape(-1, 4, 3))


def tile_uv(tile, u, v):
    """(u, v, layer) for the renderer's texture array: local (u, v) in 0..1 (v down the tile), the tile index"""
    u, v, tile = np.broadcast_arrays(np.asarray(u, np.float32), np.asarray(v, np.float32), np.asarray(tile, np.float32))
    return np.stack([u, v, tile], -1)


# direction: normal, origin corner, U axis, V axis (V points down the texture), face shade
DIRS = [((1, 0, 0), (1, 1, 1), (0, 0, -1), (0, -1, 0), 0.6), ((-1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0), 0.6),
        ((0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), 1.0), ((0, -1, 0), (0, 0, 1), (1, 0, 0), (0, 0, -1), 0.5),
        ((0, 0, 1), (0, 1, 1), (1, 0, 0), (0, -1, 0), 0.8), ((0, 0, -1), (1, 1, 0), (-1, 0, 0), (0, -1, 0), 0.8)]
AO = np.array([0.5, 0.66, 0.82, 1.0], np.float32)


def mesh(sky, blk):
    geo = Geo()
    op = np.isin(W, list(FULL))
    lumS = lm(sky); lumB = lm(blk)
    padO = np.pad(op, 1, constant_values=False)
    padS = np.pad(lumS, 1, mode='edge'); padB = np.pad(lumB, 1, mode='edge')
    ids = W
    tl = np.zeros((256, 3), np.int32)
    for b, ts in CUBE_TILES.items():
        tl[b] = [T[n] for n in ts]
    cubes = np.isin(ids, list(FULL | CUTOUT_CUBE))
    for di, (n, o, U, V, shade) in enumerate(DIRS):
        nb = shift(ids, n, AIR)
        nbop = shift(op, n, False)
        if di == 2 or di == 3:
            pass
        same_cut = np.isin(ids, [GLASS]) & (nb == GLASS)
        vis = cubes & ~nbop & ~same_cut
        xs, ys, zs = np.nonzero(vis)
        if not len(xs):
            continue
        b = ids[xs, ys, zs]
        tsel = 0 if di == 2 else (2 if di == 3 else 1)
        tile = tl[b, tsel]
        for i in range(len(xs)):
            key = (int(xs[i]), int(ys[i]), int(zs[i]))
            if key in FRONT and FRONT[key][0] == di:
                tile[i] = T[FRONT[key][1]]
        # smooth light + AO from the cells on the face's open side
        nx_, ny_, nz_ = n
        cx, cy, cz = xs + 1 + nx_, ys + 1 + ny_, zs + 1 + nz_
        U_ = np.array(U); V_ = np.array(V)
        P = np.zeros((len(xs), 4, 3), np.float32); UVs = np.zeros((len(xs), 4, 3), np.float32); C = np.zeros((len(xs), 4, 3), np.float32)
        base = np.stack([xs, ys, zs], 1) + np.array(o)
        corner_uv = [(0, 0), (1, 0), (1, 1), (0, 1)]
        for k, (cu, cv) in enumerate(corner_uv):
            P[:, k] = base + U_ * cu + V_ * cv
            su = np.sign(U_) * (1 if cu else -1); sv = np.sign(V_) * (1 if cv else -1)
            # the four cells round this vertex on the open side
            a = (cx, cy, cz)
            s1 = (cx + su[0], cy + su[1], cz + su[2]); s2 = (cx + sv[0], cy + sv[1], cz + sv[2])
            cc = (cx + su[0] + sv[0], cy + su[1] + sv[1], cz + su[2] + sv[2])
            o1 = padO[s1]; o2 = padO[s2]; oc = padO[cc] & ~(o1 & o2)
            occ = np.where(o1 & o2, 0, 3 - (o1.astype(int) + o2 + oc))
            w0 = np.ones(len(xs)); w1 = (~o1).astype(float); w2 = (~o2).astype(float); w3 = (~(padO[cc] | (o1 & o2))).astype(float)
            ws = w0 + w1 + w2 + w3
            Ls = (padS[a] * w0 + padS[s1] * w1 + padS[s2] * w2 + padS[cc] * w3) / ws
            Lb = (padB[a] * w0 + padB[s1] * w1 + padB[s2] * w2 + padB[cc] * w3) / ws
            col = np.stack([Ls, Lb, np.zeros_like(Ls)], 1) * (shade * AO[occ])[:, None]
            C[:, k] = col
            UVs[:, k] = tile_uv(tile, cu, cv)
        grp = np.where(np.isin(b, list(CUTOUT_CUBE)), 1, 0)
        for gi, gname in ((0, 'opaque'), (1, 'cutout')):
            m = grp == gi
            geo.quads(gname, P[m], UVs[m], C[m])
    # ---- plants: crossed quads, lit by their own cell
    xs, ys, zs = np.nonzero(np.isin(ids, list(PLANTS)))
    rng = np.random.default_rng(3)
    for x, y, z in zip(xs, ys, zs):
        t = T[PLANTS[int(ids[x, y, z])]]
        c = np.array([lumS[x, y, z], lumB[x, y, z], 0.0]) * 0.95
        ox, oz = (rng.random(2) - 0.5) * 0.35 if ids[x, y, z] == TALLGRASS else (0, 0)
        hgt = 1.0
        for (ax, az, bx, bz) in ((0.15, 0.15, 0.85, 0.85), (0.15, 0.85, 0.85, 0.15)):
            for flip in (False, True):
                p = [(x + ax + ox, y + hgt, z + az + oz), (x + bx + ox, y + hgt, z + bz + oz), (x + bx + ox, y, z + bz + oz), (x + ax + ox, y, z + az + oz)]
                uv = [tile_uv(t, 0, 0), tile_uv(t, 1, 0), tile_uv(t, 1, 1), tile_uv(t, 0, 1)]
                if flip:
                    p = p[::-1]; uv = uv[::-1]
                geo.quad('cutout', p, uv, [c] * 4)
    # ---- water surface (top faces only, 14/16 high)
    xs, ys, zs = np.nonzero((ids == WATER) & (shift(ids, (0, 1, 0), AIR) == AIR))
    tw = T['water']
    for x, y, z in zip(xs, ys, zs):
        c = np.array([lumS[x, y + 1, z], lumB[x, y + 1, z], 0.0])
        yy = y + 0.875
        geo.quad('water', [(x, yy, z), (x, yy, z + 1), (x + 1, yy, z + 1), (x + 1, yy, z)],
                 [tile_uv(tw, 0, 0), tile_uv(tw, 0, 1), tile_uv(tw, 1, 1), tile_uv(tw, 1, 0)], [c] * 4)
    # ---- the waterfall: a column of falling water, its sides streaming (the shader scrolls the flow tile down)
    tf = T['water_flow']
    for x, y, z in zip(*np.nonzero(ids == FALLW)):
        for (dx, dz) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb_ = ids[x + dx, y, z + dz]
            if nb_ in (FALLW, WATER) or nb_ in FULL:
                continue
            c = np.array([lumS[x + dx, y, z + dz], lumB[x + dx, y, z + dz], 0.0]) * (0.8 if dz else 0.62)
            if dx == 1:
                p = [(x + 1, y + 1, z + 1), (x + 1, y + 1, z), (x + 1, y, z), (x + 1, y, z + 1)]
            elif dx == -1:
                p = [(x, y + 1, z), (x, y + 1, z + 1), (x, y, z + 1), (x, y, z)]
            elif dz == 1:
                p = [(x, y + 1, z + 1), (x + 1, y + 1, z + 1), (x + 1, y, z + 1), (x, y, z + 1)]
            else:
                p = [(x + 1, y + 1, z), (x, y + 1, z), (x, y, z), (x + 1, y, z)]
            geo.quad('water', p, [tile_uv(tf, 0, 0), tile_uv(tf, 1, 0), tile_uv(tf, 1, 1), tile_uv(tf, 0, 1)], [c] * 4)
    # ---- special shapes: beds, torches, vines, cobwebs
    def lit(x, y, z):
        return np.array([lumS[x, y, z], lumB[x, y, z], 0.0])

    def box(grp, x0, y0, z0, x1, y1, z1, tiles, c, uvbox=None):
        """axis box with per-face tiles (top, side, bottom) and the classic face shades; uv from the box's own extent"""
        for di, (n, o, U, V, shade) in enumerate(DIRS):
            t = tiles[0] if di == 2 else (tiles[2] if di == 3 else tiles[1])
            if t is None:
                continue
            lo = np.array([x0, y0, z0], float); hi = np.array([x1, y1, z1], float)
            corners = []
            for (cu, cv) in ((0, 0), (1, 0), (1, 1), (0, 1)):
                pt = lo + (hi - lo) * np.clip(np.array(o) + np.array(U) * cu + np.array(V) * cv, 0, 1)
                corners.append(pt)
            # texture coordinates from the box's position in its block (so a 9/16 bed side shows the right rows)
            uvs = []
            for pt in corners:
                f = pt - np.floor(lo + 1e-6)
                if di in (0, 1):
                    u, v = (1 - f[2]) if di == 0 else f[2], 1 - f[1]
                elif di in (4, 5):
                    u, v = f[0] if di == 4 else 1 - f[0], 1 - f[1]
                else:
                    u, v = f[0], f[2] if di == 2 else 1 - f[2]
                uvs.append(tile_uv(t, np.clip(u, 0, 1), np.clip(v, 0, 1)))
            geo.quad(grp, corners, uvs, [c * shade] * 4)

    for x, y, z in zip(*np.nonzero(np.isin(ids, [BED_HEAD, BED_FOOT]))):
        head = ids[x, y, z] == BED_HEAD
        c = lit(x, y, z)
        tb = T['bed_head_top' if head else 'bed_foot_top']; yt = y + 9 / 16
        geo.quad('cutout', [(x, yt, z), (x, yt, z + 1), (x + 1, yt, z + 1), (x + 1, yt, z)],
                 [tile_uv(tb, 0, 1), tile_uv(tb, 1, 1), tile_uv(tb, 1, 0), tile_uv(tb, 0, 0)], [c] * 4)
        box('cutout', x, y, z, x + 1, y + 9 / 16, z + 1, (None, T['bed_side_foot'], T['oak_planks']), c)
    for x, y, z in zip(*np.nonzero(ids == FENCE)):
        c = lit(x, y, z); tp = T['oak_planks']
        box('opaque', x + 6 / 16, y, z + 6 / 16, x + 10 / 16, y + 1, z + 10 / 16, (tp, tp, tp), c)
        for (dx, dz) in ((1, 0), (0, 1)):
            if ids[x + dx, y, z + dz] == FENCE:
                for yy in (6 / 16, 12 / 16):
                    if dx:
                        box('opaque', x + 10 / 16, y + yy, z + 7 / 16, x + 1 + 6 / 16, y + yy + 3 / 16, z + 9 / 16, (tp, tp, tp), c)
                    else:
                        box('opaque', x + 7 / 16, y + yy, z + 10 / 16, x + 9 / 16, y + yy + 3 / 16, z + 1 + 6 / 16, (tp, tp, tp), c)
    tt = T['torch']
    for (x, y, z, f) in EXTRA['torches']:
        c = np.array([1.0, 1.0, 0.0])
        # a thin stick showing the middle columns of the torch tile; wall torches lean out from the wall
        lean = {'n': (0, 0, -1), 's': (0, 0, 1), 'e': (1, 0, 0), 'w': (-1, 0, 0), 'u': (0, 0, 0)}[f]
        bx, bz = x + 0.5 + lean[0] * 0.3, z + 0.5 + lean[2] * 0.3   # wall torches sit against their wall
        base_y = y + (0.2 if f != 'u' else 0.0)
        tilt = 0.0 if f == 'u' else 0.38
        def pt(px, py, pz):
            # tilt the stick away from the wall
            ty = py - base_y
            dxl, dzl = (-lean[0], -lean[2]) if f != 'u' else (0, 0)
            return (px + dxl * ty * tilt, base_y + ty * math.cos(tilt), pz + dzl * ty * tilt)
        hw = 1 / 16
        for (ax, az, cx2, cz2) in ((-hw, -hw, hw, -hw), (hw, -hw, hw, hw), (hw, hw, -hw, hw), (-hw, hw, -hw, -hw)):
            p = [pt(bx + ax, base_y + 10 / 16, bz + az), pt(bx + cx2, base_y + 10 / 16, bz + cz2), pt(bx + cx2, base_y, bz + cz2), pt(bx + ax, base_y, bz + az)]
            uv = [tile_uv(tt, 7 / 16, 6 / 16), tile_uv(tt, 9 / 16, 6 / 16), tile_uv(tt, 9 / 16, 1), tile_uv(tt, 7 / 16, 1)]
            geo.quad('cutout', p, uv, [c] * 4)
        p = [pt(bx - hw, base_y + 10 / 16, bz - hw), pt(bx + hw, base_y + 10 / 16, bz - hw), pt(bx + hw, base_y + 10 / 16, bz + hw), pt(bx - hw, base_y + 10 / 16, bz + hw)]
        uv = [tile_uv(tt, 7 / 16, 4 / 16), tile_uv(tt, 9 / 16, 4 / 16), tile_uv(tt, 9 / 16, 6 / 16), tile_uv(tt, 7 / 16, 6 / 16)]
        geo.quad('cutout', p[::-1], uv[::-1], [c] * 4)
        geo.quad('cutout', p, uv, [c] * 4)
    tv = T['vine']
    for (x, y, z, f) in EXTRA['vines']:
        if not (0 <= x < NX and 0 <= z < NZ) or ids[x, y, z] not in (AIR, TALLGRASS):
            continue
        c = lit(x, y, z) * 0.9
        e = 0.94
        if f == 's':     # hanging on the face of the block to the south (+z)
            p = [(x, y + 1, z + e), (x + 1, y + 1, z + e), (x + 1, y, z + e), (x, y, z + e)]
        elif f == 'n':
            p = [(x + 1, y + 1, z + 1 - e), (x, y + 1, z + 1 - e), (x, y, z + 1 - e), (x + 1, y, z + 1 - e)]
        elif f == 'e':
            p = [(x + e, y + 1, z + 1), (x + e, y + 1, z), (x + e, y, z), (x + e, y, z + 1)]
        else:
            p = [(x + 1 - e, y + 1, z), (x + 1 - e, y + 1, z + 1), (x + 1 - e, y, z + 1), (x + 1 - e, y, z)]
        uv = [tile_uv(tv, 0, 0), tile_uv(tv, 1, 0), tile_uv(tv, 1, 1), tile_uv(tv, 0, 1)]
        geo.quad('cutout', p, uv, [c] * 4); geo.quad('cutout', p[::-1], uv[::-1], [c] * 4)
    tw_ = T['cobweb']
    for (x, y, z) in EXTRA['webs']:
        c = lit(x, y, z)
        for (ax, az, bx2, bz2) in ((0, 0, 1, 1), (0, 1, 1, 0)):
            p = [(x + ax, y + 1, z + az), (x + bx2, y + 1, z + bz2), (x + bx2, y, z + bz2), (x + ax, y, z + az)]
            uv = [tile_uv(tw_, 0, 0), tile_uv(tw_, 1, 0), tile_uv(tw_, 1, 1), tile_uv(tw_, 0, 1)]
            geo.quad('cutout', p, uv, [c] * 4); geo.quad('cutout', p[::-1], uv[::-1], [c] * 4)
    return geo


def save(geo, sky, blk):
    meta = dict(groups={}, spawn=[SPAWN[0] + 0.5, 65, SPAWN[1] + 0.5])
    buf = bytearray()
    for k, g in geo.g.items():
        if not g['pos']:
            continue
        P = np.concatenate([a.reshape(-1, 3) for a in g['pos']]); UV = np.concatenate([a.reshape(-1, 3) for a in g['uv']])
        C = np.clip(np.concatenate([a.reshape(-1, 3) for a in g['col']]) * 255, 0, 255).astype(np.uint8)
        n = len(P) // 4
        idx = (np.array([0, 1, 2, 0, 2, 3])[None, :] + (np.arange(n) * 4)[:, None]).reshape(-1).astype(np.uint32)
        part = {}
        for name, arr in (('pos', P.astype(np.float32)), ('uv', UV.astype(np.float32)), ('idx', idx), ('col', C)):
            part[name] = [len(buf), int(arr.size)]; buf += arr.tobytes(); buf += b'\0' * ((-len(buf)) % 4)
        meta['groups'][k] = part
        print(k, n, 'quads')
    open(os.path.join(WEB, 'world2.bin'), 'wb').write(bytes(buf))
    # the light round the house and the path, for the wolf and the arm: (x0, y0, z0, nx, ny, nz) + sky, block levels
    x0, y0, z0, x1, y1, z1 = 172, 60, 172, 216, 86, 210
    meta['light'] = dict(origin=[x0, y0, z0], size=[x1 - x0, y1 - y0, z1 - z0],
                         sky=sky[x0:x1, y0:y1, z0:z1].astype(int).ravel().tolist(), blk=blk[x0:x1, y0:y1, z0:z1].astype(int).ravel().tolist())
    meta['sky_col'] = SKY_COL.tolist(); meta['torch_col'] = TORCH_COL.tolist()
    meta['door'] = dict(x=HOUSE[0], y=65, z=191)
    meta['sign'] = dict(x=190, y=65, z=189)
    meta['painting'] = dict(x0=198, x1=200, y0=66, y1=67, z=189.02)
    meta['torch_pos'] = [[x + 0.5, y + 0.7, z + 0.5] for (x, y, z, f) in EXTRA['torches']]
    json.dump(meta, open(os.path.join(WEB, 'world2.json'), 'w'))
    np.save(os.path.join(HERE, 'world2_blocks.npy'), W)


if __name__ == '__main__':
    t0 = time.time()
    rng = np.random.default_rng(10)
    H = terrain(); print('terrain %.0fs' % (time.time() - t0), flush=True)
    forest(rng)
    house(rng); path(rng); landmarks(rng)
    fall = waterfall()
    plants(rng, H)
    for x in range(176, HOUSE[0] + 1):                       # where he walks stays clear
        for y in (65, 66):
            if W[x, y, 191] in (TALLGRASS, ROSE, DANDELION, SAPLING):
                W[x, y, 191] = AIR
    W[HOUSE[0], 65:67, 191] = AIR
    print('built %.0fs' % (time.time() - t0), flush=True)
    sky, blk = light(); print('light %.0fs' % (time.time() - t0), flush=True)
    geo = mesh(sky, blk); print('mesh %.0fs' % (time.time() - t0), flush=True)
    save(geo, sky, blk); print('saved %.0fs' % (time.time() - t0))
