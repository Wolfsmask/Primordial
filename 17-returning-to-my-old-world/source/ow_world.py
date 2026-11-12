"""The old world: an old-generation landscape round a spawn point, ten years overgrown.  The first house (cobble and
planks), the path to it, the sign, the first-night dirt pillar, a pond with sugar cane, an old wheat farm and an
unlit portal frame.  Block light and sky light are flood-filled like the classic game and baked into the vertices with
smooth lighting.  -> web/world.bin + web/world.json (mesh groups, entities, a light volume for the wolf and the arm)"""
import json, math, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
sys.path.insert(0, os.path.join(HERE, '..', 'bleed3'))
from lib import fbm
import tex_ow as TX

NX, NY, NZ = 256, 112, 256
SPAWN = (128, 128)
SEA = 62
# ---- block ids
AIR, GRASS, DIRT, STONE, COBBLE, MOSSY, PLANKS, LOG, LEAVES, GLASS, SAND, GRAVEL, WATER = range(13)
TALLGRASS, POPPY, DANDELION, CRAFT, FURNACE, CHEST, BED_HEAD, BED_FOOT, OBSIDIAN, WHEAT, BOOKS, COAL = range(13, 25)
BIRCH, BIRCH_LEAVES, CANE, RED_MUSH, BROWN_MUSH, FARMLAND, WOOL, FENCE = range(25, 33)
T = TX.IDX
FULL = {GRASS, DIRT, STONE, COBBLE, MOSSY, PLANKS, LOG, SAND, GRAVEL, CRAFT, FURNACE, CHEST, OBSIDIAN, BOOKS, COAL, BIRCH, FARMLAND, WOOL}
CUTOUT_CUBE = {LEAVES, BIRCH_LEAVES, GLASS}
PLANTS = {TALLGRASS: 'tall_grass', POPPY: 'poppy', DANDELION: 'dandelion', WHEAT: 'wheat', CANE: 'sugar_cane',
          RED_MUSH: 'red_mushroom', BROWN_MUSH: 'brown_mushroom'}
# top, side, bottom tiles
CUBE_TILES = {GRASS: ('grass_top', 'grass_side', 'dirt'), DIRT: ('dirt',) * 3, STONE: ('stone',) * 3, COBBLE: ('cobblestone',) * 3,
              MOSSY: ('mossy_cobblestone',) * 3, PLANKS: ('oak_planks',) * 3, LOG: ('log_top', 'oak_log', 'log_top'),
              LEAVES: ('oak_leaves',) * 3, GLASS: ('glass',) * 3, SAND: ('sand',) * 3, GRAVEL: ('gravel',) * 3,
              CRAFT: ('crafting_top', 'crafting_side', 'oak_planks'), FURNACE: ('furnace_top', 'furnace_side', 'furnace_top'),
              CHEST: ('chest_top', 'chest_side', 'chest_top'), OBSIDIAN: ('obsidian',) * 3, BOOKS: ('oak_planks', 'bookshelf', 'oak_planks'),
              COAL: ('coal_ore',) * 3, BIRCH: ('log_top', 'birch_log', 'log_top'), BIRCH_LEAVES: ('birch_leaves',) * 3,
              FARMLAND: ('farmland', 'dirt', 'dirt'), WOOL: ('white_wool',) * 3}
FRONT = {}          # (x, y, z) -> (dir index, tile) for furnace / chest / crafting fronts

W = np.zeros((NX, NY, NZ), np.uint8)
EXTRA = dict(torches=[], vines=[], webs=[], lilies=[])


def nz(x, z, s, seed, o=4):
    return fbm(x, 0.0, z, s, seed, o)


def terrain():
    xs, zs = np.arange(NX), np.arange(NZ)
    X, Z = np.meshgrid(xs, zs, indexing='ij')
    dx, dz = X - SPAWN[0], Z - SPAWN[1]
    d = np.hypot(dx, dz)
    h = 64 + 3.0 * (nz(X, Z, 48, 1) - 0.5) + 2.0 * (nz(X, Z, 16, 2) - 0.5)
    # hills: rolling all round, a big old-generation hill to the north-west with a stone face, far ridges
    hill = np.clip(nz(X, Z, 70, 3) - 0.45, 0, None) * 60 * np.clip((d - 40) / 40, 0, 1)
    big = 34 * np.exp(-(((X - 70) / 34) ** 2 + ((Z - 66) / 28) ** 2)) + 22 * np.exp(-(((X - 196) / 30) ** 2 + ((Z - 52) / 26) ** 2))
    far = np.clip((d - 95) / 25, 0, 1) * 26 * nz(X, Z, 40, 4) ** 1.5
    h = h + hill + big + far
    # the spawn plain and the house plot are flat-ish (the game had put you on a meadow)
    flat = np.exp(-((dx / 22) ** 2 + ((dz - 6) / 26) ** 2))
    h = h * (1 - flat) + 64.2 * flat
    # the pond west of spawn
    pond = np.hypot((X - 112) / 7.5, (Z - 121) / 6.0) + 0.25 * (nz(X, Z, 6, 5) - 0.5)
    h = np.where(pond < 1.0, np.minimum(h, SEA - 1 - 2 * (1 - pond)), h)
    h = np.where((pond >= 1.0) & (pond < 1.35), np.minimum(h, SEA + 1.0), h)
    H = np.floor(h).astype(int)
    for x in range(NX):
        for z in range(NZ):
            top = H[x, z]
            W[x, :top - 3, z] = STONE
            W[x, top - 3:top, z] = DIRT
            W[x, top, z] = GRASS
    # beaches and the pond water
    beach = (pond < 1.55) & (H <= SEA + 1)
    for x, z in zip(*np.nonzero(beach)):
        W[x, H[x, z] - 2:H[x, z] + 1, z] = SAND
    for x, z in zip(*np.nonzero(pond < 1.0)):
        W[x, H[x, z] + 1:SEA + 1, z] = WATER
    # the stone face of the north-west hill: exposed rock, coal, and an old-generation overhang
    for x in range(40, 104):
        for z in range(30, 100):
            top = H[x, z]
            if top > 74:
                cut = nz(x, z, 9, 6, 2) > 0.52
                if cut:
                    W[x, top - 1:top + 1, z] = STONE
                for y in range(60, top):
                    if W[x, y, z] == STONE and nz(x * 1.0 + y * 0.7, z, 5, 7, 2) > 0.74:
                        W[x, y, z] = COAL
    for x in range(78, 92):
        for z in range(70, 80):
            if (x - 85) ** 2 / 40 + (z - 75) ** 2 / 20 < 1:
                W[x, 84:87, z] = STONE; W[x, 87, z] = GRASS
                W[x, 78:84, z] = AIR if H[x, z] >= 84 else W[x, 78:84, z]
    return H


def surface(x, z):
    col = W[x, :, z]
    ys = np.nonzero(np.isin(col, list(FULL | {WATER})))[0]
    return int(ys.max()) if len(ys) else 0


def tree(x, z, kind=LOG, leaves=LEAVES, h=None, rng=None, big=False):
    y0 = surface(x, z) + 1
    if W[x, y0 - 1, z] not in (GRASS, DIRT):
        return
    h = h or int(rng.integers(4, 7))
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
                if 0 <= x + ax < NX and 0 <= z + az < NZ and W[x + ax, y, z + az] == AIR:
                    W[x + ax, y, z + az] = leaves
    if big:
        for ax, az in ((1, 0), (0, 1), (-1, 0)):
            W[x + ax * 2, top - 2, z + az * 2] = kind


def plants(H, rng):
    """ten years of grass: tall grass nearly everywhere on the meadow, flowers, mushrooms in the shade"""
    for x in range(1, NX - 1):
        for z in range(1, NZ - 1):
            y = surface(x, z)
            if W[x, y, z] != GRASS or W[x, y + 1, z] != AIR:
                continue
            d = math.hypot(x - SPAWN[0], z - SPAWN[1])
            dens = 0.55 if d < 40 else 0.32
            r = rng.random()
            if r < dens:
                W[x, y + 1, z] = TALLGRASS
            elif r < dens + 0.025:
                W[x, y + 1, z] = POPPY if rng.random() < 0.5 else DANDELION


def house(rng):
    """the first house: 7 x 7, cobble footing, plank walls, log corners, flat roof"""
    x0, x1, z0, z1 = 125, 131, 138, 144
    for x in range(x0 - 1, x1 + 2):
        for z in range(z0 - 1, z1 + 2):
            W[x, 64, z] = GRASS if (x < x0 or x > x1 or z < z0 or z > z1) else COBBLE
            W[x, 65:76, z] = AIR
            W[x, 61:64, z] = DIRT
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
            if not corner:
                W[x, 68, z] = PLANKS if not edge else W[x, 68, z]
    for x in range(x0 + 1, x1):
        for z in range(z0 + 1, z1):
            W[x, 69, z] = PLANKS if (x in (x0 + 1, x1 - 1) or z in (z0 + 1, z1 - 1)) else AIR
    W[x0 + 2:x1 - 1, 69, z0 + 2:z1 - 1] = PLANKS
    # door gap, windows (the east one broken long ago)
    W[128, 65:67, z0] = AIR
    for (x, z) in ((126, z0), (130, z0), (x0, 141), (x1, 141), (128, z1)):
        W[x, 66, z] = GLASS
    W[x1, 66, 141] = AIR
    EXTRA['webs'].append((x1, 66, 141))
    # inside
    W[126, 65, 143] = BED_HEAD; W[126, 65, 142] = BED_FOOT
    W[127, 65, 143] = CHEST; FRONT[(127, 65, 143)] = (5, 'chest_front')
    W[130, 65, 143] = CRAFT; FRONT[(130, 65, 143)] = (5, 'crafting_front')
    W[130, 65, 142] = FURNACE; FRONT[(130, 65, 142)] = (1, 'furnace_front')
    W[129, 65, 143] = BOOKS
    EXTRA['torches'] += [(126, 66, 139, 'n'), (130, 66, 139, 'n'), (128, 67, 143, 's')]
    EXTRA['webs'] += [(126, 67, 139), (130, 67, 143), (126, 67, 143)]
    # outside: a torch beside the door, vines hanging off the walls and roof edge
    EXTRA['torches'] += [(127, 66, 137, 's')]
    for x in range(x0, x1 + 1):
        for y in range(64, 69):
            if rng.random() < 0.42 and W[x, y, z0 - 1] == AIR and not (x == 128 and y < 67) and not (x == 127 and y == 66):
                EXTRA['vines'].append((x, y, z0 - 1, 's'))          # on the north wall, seen from the path
    for z in range(z0, z1 + 1):
        for y in range(64, 69):
            if rng.random() < 0.5:
                EXTRA['vines'].append((x0 - 1, y, z, 'e'))
            if rng.random() < 0.3:
                EXTRA['vines'].append((x1 + 1, y, z, 'w'))
    # the oak that grew at the corner and leaned its crown over the roof
    tree(123, 136, h=6, rng=rng, big=True)
    for x in range(123, 128):
        for z in range(137, 142):
            if rng.random() < 0.7 and W[x, 70, z] == AIR:
                W[x, 70, z] = LEAVES
            if rng.random() < 0.35 and W[x, 71, z] == AIR:
                W[x, 71, z] = LEAVES


def path(rng):
    """the old cobble path from spawn to the door, half reclaimed by grass"""
    for z in range(130, 138):
        for x in (127, 128, 129):
            if x != 128 and rng.random() < 0.35:
                continue
            r = rng.random()
            W[x, 64, z] = GRASS if r < 0.22 else (MOSSY if r < 0.55 else (GRAVEL if r < 0.68 else COBBLE))
            if W[x, 64, z] != GRASS:
                W[x, 65, z] = AIR if (x == 128 or rng.random() < 0.6) else W[x, 65, z]
    W[128, 64, 128:130] = GRASS


def landmarks(rng):
    # spawn torch and the first-night dirt pillar (still standing, torch on top)
    py0 = surface(116, 127) + 1
    for y in range(py0, py0 + 7):
        W[116, y, 127] = DIRT
    EXTRA['torches'].append((116, py0 + 7, 127, 'u'))
    # sign on a post beside the path: entity in the renderer
    # the old wheat farm east of the house: fence half gone, water in the middle, wheat run wild among the grass
    for x in range(134, 141):
        for z in range(137, 144):
            border = x in (134, 140) or z in (137, 143)
            if border:
                if rng.random() < 0.72:
                    W[x, 65, z] = FENCE
                W[x, 64, z] = GRASS
            else:
                W[x, 64, z] = FARMLAND if rng.random() < 0.55 else GRASS
                W[x, 65, z] = WHEAT if rng.random() < 0.6 else (TALLGRASS if rng.random() < 0.6 else AIR)
    W[137, 64, 140] = WATER; W[137, 65, 140] = AIR
    # the portal frame that was never lit, out to the north-east, vines on it
    px, pz = 99, 133                       # facing spawn across the meadow, out to the west
    base = surface(px, pz) + 1
    for k in range(-1, 5):
        for y in range(base - 1, base + 7):
            if W[px, y, pz + k] in (TALLGRASS, POPPY, DANDELION):
                W[px, y, pz + k] = AIR
    for k in range(4):
        W[px, base - 1, pz + k] = STONE
        for y in range(5):
            if (k in (0, 3) or y in (0, 4)) and not ((k, y) in ((0, 0), (3, 4))):
                W[px, base + y, pz + k] = OBSIDIAN
    for y in range(base, base + 5):
        EXTRA['vines'].append((px + 1, y, pz + 3, 'w'))
        if y < base + 3:
            EXTRA['vines'].append((px + 1, y, pz, 'w'))
    # the pond's sugar cane and lily pads
    for (x, z) in ((105, 119), (106, 125), (118, 117), (119, 123), (110, 113)):
        y = surface(x, z)
        if W[x, y, z] in (GRASS, SAND):
            for k in range(1, 4):
                W[x, y + k, z] = CANE
    for (x, z) in ((110, 120), (114, 122), (112, 118), (115, 119)):
        EXTRA['lilies'].append((x, SEA + 1, z))
    # mushrooms in the shade of the house and trees
    for (x, z) in ((124, 145), (132, 139), (122, 138)):
        y = surface(x, z)
        if W[x, y + 1, z] in (AIR, TALLGRASS):
            W[x, y + 1, z] = RED_MUSH if x % 2 else BROWN_MUSH


def forest(H, rng):
    for k in range(260):
        x, z = int(rng.integers(6, NX - 6)), int(rng.integers(6, NZ - 6))
        d = math.hypot(x - SPAWN[0], z - SPAWN[1])
        if d < 14 or (118 < x < 145 and 126 < z < 150) or (100 < x < 124 and 108 < z < 132) or (94 < x < 104 and 129 < z < 139):
            continue
        dens = nz(x, z, 30, 9, 2)
        if rng.random() > dens * 1.25:
            continue
        if rng.random() < 0.25:
            tree(x, z, BIRCH, BIRCH_LEAVES, h=int(rng.integers(5, 7)), rng=rng)
        else:
            tree(x, z, rng=rng, big=rng.random() < 0.15)
    for (x, z) in ((116, 131), (141, 128), (147, 146), (109, 140), (138, 115), (100, 128)):
        tree(x, z, rng=rng, big=(x == 141))


# ------------------------------------------------------------------ light
def opacity():
    op = np.zeros(W.shape, np.int8)
    op[np.isin(W, list(FULL))] = 15
    op[np.isin(W, [LEAVES, BIRCH_LEAVES])] = 1
    op[W == WATER] = 2
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


def lm(level):
    """the classic light curve: level 0..15 -> brightness"""
    f = 1.0 - np.asarray(level, np.float32) / 15.0
    return (1 - f) / (f * 3 + 1) * 0.95 + 0.05


SKY_COL = np.array([1.0, 0.93, 0.84], np.float32) * 0.92       # late afternoon, a little warm
TORCH_COL = np.array([1.0, 0.82, 0.58], np.float32)


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
            col = np.maximum(SKY_COL[None] * Ls[:, None], TORCH_COL[None] * Lb[:, None]) * (shade * AO[occ])[:, None]
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
        c = np.maximum(SKY_COL * lumS[x, y, z], TORCH_COL * lumB[x, y, z]) * 0.95
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
        c = np.maximum(SKY_COL * lumS[x, y + 1, z], TORCH_COL * lumB[x, y + 1, z])
        yy = y + 0.875
        geo.quad('water', [(x, yy, z), (x, yy, z + 1), (x + 1, yy, z + 1), (x + 1, yy, z)],
                 [tile_uv(tw, 0, 0), tile_uv(tw, 0, 1), tile_uv(tw, 1, 1), tile_uv(tw, 1, 0)], [c] * 4)
    # ---- special shapes: beds, fences, torches, vines, cobwebs, lily pads
    def lit(x, y, z):
        return np.maximum(SKY_COL * lumS[x, y, z], TORCH_COL * lumB[x, y, z])

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
        box('cutout', x, y + 3 / 16, z, x + 1, y + 9 / 16, z + 1, (T['bed_head_top' if head else 'bed_foot_top'], None, T['oak_planks']), c)
        # sides: drawn separately so the head end shows the pillow side
        for di in (0, 1):
            pass
        box('cutout', x, y, z, x + 1, y + 9 / 16, z + 1, (None, T['bed_side_head' if head else 'bed_side_foot'], None), c)
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
        c = np.array([1.0, 1.0, 1.0])
        # a thin stick showing the middle columns of the torch tile; wall torches lean out from the wall
        lean = {'n': (0, 0, -1), 's': (0, 0, 1), 'e': (1, 0, 0), 'w': (-1, 0, 0), 'u': (0, 0, 0)}[f]
        bx, bz = x + 0.5 - lean[0] * 0.3, z + 0.5 - lean[2] * 0.3   # wall torches sit against the wall behind them
        if f == 'n':
            bz = z + 0.2
        if f == 's':
            bz = z + 0.8
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
    tl_ = T['lily_pad']
    for (x, y, z) in EXTRA['lilies']:
        c = lit(x, y, z)
        yy = y - 0.1
        geo.quad('cutout', [(x, yy, z), (x, yy, z + 1), (x + 1, yy, z + 1), (x + 1, yy, z)],
                 [tile_uv(tl_, 0, 0), tile_uv(tl_, 0, 1), tile_uv(tl_, 1, 1), tile_uv(tl_, 1, 0)], [c] * 4)
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
    open(os.path.join(WEB, 'world.bin'), 'wb').write(bytes(buf))
    # the light round the house and the path, for the wolf and the arm: (x0, y0, z0, nx, ny, nz) + sky, block levels
    x0, y0, z0, x1, y1, z1 = 112, 60, 116, 150, 80, 150
    meta['light'] = dict(origin=[x0, y0, z0], size=[x1 - x0, y1 - y0, z1 - z0],
                         sky=sky[x0:x1, y0:y1, z0:z1].astype(int).ravel().tolist(), blk=blk[x0:x1, y0:y1, z0:z1].astype(int).ravel().tolist())
    meta['sky_col'] = SKY_COL.tolist(); meta['torch_col'] = TORCH_COL.tolist()
    meta['door'] = dict(x=128, y=65, z=138)
    meta['sign'] = dict(x=126, y=65, z=135)
    meta['torch_pos'] = [[x + 0.5, y + 0.7, z + 0.5] for (x, y, z, f) in EXTRA['torches']]
    json.dump(meta, open(os.path.join(WEB, 'world.json'), 'w'))
    np.save(os.path.join(HERE, 'world_blocks.npy'), W)


if __name__ == '__main__':
    t0 = time.time()
    rng = np.random.default_rng(10)
    H = terrain(); print('terrain %.0fs' % (time.time() - t0), flush=True)
    forest(H, rng)
    house(rng); path(rng); landmarks(rng)
    plants(H, rng)
    # the path and the doorway stay clear of grass tufts where he walks
    for z in range(128, 138):
        if W[128, 65, z] in (TALLGRASS, POPPY, DANDELION):
            W[128, 65, z] = AIR if z > 131 else W[128, 65, z]
    W[128, 65, 138] = AIR; W[128, 66, 138] = AIR
    print('built %.0fs' % (time.time() - t0), flush=True)
    sky, blk = light(); print('light %.0fs' % (time.time() - t0), flush=True)
    geo = mesh(sky, blk); print('mesh %.0fs' % (time.time() - t0), flush=True)
    save(geo, sky, blk); print('saved %.0fs' % (time.time() - t0))
