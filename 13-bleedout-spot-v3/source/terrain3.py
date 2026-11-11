"""Detailed blocks around the ledge (v3): the same ledge, trail and cherry tree; the near pillars get groves on top.
Writes world3.npz (vol, origin, spot)."""
import math, time
import numpy as np
import hf3 as hf
from hf3 import ss, nz

t0 = time.time()
X0, Z0 = -400, -280
NX, NY, NZ = 560, 336, 560
SPOT = np.array([10.0, float(hf.LEDGE_Y), 0.0])
B = dict(AIR=0, GRASS=1, DIRT=2, STONE=3, ANDESITE=4, GRANITE=5, TUFF=6, GRAVEL=7, SAND=8, SNOW=9, WATER=10,
         OAK_LOG=11, OAK_LEAVES=12, BIRCH_LOG=13, BIRCH_LEAVES=14, SPRUCE_LOG=15, SPRUCE_LEAVES=16, CHERRY_LOG=17,
         CHERRY_LEAVES=18, COARSE_DIRT=19, MOSS=20, CALCITE=21, PODZOL=22, MOSSY_STONE=23, CLAY=24,
         SHORT_GRASS=30, FERN=31, POPPY=32, DANDELION=33, CORNFLOWER=34, ALLIUM=35, OXEYE=36, PINK_PETALS=37)
G = type('G', (), B)
xs_w = np.arange(NX) + X0; zs_w = np.arange(NZ) + Z0
X, Z = np.meshgrid(xs_w.astype(float), zs_w.astype(float), indexing='ij')
H = hf.height(X, Z)
# the ledge: a slab of rock jutting out of the face, flat on top
dl = np.hypot((X - (SPOT[0] - 0.5)) / 2.3, Z / 5.6) * 5.5          # shallow and wide: about three blocks to the edge
shelf = (dl < 5.5) & (X < SPOT[0] + 3)
H_pre = H.copy()
H = np.where(shelf, SPOT[1], np.where((dl < 8) & (X >= SPOT[0] + 3), np.maximum(H, SPOT[1]), H))
trail = (X >= SPOT[0] + 1) & (X <= SPOT[0] + 13) & (np.abs(Z + 0.3 + 0.4 * np.sin(X / 3.0)) < 2.4)     # a worn path cut into the slope
H = np.where(trail, SPOT[1], H)
shelf = shelf | trail
cliff = (X < SPOT[0] - 2) & (X > SPOT[0] - 40) & (np.abs(Z) < 14 + 0.6 * (SPOT[0] - 2 - X))      # a clean drop off the lip of the ledge
H = np.where(cliff & ~shelf, np.minimum(H, SPOT[1] - 4 - (SPOT[0] - 2 - X) * 2.5), H)
H = np.clip(np.round(H).astype(int), 2, NY - 40)
H_pre = np.round(H_pre).astype(int)
pad = np.pad(H, 1, mode='edge')
slope = np.max(np.stack([np.abs(pad[1:-1, 1:-1] - pad[:-2, 1:-1]), np.abs(pad[1:-1, 1:-1] - pad[2:, 1:-1]),
                         np.abs(pad[1:-1, 1:-1] - pad[1:-1, :-2]), np.abs(pad[1:-1, 1:-1] - pad[1:-1, 2:])]), axis=0)
print('heights %.0fs' % (time.time() - t0), flush=True)

vol = np.zeros((NX, NY, NZ), np.uint8)
Y = np.arange(NY)[None, :, None]
strata_n = (6 * nz(X, Z, 26, 11))[:, None, :]
band = ((Y + strata_n) // 5).astype(int) % 7
strata = np.choose(band, [G.STONE, G.ANDESITE, G.STONE, G.TUFF, G.STONE, G.GRANITE, G.ANDESITE]).astype(np.uint8)
solid = Y <= H[:, None, :]
vol[solid] = np.broadcast_to(strata, vol.shape)[solid]
del solid, strata
steep = (slope >= 3) & ~shelf
alpine = H > hf.TREE_Y - 40 + 20 * nz(X, Z, 30, 13)
soil = ~steep
for k in range(1, 4):
    m = soil & (H - k >= 0)
    a, b = np.nonzero(m)
    vol[a, H[a, b] - k, b] = G.DIRT
top = np.full(H.shape, G.GRASS, np.uint8)
ii, kk = np.arange(NX)[:, None], np.arange(NZ)[None, :]
top[steep] = vol[ii, H, kk][steep]
top[(slope == 2) & (nz(X, Z, 9, 14) > 0.55) & ~shelf] = G.COARSE_DIRT
top[alpine & (slope <= 2) & (nz(X, Z, 7, 15) > 0.6) & ~shelf] = G.GRAVEL
wet = (H <= hf.WATER_Y + 1)
top[wet] = np.where(nz(X, Z, 8, 16)[wet] > 0.5, G.SAND, G.GRAVEL)
bed = H < hf.WATER_Y
top[bed] = np.where(nz(X, Z, 10, 17)[bed] > 0.62, G.CLAY, np.where(nz(X, Z, 6, 18)[bed] > 0.45, G.SAND, G.GRAVEL))
top[steep & (nz(X, Z, 12, 19) > 0.64)] = G.MOSSY_STONE
a, b = np.nonzero(np.ones_like(H, bool))
vol[a, H[a, b], b] = top[a, b]
# hollow out beneath the ledge so it's a slab, not a pillar
a, b = np.nonzero(shelf & (H_pre < SPOT[1] - 5))
for x, z in zip(a, b):
    th = int(3 + 3 * np.clip((xs_w[x] - (SPOT[0] - 6)) / 7, 0, 1) + 2 * nz(np.array([xs_w[x]]), np.array([zs_w[z]]), 4, 53)[0])
    vol[x, max(H_pre[x, z], 0) + 1:int(SPOT[1]) - th + 1, z] = G.AIR
# water
for y in range(int(H.min()) + 1, hf.WATER_Y + 1):
    vol[:, y, :][H < y] = G.WATER
print('ground %.0fs' % (time.time() - t0), flush=True)

# ---------------------------------------------------------------- trees and plants
rng = np.random.default_rng(42)
LOGS = (G.OAK_LOG, G.BIRCH_LOG, G.SPRUCE_LOG, G.CHERRY_LOG)


def put(x, y, z, b):
    x -= X0; z -= Z0
    if 0 <= x < NX and 0 <= y < NY and 0 <= z < NZ:
        if vol[x, y, z] in (G.AIR, G.SHORT_GRASS, G.FERN, G.PINK_PETALS) or b in LOGS:
            vol[x, y, z] = b


def blob(cx, cy, cz, rx, ry, rz, leaf, holes=0.12):
    for x in range(int(cx - rx - 1), int(cx + rx + 2)):
        for y in range(int(cy - ry - 1), int(cy + ry + 2)):
            for z in range(int(cz - rz - 1), int(cz + rz + 2)):
                d = ((x - cx) / (rx + 0.5)) ** 2 + ((y - cy) / (ry + 0.5)) ** 2 + ((z - cz) / (rz + 0.5)) ** 2
                if d < 1 and not (d > 0.75 and rng.random() < holes):
                    put(x, y, z, leaf)


def oak(x, y, z, big=False):
    h = rng.integers(5, 8) + (3 if big else 0)
    for k in range(h):
        put(x, y + k, z, G.OAK_LOG)
    r = 2.6 if not big else 4.0
    blob(x, y + h - 1, z, r, r * 0.75, r, G.OAK_LEAVES)


def birch(x, y, z):
    h = rng.integers(6, 9)
    for k in range(h):
        put(x, y + k, z, G.BIRCH_LOG)
    blob(x, y + h - 1, z, 1.8, 2.4, 1.8, G.BIRCH_LEAVES)


def spruce(x, y, z, tall=1.0):
    h = int(rng.integers(9, 15) * tall)
    for k in range(h):
        put(x, y + k, z, G.SPRUCE_LOG)
    put(x, y + h, z, G.SPRUCE_LEAVES); put(x, y + h + 1, z, G.SPRUCE_LEAVES)
    for k in range(3, h):
        r = min((h - k) * 0.42 + (0.6 if (h - k) % 2 == 0 else 0.0), 3.4)
        for dx in range(-4, 5):
            for dz in range(-4, 5):
                if dx * dx + dz * dz <= r * r + 0.3 and (dx or dz):
                    put(x + dx, y + k, z + dz, G.SPRUCE_LEAVES)


def cherry(x, y, z, scale=1.0, lean=(1, 0)):
    h = int(rng.integers(4, 6) * scale)
    cx, cz = x, z
    for k in range(h):
        if k > 1 and rng.random() < 0.35:
            cx += lean[0]; cz += lean[1]
        put(cx, y + k, cz, G.CHERRY_LOG)
    tops = []
    for ang in rng.uniform(0, 2 * math.pi, 3):
        bx, bz, by = cx, cz, y + h - 1
        for k in range(int(3 * scale)):
            bx = int(round(cx + math.cos(ang) * (k + 1))); bz = int(round(cz + math.sin(ang) * (k + 1))); by += (k % 2)
            put(bx, by, bz, G.CHERRY_LOG)
        tops.append((bx, by, bz))
    for bx, by, bz in tops + [(cx, y + h, cz)]:
        blob(bx, by + 1, bz, 2.6 * scale, 1.4 * scale, 2.6 * scale, G.CHERRY_LEAVES, holes=0.22)
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                if rng.random() < 0.18:
                    put(bx + dx, by - 1, bz + dz, G.CHERRY_LEAVES)


F = hf.forest(X, Z, H.astype(float))
PIL = (hf.pillars(X, Z) >= H - 1) & (H > hf.CLOUD_HI)
F = np.where(PIL, 0.75, F)
view = (X < SPOT[0] - 3) & (X > SPOT[0] - 90) & (np.abs(Z) < (SPOT[0] - X) * 0.6 + 5)
n_tree = 0
for gx in range(3, NX - 3, 5):
    for gz in range(3, NZ - 3, 5):
        x = min(max(gx + int(rng.integers(-2, 3)), 2), NX - 3); z = min(max(gz + int(rng.integers(-2, 3)), 2), NZ - 3)
        if top[x, z] not in (G.GRASS, G.COARSE_DIRT) or slope[x, z] > 1 or H[x, z] <= hf.WATER_Y + 1 or shelf[x, z]:
            continue
        wx, wz = int(xs_w[x]), int(zs_w[z])
        if abs(wx - SPOT[0]) < 10 and abs(wz) < 12:
            continue
        if rng.random() > F[x, z] * (0.3 if view[x, z] else 1.0):
            continue
        y = H[x, z] + 1
        if PIL[x, z]:
            k = rng.random()
            if k < 0.45:
                cherry(wx, y, wz, 0.9 + 0.5 * rng.random(), lean=(int(rng.integers(-1, 2)), int(rng.integers(-1, 2))))
            elif k < 0.8:
                oak(wx, y, wz, big=rng.random() < 0.25)
            else:
                birch(wx, y, wz)
        elif H[x, z] > 140:
            spruce(wx, y, wz, tall=1.0 if H[x, z] < 230 else 0.75)
        elif rng.random() < 0.3:
            birch(wx, y, wz)
        elif rng.random() < 0.35:
            spruce(wx, y, wz)
        else:
            oak(wx, y, wz, big=rng.random() < 0.08)
        n_tree += 1
# a few cherry trees on the shoulder behind the ledge, and the old one on the ledge itself
for _ in range(7):
    wx, wz = int(SPOT[0] + rng.integers(12, 34)), int(rng.integers(-26, 27))
    x, z = wx - X0, wz - Z0
    if slope[x, z] <= 2 and top[x, z] == G.GRASS:
        cherry(wx, H[x, z] + 1, wz, 0.9 + 0.3 * rng.random(), lean=(int(rng.integers(-1, 2)), int(rng.integers(-1, 2))))
cherry(int(SPOT[0]) + 1, int(SPOT[1]) + 1, 1, scale=1.25, lean=(-1, 0))
# grass, flowers, petals
meadow = (F < 0.45) & (H < 160)
a, b = np.nonzero(top == G.GRASS)
r = rng.random(len(a)); f = rng.random(len(a))
for x, z, rr, ff in zip(a, b, r, f):
    y = H[x, z] + 1
    if y >= NY or vol[x, y, z] != G.AIR:
        continue
    if shelf[x, z]:
        if rr < 0.45:
            vol[x, y, z] = G.PINK_PETALS
        continue
    if meadow[x, z]:
        if rr < 0.30:
            vol[x, y, z] = G.SHORT_GRASS
        elif rr < 0.40:
            vol[x, y, z] = [G.POPPY, G.DANDELION, G.CORNFLOWER, G.ALLIUM, G.OXEYE][int(ff * 5)]
    elif rr < 0.22:
        vol[x, y, z] = G.SHORT_GRASS if ff < 0.7 else G.FERN
np.savez_compressed('world3.npz', vol=vol, origin=np.array([X0, 0, Z0]), spot=SPOT, water_y=hf.WATER_Y)
print('world3 %dx%dx%d, %d trees, %.0fs' % (NX, NY, NZ, n_tree, time.time() - t0))
