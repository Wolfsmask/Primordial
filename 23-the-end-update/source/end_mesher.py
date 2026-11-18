"""Voxel volume -> chunked meshes (numpy, vectorised) for the End.  Faces between solid blocks are culled, every vertex
gets blocky-game style ambient occlusion, quads split along the darker diagonal.  (From the bleedout-spot mesher.)
Output groups are dicts of numpy arrays: pos (f32 xyz), nrm (i8), uv (u16 normalised), col (u8 rgb), idx (u32) and chunks."""
import numpy as np
import end_tex as TX

AIR = 0
B = dict(end_stone=1, end_bricks=2, obsidian=3, purpur=4, purpur_pillar=5, bedrock=6, chorus_stem=7, chorus_flower=8,
         void_moss=9, bloom_leaves=10, bloom_log=11, void_crystal=12, end_rod=13, cracked_obsidian=14, gold_trim=15,
         end_grass=16, shardstone=17)
OPAQUE = {1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 14, 15, 17}
LEAVES = {10}
PLANTS = {13, 16}
PETALS = 99
T = TX.IDX
# top, side, bottom
TILES = {1: ('end_stone',) * 3, 2: ('end_bricks',) * 3, 3: ('obsidian',) * 3, 4: ('purpur',) * 3,
         5: ('purpur_top', 'purpur_pillar', 'purpur_top'), 6: ('bedrock',) * 3, 7: ('chorus_stem',) * 3, 8: ('chorus_flower',) * 3,
         9: ('void_moss_top', 'void_moss_side', 'end_stone'), 10: ('bloom_leaves',) * 3, 11: ('bloom_log_top', 'bloom_log', 'bloom_log_top'),
         12: ('void_crystal',) * 3, 13: ('end_rod',) * 3, 14: ('cracked_obsidian',) * 3, 15: ('gold_trim',) * 3,
         16: ('end_grass',) * 3, 17: ('shardstone',) * 3}
TILE_LUT = np.zeros((256, 3), np.int32)
for b, ts in TILES.items():
    TILE_LUT[b] = [T[n] for n in ts]
COLS, ROWS = TX.COLS, (len(TX.TILES) + TX.COLS - 1) // TX.COLS

# per direction: normal, face origin corner, U axis, V axis (V points down the texture)
DIRS = [((1, 0, 0), (1, 1, 1), (0, 0, -1), (0, -1, 0)), ((-1, 0, 0), (0, 1, 0), (0, 0, 1), (0, -1, 0)),
        ((0, 0, 1), (0, 1, 1), (1, 0, 0), (0, -1, 0)), ((0, 0, -1), (1, 1, 0), (-1, 0, 0), (0, -1, 0)),
        ((0, 1, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, 1), (1, 0, 0), (0, 0, -1))]
CORNERS = [(0, 0), (1, 0), (1, 1), (0, 1)]
AO_LEVEL = np.array([0.42, 0.60, 0.80, 1.0], np.float32)


def _shift(a, d, fill):
    """a[x+dx, y+dy, z+dz] at every (x, y, z) (out of range -> fill)"""
    out = np.full_like(a, fill)
    sl_src, sl_dst = [], []
    for k in d:
        if k > 0:
            sl_src.append(slice(k, None)); sl_dst.append(slice(None, -k))
        elif k < 0:
            sl_src.append(slice(None, k)); sl_dst.append(slice(-k, None))
        else:
            sl_src.append(slice(None)); sl_dst.append(slice(None))
    out[tuple(sl_dst)] = a[tuple(sl_src)]
    return out


class Builder:
    def __init__(self):
        self.parts = []          # (chunk keys, pos(n,4,3), nrm(n,3), uv(n,4,2), col(n,4,3), flip(n))

    def add(self, keys, pos, nrm, uv, col, flip=None):
        n = len(pos)
        if n == 0:
            return
        if flip is None:
            flip = np.zeros(n, bool)
        self.parts.append((keys, pos, nrm, uv, col, flip))

    def build(self):
        if not self.parts:
            return None
        keys = np.concatenate([p[0] for p in self.parts]); pos = np.concatenate([p[1] for p in self.parts])
        nrm = np.concatenate([p[2] for p in self.parts]); uv = np.concatenate([p[3] for p in self.parts])
        col = np.concatenate([p[4] for p in self.parts]); flip = np.concatenate([p[5] for p in self.parts])
        order = np.argsort(keys, kind='stable')
        keys, pos, nrm, uv, col, flip = keys[order], pos[order], nrm[order], uv[order], col[order], flip[order]
        n = len(keys)
        P = pos.reshape(-1, 3).astype(np.float32)
        N = np.repeat(nrm[:, None, :], 4, 1).reshape(-1, 3).astype(np.int8)
        U = np.clip(uv.reshape(-1, 2) * 65535, 0, 65535).astype(np.uint16)
        C = np.clip(col.reshape(-1, 3), 0, 255).astype(np.uint8)
        tri_a = np.array([0, 1, 2, 0, 2, 3]); tri_b = np.array([1, 2, 3, 1, 3, 0])
        chunks, idx = [], []
        starts = np.concatenate([[0], np.nonzero(np.diff(keys))[0] + 1, [n]])
        i0 = 0
        for a, b in zip(starts[:-1], starts[1:]):
            q = np.arange(b - a)
            loc = np.where(flip[a:b, None], tri_b[None, :], tri_a[None, :]) + (q * 4)[:, None]
            idx.append(loc.reshape(-1).astype(np.uint32))
            chunks.append((int(keys[a]), int(a * 4), int((b - a) * 4), int(i0), int((b - a) * 6)))
            i0 += (b - a) * 6
        return dict(pos=P, nrm=N, uv=U, col=C, idx=np.concatenate(idx), chunks=chunks)


def mesh_volume(vol, origin, scale=1.0, mode='textured', colors=None, chunk=64, leaves_cull=True):
    """returns dict group -> Builder.build() output.  colors: (256, 3) array (linear 0..1) for 'colored' mode"""
    vol = np.asarray(vol)
    ox, oy, oz = origin
    opaque = np.isin(vol, list(OPAQUE)) if mode == 'textured' else (vol > 0)
    leaves = np.isin(vol, list(LEAVES)) if mode == 'textured' else np.zeros_like(opaque)
    groups = {'opaque': Builder(), 'leaves': Builder(), 'plants': Builder()}
    occl = opaque | leaves                                   # what darkens corners
    rng = np.random.default_rng(5)
    for n, o, U, V in DIRS:
        n_, o_, U_, V_ = np.array(n), np.array(o), np.array(U), np.array(V)
        nb_op = _shift(opaque, n, False); nb_lv = _shift(leaves, n, False)
        vis_lv = (leaves & ~nb_op & ~nb_lv) if leaves_cull else (leaves & ~nb_op)
        for gname, solid, vis in (('opaque', opaque, opaque & ~nb_op), ('leaves', leaves, vis_lv)):
            xs, ys, zs = np.nonzero(vis)
            if len(xs) == 0:
                continue
            B = np.stack([xs, ys, zs], 1)
            blk = vol[xs, ys, zs]
            # ambient occlusion from the layer in front of the face
            ao = np.zeros((len(xs), 4), np.float32)
            front = B + n_
            for ci, (cu, cv) in enumerate(CORNERS):
                du = (U_ if cu else -U_); dv = (V_ if cv else -V_)
                def occ(p):
                    p = np.asarray(p)
                    ok = (p[:, 0] >= 0) & (p[:, 0] < vol.shape[0]) & (p[:, 1] >= 0) & (p[:, 1] < vol.shape[1]) & (p[:, 2] >= 0) & (p[:, 2] < vol.shape[2])
                    r = np.zeros(len(p), bool)
                    q = p[ok]; r[ok] = occl[q[:, 0], q[:, 1], q[:, 2]]
                    return r
                s1 = occ(front + du); s2 = occ(front + dv); c = occ(front + du + dv)
                lvl = np.where(s1 & s2, 0, 3 - (s1.astype(int) + s2.astype(int) + c.astype(int)))
                ao[:, ci] = AO_LEVEL[lvl]
            pos = np.stack([(B + o_ + (U_ * cu + V_ * cv) * 1.0) for cu, cv in CORNERS], 1).astype(np.float32)
            # U x V must point along n for counter-clockwise front faces
            if np.dot(np.cross(U_, V_), n_) < 0:
                pos = pos[:, [0, 3, 2, 1]]; ao = ao[:, [0, 3, 2, 1]]
                cuv = [CORNERS[i] for i in (0, 3, 2, 1)]
            else:
                cuv = CORNERS
            pos = pos * scale + np.array([ox, oy, oz], np.float32)
            flip = (ao[:, 0] + ao[:, 2]) < (ao[:, 1] + ao[:, 3])
            if mode == 'textured':
                face = 0 if n[1] > 0 else (2 if n[1] < 0 else 1)
                tile = TILE_LUT[blk, face]
                tc, tr = tile % COLS, tile // COLS
                e = 0.002
                uv = np.stack([np.stack([(tc + (e + (1 - 2 * e) * cu)) / COLS, (tr + (e + (1 - 2 * e) * cv)) / ROWS], 1) for cu, cv in cuv], 1)
                shade = np.ones(len(xs), np.float32)
                col = ao[:, :, None] * shade[:, None, None] * np.ones(3, np.float32) * 255
            else:
                uv = np.zeros((len(xs), 4, 2), np.float32)
                base = colors[blk]                                          # linear rgb
                jit = 1 + 0.07 * (rng.random(len(xs)) - 0.5)
                col = ao[:, :, None] * (base * jit[:, None])[:, None, :] * 255
            keys = ((xs * scale + ox) // chunk).astype(np.int64) * 100000 + ((zs * scale + oz) // chunk).astype(np.int64)
            groups[gname].add(keys, pos, np.broadcast_to(np.array(n, np.int8) * 127, (len(xs), 3)), uv, col, flip)
    if mode == 'textured':
        # crossed plants and flat petals
        for bset, kind in ((PLANTS, 'cross'), ({PETALS}, 'flat')):
            xs, ys, zs = np.nonzero(np.isin(vol, list(bset)))
            if len(xs) == 0:
                continue
            blk = vol[xs, ys, zs]; tile = TILE_LUT[blk, 1]; tc, tr = tile % COLS, tile // COLS
            B = np.stack([xs, ys, zs], 1).astype(np.float32) + np.array([ox, oy, oz], np.float32)
            keys = ((xs + ox) // chunk).astype(np.int64) * 100000 + ((zs + oz) // chunk).astype(np.int64)
            e = 0.002
            uvq = np.stack([np.stack([(tc + e + (1 - 2 * e) * cu) / COLS, (tr + e + (1 - 2 * e) * cv) / ROWS], 1) for cu, cv in CORNERS], 1)
            up = np.broadcast_to(np.array([0, 127, 0], np.int8), (len(xs), 3))
            col = np.full((len(xs), 4, 3), 255.0, np.float32)
            if kind == 'cross':
                for a, b in (((0.15, 0.15), (0.85, 0.85)), ((0.15, 0.85), (0.85, 0.15))):
                    pos = np.stack([B + [a[0], 1, a[1]], B + [b[0], 1, b[1]], B + [b[0], 0, b[1]], B + [a[0], 0, a[1]]], 1)
                    groups['plants'].add(keys, pos.astype(np.float32), up, uvq, col)
            else:
                y = 0.04
                pos = np.stack([B + [0, y, 0], B + [1, y, 0], B + [1, y, 1], B + [0, y, 1]], 1)
                groups['plants'].add(keys, pos.astype(np.float32), up, uvq, col)
    return {k: g.build() for k, g in groups.items()}


def save_groups(groups, out_dir, prefix, meta):
    """one .bin per group: pos f32 | nrm i8 (padded) | uv u16 | col u8 (padded) | idx u32"""
    import os
    for name, g in groups.items():
        if g is None:
            continue
        parts, buf, off = {}, bytearray(), 0
        for k in ('pos', 'idx', 'uv', 'nrm', 'col'):
            b = g[k].tobytes()
            parts[k] = [off, int(g[k].size)]
            buf += b; off += len(b)
            pad = (-off) % 4
            buf += b'\0' * pad; off += pad
        fn = f'{prefix}_{name}.bin'
        open(os.path.join(out_dir, fn), 'wb').write(bytes(buf))
        meta[f'{prefix}_{name}'] = dict(file=fn, parts=parts, chunks=g['chunks'], verts=int(len(g['pos'])), tris=int(len(g['idx']) // 3))
    return meta
