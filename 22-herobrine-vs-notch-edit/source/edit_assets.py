"""Original skins and props for the edit: an ash-skinned hooded figure with white eyes (Herobrine) and a white-and-gold
creator figure (Notch), their capes and blades, and the arena's block textures.  Standard 64 x 64 box-UV skin layout.
-> web/herobrine.png, web/notch.png, web/cape_h.png, web/cape_n.png, web/blade_h.png, web/blade_n.png, web/blocks.png"""
import os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web')
sys.path.insert(0, os.path.join(HERE, 'deps'))
import tex_ow, tex2

rng = np.random.default_rng(3)


def noisy(base, h, w, sp=0.08, seed=None):
    r = np.random.default_rng(seed)
    a = np.zeros((h, w, 4)); a[..., :3] = np.array(base, float) * (1 - sp + 2 * sp * r.random((h, w)))[..., None]; a[..., 3] = 255
    return a


def put(img, u, v, w, h, d, faces):
    """box-UV: faces = dict(top, bottom, right, front, left, back) of arrays (h, w, 4) sized per face"""
    reg = {'top': (u + d, v, w, d), 'bottom': (u + d + w, v, w, d), 'right': (u, v + d, d, h), 'front': (u + d, v + d, w, h),
           'left': (u + d + w, v + d, d, h), 'back': (u + 2 * d + w, v + d, w, h)}
    for k, (x, y, ww, hh) in reg.items():
        if k in faces:
            img[y:y + hh, x:x + ww] = faces[k](ww, hh) if callable(faces[k]) else faces[k]


def herobrine():
    img = np.zeros((64, 64, 4))
    ash, hood, hood2, cloak, cloak2, trous, boot = (126, 124, 120), (34, 33, 40), (48, 46, 56), (30, 29, 36), (44, 42, 52), (40, 40, 44), (20, 20, 22)
    f = lambda c, sp=0.08: (lambda w, h: noisy(c, h, w, sp))
    def face(w, h):
        a = noisy(ash, h, w, 0.06)
        a[3, 1:3, :3] = a[3, 5:7, :3] = (250, 250, 250)        # eyes
        a[2, 1:3, :3] = a[2, 5:7, :3] = (88, 86, 84)            # brows in shadow
        a[6, 2:6, :3] = (70, 68, 66)                             # mouth line
        a[4:6, 0, :3] = a[4:6, 7, :3] = (104, 102, 98)
        return a
    put(img, 0, 0, 8, 8, 8, dict(top=f(ash), bottom=f(ash), right=f(ash), front=face, left=f(ash), back=f(ash)))
    # the hood (outer head layer): over the top, sides and back, open at the face, ragged edge on the brow
    def hood_front(w, h):
        a = np.zeros((h, w, 4)); a[0:2] = noisy(hood, 2, w, 0.1); a[2, [0, 1, 6, 7]] = noisy(hood, 1, 4, 0.1).reshape(4, 4)
        a[2:8, 0] = noisy(hood2, 6, 1, 0.1)[:, 0]; a[2:8, 7] = noisy(hood2, 6, 1, 0.1)[:, 0]
        return a
    put(img, 32, 0, 8, 8, 8, dict(top=f(hood, 0.12), right=f(hood2, 0.12), front=hood_front, left=f(hood2, 0.12), back=f(hood, 0.12)))
    def cloak_front(w, h):
        a = noisy(cloak, h, w, 0.1)
        a[:, 3:5, :3] = np.array(cloak2)                        # the opening of the cloak
        for y in range(2, h, 3):
            a[y, 3:5, :3] = (90, 24, 24) if y % 2 else (120, 30, 28)   # dull red clasps / cracks
        a[8, :, :3] = (22, 20, 24)                               # belt
        return a
    put(img, 16, 16, 8, 12, 4, dict(top=f(cloak), bottom=f(cloak), right=f(cloak2), front=cloak_front, left=f(cloak2), back=f(cloak)))
    def sleeve(w, h):
        a = noisy(cloak2, h, w, 0.1); a[h - 3:, :, :3] = np.array(ash) * (0.95 + 0.1 * rng.random((3, w, 1))); return a
    for (u, v) in ((40, 16), (32, 48)):
        put(img, u, v, 4, 12, 4, dict(top=f(cloak2), bottom=f(ash), right=sleeve, front=sleeve, left=sleeve, back=sleeve))
    def leg(w, h):
        a = noisy(trous, h, w, 0.08); a[h - 3:, :, :3] = np.array(boot); return a
    for (u, v) in ((0, 16), (16, 48)):
        put(img, u, v, 4, 12, 4, dict(top=f(trous), bottom=f(boot), right=leg, front=leg, left=leg, back=leg))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(WEB, 'herobrine.png'))


def notch():
    img = np.zeros((64, 64, 4))
    skin, hair, hair2, coat, coat2, gold, gold2, trous, boot = (206, 156, 112), (214, 176, 72), (190, 150, 56), (232, 232, 226), (210, 210, 204), (232, 184, 48), (196, 150, 36), (176, 176, 172), (240, 240, 236)
    f = lambda c, sp=0.06: (lambda w, h: noisy(c, h, w, sp))
    def face(w, h):
        a = noisy(skin, h, w, 0.05)
        a[0:2] = noisy(hair, 2, w, 0.08)
        a[2, 0] = a[2, 7] = np.array([*hair, 255])
        a[3, 1:3, :3] = a[3, 5:7, :3] = (255, 220, 120)          # golden eyes
        a[3, 1, :3] = a[3, 6, :3] = (255, 255, 230)
        a[6, 2:6, :3] = (150, 104, 74)
        a[7, 1:7, :3] = np.array(hair2) * 0.95                    # a short beard line
        return a
    def head_side(w, h):
        a = noisy(skin, h, w, 0.05); a[0:3] = noisy(hair, 3, w, 0.08); a[3:6, 4:] = noisy(hair2, 3, w - 4, 0.08); return a
    put(img, 0, 0, 8, 8, 8, dict(top=f(hair, 0.1), bottom=f(skin), right=head_side, front=face, left=head_side, back=f(hair2, 0.1)))
    def coat_front(w, h):
        a = noisy(coat, h, w, 0.05)
        a[:, 3:5, :3] = np.array(gold)                            # gold placket
        a[0, :, :3] = np.array(gold); a[8, :, :3] = np.array(gold2)
        a[3:5, 1:3, :3] = np.array(gold)                          # a small block emblem
        return a
    put(img, 16, 16, 8, 12, 4, dict(top=f(coat), bottom=f(coat), right=f(coat2), front=coat_front, left=f(coat2), back=f(coat)))
    def sleeve(w, h):
        a = noisy(coat, h, w, 0.05); a[h - 4, :, :3] = np.array(gold); a[h - 3:, :, :3] = np.array(skin); return a
    for (u, v) in ((40, 16), (32, 48)):
        put(img, u, v, 4, 12, 4, dict(top=f(coat), bottom=f(skin), right=sleeve, front=sleeve, left=sleeve, back=sleeve))
    def leg(w, h):
        a = noisy(trous, h, w, 0.05); a[h - 4:, :, :3] = np.array(boot); a[h - 4, :, :3] = np.array(gold); return a
    for (u, v) in ((0, 16), (16, 48)):
        put(img, u, v, 4, 12, 4, dict(top=f(trous), bottom=f(boot), right=leg, front=leg, left=leg, back=leg))
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(os.path.join(WEB, 'notch.png'))


def cape(name, base, trim, ragged):
    W_, H_ = 10, 16
    a = noisy(base, H_, W_, 0.08)
    if trim is not None:
        a[:, 0, :3] = a[:, -1, :3] = np.array(trim); a[-1, :, :3] = np.array(trim)
    if ragged:
        for x in range(W_):
            cut = int(rng.integers(0, 4))
            a[H_ - cut:, x, 3] = 0
        a[rng.random((H_, W_)) < 0.04, 3] = 0
    big = np.repeat(np.repeat(a, 4, 0), 4, 1)
    Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).save(os.path.join(WEB, name))


def blade(name, edge, core, hilt, guard):
    """a 16 x 16 diagonal sword sprite (original): a clean three-pixel blade, core line between two edge lines"""
    a = np.zeros((16, 16, 4))
    for x in range(4, 16):
        for y in range(0, 13):
            s = x + y - 15
            if s == 0:
                a[y, x] = (*core, 255)
            elif s in (-1, 1) and x < 15:
                a[y, x] = (*edge, 255)
    a[0, 15] = (*edge, 255)
    for (y, x) in ((10, 2), (11, 3), (12, 4), (13, 5), (9, 3), (12, 3), (11, 4), (10, 3), (13, 4)):   # cross-guard
        a[y, x] = (*guard, 255)
    for (y, x) in ((12, 2), (13, 1), (14, 1), (14, 0), (15, 0)):                                       # grip and pommel
        a[y, x] = (*hilt, 255)
    big = np.repeat(np.repeat(a, 4, 0), 4, 1)
    Image.fromarray(np.clip(big, 0, 255).astype(np.uint8)).save(os.path.join(WEB, name))


def blocks():
    """arena textures in a strip: stone, cobble, obsidian, mossy cobble, gravel, cracked lava-stone, lava, dark stone"""
    tiles = [tex2.tile('stone'), tex2.tile('cobblestone'), tex_ow.tile('obsidian'), tex2.tile('mossy_cobblestone'), tex2.tile('gravel'),
             None, tex2.tile('lava'), None]
    cr = tex2.tile('stone').copy()
    yy, xx = np.mgrid[0:16, 0:16]
    crack = (np.abs(np.sin(xx * 0.9 + yy * 0.4)) < 0.18) | (np.abs(np.sin(yy * 1.1 - xx * 0.3 + 2)) < 0.14)
    cr[crack, :3] = (255, 120, 30)
    tiles[5] = cr
    dk = tex2.tile('stone').copy(); dk[..., :3] *= 0.45; tiles[7] = dk
    strip = np.concatenate(tiles, 1)
    Image.fromarray(np.clip(strip, 0, 255).astype(np.uint8)).save(os.path.join(WEB, 'blocks.png'))
    # emission mask for the cracked stone and lava (white where it glows)
    em = np.zeros_like(strip); em[..., 3] = 255
    em[:, 5 * 16:6 * 16][crack] = (255, 255, 255, 255); em[:, 6 * 16:7 * 16, :3] = 255
    Image.fromarray(em.astype(np.uint8)).save(os.path.join(WEB, 'blocks_em.png'))


if __name__ == '__main__':
    os.makedirs(WEB, exist_ok=True)
    herobrine(); notch()
    cape('cape_h.png', (28, 27, 34), None, True)
    cape('cape_n.png', (236, 236, 230), (232, 184, 48), False)
    blade('blade_h.png', (214, 36, 28), (112, 14, 12), (50, 40, 36), (70, 20, 20))
    blade('blade_n.png', (255, 236, 160), (255, 255, 245), (120, 90, 40), (232, 184, 48))
    blocks()
    print('assets written')
