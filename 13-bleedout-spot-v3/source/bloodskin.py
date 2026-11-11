"""Paint pixel-art blood onto a copy of the hero skin: a stain round each arrow wound with runs down the shirt (both
layers, so the jacket layer shows it in 3D), and the left hand that has been pressed to the wounds.
Points are given in the part's model px space and mapped to skin pixels through the part's own mesh UVs."""
import base64, json, os
import numpy as np
from PIL import Image
WEB = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web')
J = json.load(open(os.path.join(WEB, 'static.json')))
PB = {b['name']: b for b in J['bones']}
img = np.asarray(Image.open(os.path.join(WEB, 'static.png')).convert('RGBA')).copy()
# this video only: the black trousers and sleeves a shade lighter, so his legs read as legs in the tree's shade
_l = img.astype(np.float32)
for (u0, v0, u1, v1), k in (((0, 16, 16, 32), 1.65), ((16, 48, 32, 64), 1.65), ((0, 32, 16, 48), 1.65), ((0, 48, 16, 64), 1.65),
                            ((40, 16, 56, 32), 1.3), ((32, 48, 48, 64), 1.3), ((40, 32, 56, 48), 1.3), ((48, 48, 64, 64), 1.3)):
    _l[v0:v1, u0:u1, :3] = np.clip(_l[v0:v1, u0:u1, :3] * k + 4, 0, 255)
img = _l.astype(np.uint8)
rng = np.random.default_rng(3)
DARK, MID, FRESH = (58, 6, 9), (80, 8, 12), (104, 12, 17)


def tris(part):
    b = PB[part]
    f = lambda k, n: np.frombuffer(base64.b64decode(b[k]), np.float32).reshape(-1, n)
    return f('pos', 3).reshape(-1, 3, 3), f('nrm', 3).reshape(-1, 3, 3)[:, 0], f('uv', 2).reshape(-1, 3, 2)


def pix(part, p, normal, outer):
    """skin pixel of the face of `part` with this normal that contains model point p (outer = the overlay layer)"""
    P, N, U = tris(part)
    ax = int(np.argmax(np.abs(normal)))
    ok = (N @ np.asarray(normal, float)) > 0.9
    depth = P[:, 0, ax] * np.sign(normal[ax])
    cand = np.nonzero(ok)[0]
    if not len(cand):
        return None
    d = depth[cand]
    cand = cand[np.isclose(d, d.max() if outer else d.min(), atol=0.05)]
    o = [i for i in range(3) if i != ax]
    for i in cand:
        a, b, c = P[i][:, o]
        q = np.asarray(p, float)[o]
        m = np.array([b - a, c - a]).T
        try:
            l1, l2 = np.linalg.solve(m, q - a)
        except np.linalg.LinAlgError:
            continue
        if l1 >= -1e-4 and l2 >= -1e-4 and l1 + l2 <= 1 + 1e-4:
            uv = U[i][0] + l1 * (U[i][1] - U[i][0]) + l2 * (U[i][2] - U[i][0])
            return int(np.clip(np.floor(uv[0] * 64), 0, 63)), int(np.clip(np.floor(uv[1] * 64), 0, 63))
    return None


def paint(part, p, normal, col, layers=(False, True), a=255):
    for outer in layers:
        q = pix(part, p, normal, outer)
        if q is not None:
            u, v = q
            img[v, u, :3] = col; img[v, u, 3] = a


# arrows in body px (x across, y down from the shoulders); wound pixel centres
for (x, y), run in (((-1.6, 3.2), 5), ((1.9, 5.6), 4), ((0.3, 8.2), 3)):
    cx, cy = int(np.floor(x)) + 0.5, int(np.floor(y)) + 0.5
    for dx, dy, c in ((0, 0, DARK), (-1, 0, MID), (1, 0, MID), (0, -1, MID), (0, 1, FRESH), (1, 1, MID), (-1, 1, DARK)):
        if rng.random() < (1.0 if (dx, dy) in ((0, 0), (0, 1)) else 0.75):
            paint('body', (cx + dx, cy + dy, -2), (0, 0, -1), c)
    rx = cx + rng.choice([-1, 0, 1]) * (rng.random() < 0.4)
    for k in range(2, 2 + run):                                    # the run down the shirt
        if cy + k > 11.5:
            break
        paint('body', (rx, cy + k, -2), (0, 0, -1), FRESH if k < run else MID)
        if rng.random() < 0.3:
            rx += rng.choice([-1, 1])
# the left hand: pressed to the wounds (slim arm: x -1..2, y -2..10 in arm px; the hand is the last rows)
for y in (7.5, 8.5, 9.5):
    for x in (-0.5, 0.5, 1.5):
        if rng.random() < 0.75:
            paint('left_arm', (x, y, -2), (0, 0, -1), MID if rng.random() < 0.6 else DARK)
        if rng.random() < 0.6:
            paint('left_arm', (x, y, 2), (0, 0, 1), DARK)
    for z in (-1.5, -0.5, 0.5):
        if rng.random() < 0.7:
            paint('left_arm', (-1, y, z), (-1, 0, 0), MID)
for x in (-0.5, 0.5, 1.5):
    for z in (-1.5, -0.5, 0.5, 1.5):
        if rng.random() < 0.8:
            paint('left_arm', (x, 10, z), (0, 1, 0), DARK if rng.random() < 0.5 else MID)
# a smear on the right fingertips where he touched it once
for x, z in ((-1.5, -1.5), (-0.5, -1.5), (-1.5, -0.5)):
    paint('right_arm', (x, 10, z), (0, 1, 0), DARK)
Image.fromarray(img).save(os.path.join(WEB, 'static_blood.png'))
print('wrote static_blood.png')
