"""Precompute the black hole's lensing map (Schwarzschild light bending, traced backwards from the eye).
The map covers +-F (gnomonic) round the hole's direction.  Per pixel:
  A = (u1, v1, g1, hit1)   first crossing of the disk plane: disk-plane coords (rs units), Doppler*gravity factor
  B = (u2, v2, g2, hit2)   second crossing (the lensed far side / the photon ring)
  C = (escape dir xyz in world, captured)
python3 end2_bh.py [N] -> web/bh_a.bin, bh_b.bin, bh_c.bin (float16 rgba), web/bh.json"""
import json, sys, time
import numpy as np

DEG = np.pi / 180
def dirYE(yaw, el): return np.array([-np.sin(yaw * DEG) * np.cos(el * DEG), np.sin(el * DEG), -np.cos(yaw * DEG) * np.cos(el * DEG)])
def unit(v): return v / np.linalg.norm(v)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
SC = json.load(open('web/scene.json'))
c = dirYE(SC['bh']['yaw'], SC['bh']['el']); ex = unit(np.cross(c, [0, 1, 0])); ey = np.cross(ex, c)
F = np.tan(34 * DEG); D = 34.0; RIN, ROUT = 3.0, 13.5
tilt, roll = 8.5 * DEG, -7 * DEG
nd = unit(ey * np.cos(tilt) - c * np.sin(tilt)); nd = unit(nd * np.cos(roll) + np.cross(c, nd) * np.sin(roll))
e1 = unit(np.cross(nd, c)); e2 = np.cross(nd, e1)

t0 = time.time()
g = (np.arange(N) + 0.5) / N * 2 - 1
GX, GY = np.meshgrid(g * F, g * F)                  # row = y
rd = GX[..., None] * ex + GY[..., None] * ey + c; rd /= np.linalg.norm(rd, axis=-1, keepdims=True)
rd = rd.reshape(-1, 3).astype(np.float64); M = len(rd)
x = np.tile(-D * c, (M, 1)); v = rd.copy()
out_a = np.zeros((M, 4), np.float32); out_b = np.zeros((M, 4), np.float32); out_c = np.zeros((M, 4), np.float32)
nhit = np.zeros(M, np.int8)
act = np.arange(M)
h2 = np.sum(np.cross(x, v) ** 2, 1)
side = np.sign(x @ nd)
for step in range(2000):
    if len(act) == 0: break
    xa, va, hh = x[act], v[act], h2[act]
    r = np.linalg.norm(xa, axis=1); dt = np.clip(0.035 * r, 0.004, 3.0)
    # velocity Verlet
    acc = -1.5 * hh[:, None] * xa / r[:, None] ** 5
    vh = va + 0.5 * dt[:, None] * acc
    xn = xa + dt[:, None] * vh
    rn = np.linalg.norm(xn, axis=1)
    acc2 = -1.5 * hh[:, None] * xn / rn[:, None] ** 5
    vn = vh + 0.5 * dt[:, None] * acc2
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    # disk plane crossings
    s0 = xa @ nd; s1 = xn @ nd; cr = (np.sign(s0) != np.sign(s1)) & (s0 != 0)
    if cr.any():
        k = np.nonzero(cr)[0]; f = s0[k] / (s0[k] - s1[k]); xi = xa[k] + (xn[k] - xa[k]) * f[:, None]
        ri = np.linalg.norm(xi, axis=1); u1 = xi @ e1; u2 = xi @ e2
        phat = np.cross(nd, xi); phat /= np.linalg.norm(phat, axis=1, keepdims=True)
        beta = np.clip(np.sqrt(0.5 / np.maximum(ri - 1.0, 0.05)), 0, 0.72); gam = 1 / np.sqrt(1 - beta ** 2)
        vd = vn[k]; delta = 1 / (gam * (1 + beta * np.sum(phat * vd, 1)))
        gg = delta * np.sqrt(np.clip(1 - 1 / np.maximum(ri, 1.0001), 0.02, 1))
        ids = act[k]; nh = nhit[ids]
        for which, arr in ((0, out_a), (1, out_b)):
            m = nh == which
            arr[ids[m]] = np.stack([u1[m], u2[m], gg[m], np.ones(m.sum())], 1)
        nhit[ids] = np.minimum(nh + 1, 3)
    x[act] = xn; v[act] = vn
    cap = rn < 1.02
    esc = (rn > 3 * D) & (np.sum(xn * vn, 1) > 0)
    if cap.any(): out_c[act[cap], 3] = 1.0
    if esc.any(): out_c[act[esc], :3] = vn[esc]
    act = act[~(cap | esc)]
    if step % 100 == 0: print('step', step, 'active', len(act), '%.0fs' % (time.time() - t0), flush=True)
if len(act): out_c[act, :3] = v[act]; print('unfinished', len(act))
for nm, arr in (('a', out_a), ('b', out_b), ('c', out_c)):
    arr.reshape(N, N, 4).astype(np.float16).tofile('web/bh_%s.bin' % nm)
json.dump(dict(N=N, F=F, D=D, rin=RIN, rout=ROUT, c=c.tolist(), ex=ex.tolist(), ey=ey.tolist(), nd=nd.tolist()), open('web/bh.json', 'w'))
print('done %.0fs' % (time.time() - t0))
# a quick look
img = np.zeros((N, N, 3))
A = out_a.reshape(N, N, 4); Bb = out_b.reshape(N, N, 4); Cc = out_c.reshape(N, N, 4)
def shade(H):
    r = np.hypot(H[..., 0], H[..., 1]); p = np.clip(r / RIN, 0.5, 9) ** -2.0 * (r > RIN) * (r < ROUT) * H[..., 3]
    return p * np.clip(H[..., 2], 0, 2) ** 3
d1 = shade(A); d2 = shade(Bb); al1 = np.clip(d1 * 2, 0, 1)
bgk = (np.sin(Cc[..., 0] * 40) * np.sin(Cc[..., 1] * 40) > 0.6) * 0.15 * (1 - Cc[..., 3])
lum = d1 + (1 - al1) * (d2 + bgk)
from PIL import Image
Image.fromarray((np.clip(lum[::-1], 0, 1) ** 0.5 * 255).astype(np.uint8)).resize((768, 768)).save('bh_preview.png')
