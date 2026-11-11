"""10/10 bleedout spot v5, first person, to "Feel Good Inc." (59.85 s - 75.95 s of the song).
A few quick staggering steps to the cherry tree with his eyes on the ground (blood dripping off the arrows), a hand on
the bark, a turn, the slide and the plop on the last hit before the bridge; slumped, breathing; on "Windmill" he looks
up at a world that shouldn't exist, looks round, and his eyes close.  The shader look is on the whole time (render3).
python3 fp3.py test [times] | render [a:b]"""
import base64, io, math, os, pickle, subprocess, sys, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
import view5 as V
import motion3 as M
import rig
HERE = os.path.dirname(os.path.abspath(__file__))
FPS, DUR = M.FPS, M.DUR
W, H = 1080, 1920
NF = int(round(DUR * FPS))
ss = M.ss
TREE_C = [11.5, 251.6, 1.5]
ARROWS = [(-1.6, 3.2, 0.10, 0.18), (1.9, 5.6, -0.08, -0.22), (0.3, 8.2, 0.16, 0.05)]


# ---------------------------------------------------------------- blood
def arrow_frame(F, pose, a):
    x, y, rx_, ry_ = a
    w = 1 - ss((y - 3.8) / 4.4)
    R = rig.rzyx(pose['waist'][0] * w, 0, 0)
    piv = np.array([0.0, -6.0, 0.0])
    pos = piv + R @ (np.array([x, y - 12.0, -2.0]) - piv)
    return F['hips'] @ rig.M(R @ rig.rzyx(rx_, ry_, 0), pos)


def blood(frames):
    """drops gather under the shafts, let go, fall and leave pixel splats on the grass"""
    rng = np.random.default_rng(11)
    T_STOP = M.T_PLOP - 0.5
    ev = []
    for ai in range(3):
        t = -0.45 + ai * 0.3 + rng.random() * 0.3
        while t < T_STOP:
            ev.append(dict(a=ai, t0=t, z=-rng.uniform(0.8, 3.6), form=rng.uniform(0.35, 0.55), size=rng.uniform(0.019, 0.026),
                           splat=rng.uniform(0.05, 0.10), rot=float(rng.uniform(0, 6.28))))
            t += rng.uniform(0.75, 1.2)
    pts = {(e['a'], round(e['z'], 3)): [] for e in ev}
    for fr in frames:
        F = M.ST.frames(fr['pose'])
        for (ai, z) in pts:
            pts[(ai, z)].append(rig.pt(arrow_frame(F, fr['pose'], ARROWS[ai]), (0.0, 0.0, z)) - np.array([0, 0.012, 0]))
    for key in pts:
        pts[key] = np.array(pts[key])

    def at(key, t):
        k = min(max(t * FPS, 0), NF - 1.001); i = int(k); u = k - i
        P = pts[key]
        return P[i] + (P[min(i + 1, NF - 1)] - P[i]) * u
    g = 9.81
    for e in ev:
        key = (e['a'], round(e['z'], 3)); td = e['t0'] + e['form']
        p0 = at(key, td); v0 = (at(key, td + 1 / FPS) - at(key, td - 1 / FPS)) * FPS / 2
        a_, b_, c_ = -0.5 * g, v0[1], p0[1] - M.GROUND
        s_land = (-b_ - math.sqrt(b_ * b_ - 4 * a_ * c_)) / (2 * a_)
        e.update(key=key, td=td, p0=p0, v0=v0, tl=td + s_land, land=p0 + v0 * s_land + np.array([0, -0.5 * g * s_land ** 2, 0]))
    per_frame = []
    for k in range(NF):
        t = k / FPS
        drops, splats = [], []
        for e in ev:
            if e['tl'] <= t:
                L = e['land']; sz = e['splat'] * min(1.0, 0.4 + (t - e['tl']) / 0.08)
                splats.append([float(L[0]), M.GROUND + 0.004, float(L[2]), float(sz), e['rot']])
            elif e['td'] <= t:
                s = t - e['td']; p = e['p0'] + e['v0'] * s + np.array([0, -0.5 * g * s * s, 0])
                sp = abs(e['v0'][1] - g * s)
                drops.append([float(p[0]), float(p[1]), float(p[2]), e['size'] * 0.85, e['size'] * (1 + 0.35 * min(sp, 3))])
            elif e['t0'] <= t:
                u = (t - e['t0']) / e['form']
                p = at(e['key'], t) - np.array([0, 0.004 * u, 0])
                drops.append([float(p[0]), float(p[1]), float(p[2]), e['size'] * (0.35 + 0.6 * u), e['size'] * (0.35 + 0.8 * u)])
        per_frame.append((drops, splats))
    return per_frame


# ---------------------------------------------------------------- finishing: fade in, the impact, grain, eyelids, caption
FONT = ImageFont.truetype('/usr/share/fonts/opentype/inter/Inter-Bold.otf', 66) if os.path.exists('/usr/share/fonts/opentype/inter/Inter-Bold.otf') else None
YY, XX = np.mgrid[0:H, 0:W]


def post(img, t, w=W, h=H):
    a = img.astype(np.float32) / 255
    bl = 5.0 * math.exp(-((t - (M.T_PLOP + 0.06)) / 0.35) ** 2) + 2.2 * math.exp(-((t - (M.T_PLOP + 0.9)) / 0.6) ** 2)
    if bl > 0.35:
        a = cv2.GaussianBlur(a, (0, 0), bl * w / W)
    if M.T_PLOP - 0.05 < t < M.T_PLOP + 0.9:                     # the jolt darkens the edges for a moment
        k = math.exp(-((t - M.T_PLOP - 0.1) / 0.28) ** 2)
        yy, xx = np.mgrid[0:h, 0:w]
        r = np.hypot((xx - w / 2) / w, (yy - h * 0.48) / h)
        a *= (1 - 0.55 * k * np.clip(r - 0.2, 0, None) ** 1.2 * 2.0)[..., None]
    if t > DUR - 1.9:                                            # eyelids close as it fades
        u = ss((t - (DUR - 1.9)) / 1.8)
        yy = np.mgrid[0:h, 0:w][0]
        ey = np.abs(yy - h * 0.47) / (h * 0.5)
        a *= np.clip((1.0 - u * 0.95 - ey) / 0.3 + (1 - u) * 1.5, 0, 1)[..., None]
    g = np.random.default_rng(int(t * 1000)).normal(0, 0.008, (h // 2, w // 2)).repeat(2, 0).repeat(2, 1)[..., None]
    a = np.clip(a + g, 0, 1)
    a *= ss(t / 0.35) * (1 - ss((t - (DUR - 0.9)) / 0.85))
    out = Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
    ca = ss((t - 8.4) / 0.7) * (1 - ss((t - 11.4) / 0.7))
    if ca > 0.01 and FONT is not None and w == W:
        txt = '10/10 bleedout spot'
        layer = Image.new('RGBA', out.size, (0, 0, 0, 0)); d = ImageDraw.Draw(layer)
        w_ = d.textlength(txt, font=FONT); x0, y0 = (W - w_) / 2, H * 0.15
        for dx, dy in ((0, 4), (2, 4), (-2, 4)):
            d.text((x0 + dx, y0 + dy), txt, font=FONT, fill=(20, 10, 20, int(120 * ca)))
        d.text((x0, y0), txt, font=FONT, fill=(255, 250, 244, int(235 * ca)))
        out = Image.alpha_composite(out.convert('RGBA'), layer).convert('RGB')
    return out


# ---------------------------------------------------------------- sound: the song (with v2's thud and skid on the same hits) + new steps
def audio(path):
    from scipy.io import wavfile
    SR = 48000
    sr, bed = wavfile.read(os.path.join(HERE, 'audio', 'v2_bed.wav'))
    bed = bed.astype(np.float32)
    s0 = int((M.SONG0 - 50.60) * sr); n = int(DUR * sr)
    song = bed[s0:s0 + n].copy()
    tt = np.arange(len(song)) / sr
    song *= (np.clip(tt / 0.15, 0, 1) * np.clip((DUR - tt) / 1.5, 0, 1) ** 1.5)[:, None]
    fx = np.zeros(len(song), np.float32)

    def load(name):
        r, x = wavfile.read(os.path.join(HERE, 'sfx', name)); x = x.astype(np.float32); x /= np.abs(x).max() + 1e-9
        return x if x.ndim == 1 else x.mean(1)

    def put(x, t0, gain, rate=1.0):
        if rate != 1.0:
            x = np.interp(np.arange(0, len(x) - 1, rate), np.arange(len(x)), x)
        i = int(t0 * sr)
        if 0 <= i < len(fx):
            m = min(len(x), len(fx) - i); fx[i:i + m] += x[:m] * gain
    steps = [load('leaves01.wav'), load('leaves02.wav'), load('mud02.wav')]
    rng = np.random.default_rng(4)
    for k, tc in enumerate((M.beat(23), M.beat(25), M.beat(25.75))):          # the bed already has the steps on beats 22 and 24
        put(steps[k % 3], tc - 0.02, 0.075, float(rng.uniform(0.9, 1.05)))
    put(load('wood01.wav'), M.T_REACH1 - 0.01, 0.045, 0.85)                       # his hand on the bark
    fx = cv2.GaussianBlur(fx.reshape(1, -1), (0, 0), 1.6).ravel()
    mix = song + fx[:, None] * np.array([1.0, 0.96], np.float32)
    pk = np.abs(mix).max()
    if pk > 0.95:
        mix *= 0.95 / pk
    wavfile.write(path, sr, (mix * 32767).astype(np.int16))


# ---------------------------------------------------------------- render
def view_of(fr, bl, w, h):
    e = fr['eye']; f = fr['f']; up = fr['up']
    return dict(w=w, h=h, fov=float(fr['fov']), pos=e.tolist(), look=(e + f).tolist(), up=up.tolist(), hero=fr['pose'], near=0.04,
                time=float(fr['t']), fp=True, petalsAt=TREE_C, drops=bl[0], splats=bl[1], god=1.0, exposure=1.2)


def run(views, out_dir, final, names=None):
    os.makedirs(out_dir, exist_ok=True)
    V.ensure_server()
    with sync_playwright() as pw:
        b, pg = V.open_page(pw)
        t0 = time.time()
        for k, (t, v) in enumerate(views):
            v['shadows'] = (k == 0)
            name = names[k] if names else 'f_%04d.png' % k
            img = np.asarray(V.shot(pg, v))
            (post(img, t) if final else Image.fromarray(img)).save(os.path.join(out_dir, name))
            if k % 30 == 0:
                print('frame %d/%d  %.0fs' % (k, len(views), time.time() - t0), flush=True)
        b.close()


def sheet(paths, out, cols, w, h):
    rows = (len(paths) + cols - 1) // cols
    s = Image.new('RGB', (cols * w, rows * h))
    for i, p in enumerate(paths):
        s.paste(Image.open(p).resize((w, h)), ((i % cols) * w, (i // cols) * h))
    s.save(out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    frames = pickle.load(open(os.path.join(HERE, 'motion3.pkl'), 'rb'))
    bl = blood(frames)
    if mode == 'test':
        ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.2, 0.5, 1.0, 1.6, 2.3, 2.8, 3.1, 4.4, 6.0, 7.6, 9.5, 11.0, 13.6, 15.2, 15.9]
        vs = [(t, view_of(frames[min(int(t * FPS), NF - 1)], bl[min(int(t * FPS), NF - 1)], 360, 640)) for t in ts]
        d = os.path.join(HERE, 'fp5_test'); run(vs, d, False)
        for k, t in enumerate(ts):
            im = Image.open(os.path.join(d, 'f_%04d.png' % k))
            post(np.asarray(im), t, 360, 640).save(os.path.join(d, 'g_%04d.png' % k))
        sheet([os.path.join(d, 'g_%04d.png' % k) for k in range(len(ts))], os.path.join(HERE, 'fp5_test.png'), 5, 360, 640)
        print('test sheet written')
    else:
        out = os.path.join(HERE, 'fp5_frames')
        a, b_ = (int(x) for x in sys.argv[2].split(':')) if len(sys.argv) > 2 else (0, NF)
        vs = [(k / FPS, view_of(frames[k], bl[k], W, H)) for k in range(a, b_)]
        run(vs, out, True, names=['f_%04d.png' % k for k in range(a, b_)])
        print('frames %d..%d done' % (a, b_))
