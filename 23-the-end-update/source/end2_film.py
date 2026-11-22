"""Render the End Update (v2): 3D frames from web/end2.html; the phone, the caption, the title and the last notification
drawn in 2D.   python3 end2_film.py test t1,t2,... [WxH] [sheet] | render a:b [WxH]   -> frames/f_%05d.jpg"""
import os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import end2_anim as A

HERE = os.path.dirname(os.path.abspath(__file__))
FD = '/usr/share/fonts/opentype/inter/'
_fc = {}
def font(name, size):
    k = (name, int(size))
    if k not in _fc: _fc[k] = ImageFont.truetype(FD + name, int(size))
    return _fc[k]


def ease(x): x = min(max(x, 0.0), 1.0); return x * x * (3 - 2 * x)
def spring(x):
    """0 -> 1 with a little overshoot"""
    x = min(max(x, 0.0), 1.0); return 1 - np.exp(-6 * x) * np.cos(9 * x)


def rrect(d, box, r, fill, outline=None, width=1):
    d.rounded_rectangle(box, r, fill=fill, outline=outline, width=width)


# ---------------------------------------------------------------- the phone
def wallpaper(W, H, v):
    y = np.linspace(0, 1, H)[:, None]; x = np.linspace(0, 1, W)[None, :]
    Y = y + 0 * x; X = x + 0 * y
    base = np.stack([10 + 22 * Y + 10 * X, 5 + 6 * Y, 22 + 46 * Y + 8 * X], -1)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), 'RGB').convert('RGBA')
    ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov); rng = np.random.default_rng(3)
    for i in range(16):                                           # soft violet bokeh drifting up
        cx = rng.uniform(0, W); cy = (rng.uniform(0, H) - v * rng.uniform(4, 14) * W / 720) % H; r = rng.uniform(30, 120) * W / 720
        a = int(rng.uniform(20, 55)); col = (int(rng.uniform(120, 200)), int(rng.uniform(40, 90)), 255, a)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
    ov = ov.filter(ImageFilter.GaussianBlur(28 * W / 720))
    return Image.alpha_composite(img, ov)


def icon_game(sz):
    im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    g = np.linspace(0, 1, sz)[:, None] * np.ones((1, sz))
    grad = np.stack([255 - 40 * g, 90 + 40 * (1 - g), 120 + 100 * g, np.full_like(g, 255)], -1).astype(np.uint8)
    mask = Image.new('L', (sz, sz), 0); ImageDraw.Draw(mask).rounded_rectangle([0, 0, sz - 1, sz - 1], sz // 4, fill=255)
    im.paste(Image.fromarray(grad, 'RGBA'), (0, 0), mask)
    s = sz / 64; c = (255, 255, 255, 235)                        # a little game controller
    d.rounded_rectangle([12 * s, 24 * s, 52 * s, 42 * s], 9 * s, fill=c)
    d.ellipse([10 * s, 30 * s, 24 * s, 48 * s], fill=c); d.ellipse([40 * s, 30 * s, 54 * s, 48 * s], fill=c)
    k = (200, 70, 140, 255); d.rectangle([19 * s, 30 * s, 21 * s, 38 * s], fill=k); d.rectangle([16 * s, 33 * s, 24 * s, 35 * s], fill=k)
    d.ellipse([41 * s, 29 * s, 45 * s, 33 * s], fill=k); d.ellipse([45 * s, 33 * s, 49 * s, 37 * s], fill=k)
    return im


def icon_eye(sz):
    """a pixel-art eye of the End (original 8 x 8), on a dark tile"""
    P = ['........', '..aaaa..', '.abbbba.', 'abccccba', 'abcddcba', '.abbbba.', '..aaaa..', '........']
    C = {'a': (24, 74, 70), 'b': (52, 150, 120), 'c': (120, 220, 170), 'd': (10, 30, 28)}
    im = Image.new('RGBA', (sz, sz), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, sz - 1, sz - 1], sz // 4, fill=(20, 14, 34, 255))
    s = sz * 0.8 / 8; o = sz * 0.1
    for j, row in enumerate(P):
        for i, ch in enumerate(row):
            if ch in C: d.rectangle([o + i * s, o + j * s, o + (i + 1) * s - 1, o + (j + 1) * s - 1], fill=C[ch] + (255,))
    return im


def card(W, app, title, body, icon, when='now'):
    w = int(W * 0.92); h = int(W * 0.235); im = Image.new('RGBA', (w, h), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    rrect(d, [0, 0, w - 1, h - 1], int(W * 0.05), (245, 240, 255, 58), outline=(255, 255, 255, 40), width=1)
    isz = int(W * 0.105); im.alpha_composite(icon.resize((isz, isz), Image.LANCZOS), (int(W * 0.035), int(h / 2 - isz / 2)))
    x0 = int(W * 0.035 + isz + W * 0.03); f1 = font('Inter-SemiBold.otf', W * 0.034); f2 = font('Inter-Regular.otf', W * 0.031)
    d.text((x0, int(h * 0.17)), app, font=f1, fill=(255, 255, 255, 235))
    tw = d.textlength(when, font=f2); d.text((w - W * 0.04 - tw, int(h * 0.17)), when, font=f2, fill=(255, 255, 255, 150))
    d.text((x0, int(h * 0.40)), title, font=font('Inter-SemiBold.otf', W * 0.032), fill=(255, 255, 255, 245))
    d.text((x0, int(h * 0.64)), body, font=f2, fill=(255, 255, 255, 190))
    return im


CARD_CACHE = {}
def cards(W):
    if W not in CARD_CACHE:
        CARD_CACHE[W] = (card(W, 'GTA VI', "It's finally here.", 'Grand Theft Auto VI is out now.', icon_game(128)),
                         card(W, 'Minecraft', 'The End Update is out now', 'The End has been rebuilt. Come and see.', icon_eye(128)),
                         card(W, 'GTA VI', 'Your game is ready to play.', 'Tap to start Grand Theft Auto VI.', icon_game(128), when='2m ago'))
    return CARD_CACHE[W]


def phone(W, H, p):
    v = p['v']; img = wallpaper(W, H, v); d = ImageDraw.Draw(img)
    c1, c2, _ = cards(W)
    # the clock rolls over to midnight
    tstr = '11:59' if v < 0.10 else '12:00'
    fb = font('InterDisplay-SemiBold.otf', W * 0.235); fd = font('Inter-Medium.otf', W * 0.042)
    ds = 'Friday, 20 November'; d.text(((W - d.textlength(ds, font=fd)) / 2, H * 0.115), ds, font=fd, fill=(255, 255, 255, 215))
    d.text(((W - d.textlength(tstr, font=fb)) / 2, H * 0.135), tstr, font=fb, fill=(255, 255, 255, 240))
    fs = font('Inter-SemiBold.otf', W * 0.035); d.text((W * 0.07, H * 0.018), tstr, font=fs, fill=(255, 255, 255, 230))
    for i in range(4): d.rectangle([W * (0.80 + i * 0.016), H * (0.034 - 0.004 * i), W * (0.81 + i * 0.016), H * 0.04], fill=(255, 255, 255, 230))
    rrect(d, [W * 0.875, H * 0.024, W * 0.935, H * 0.041], int(W * 0.008), None, outline=(255, 255, 255, 200), width=2)
    d.rectangle([W * 0.879, H * 0.028, W * 0.92, H * 0.037], fill=(255, 255, 255, 230))
    # notifications: GTA at midnight, then the End Update lands on top of it
    y_slot = H * 0.42; gap = c1.height + W * 0.025
    k1 = spring((v - 0.10) / 0.45); k2 = spring((v - p['n2']) / 0.45)
    y1 = -c1.height + (y_slot + c1.height) * k1 + gap * min(k2, 1.0) * (1 if v >= p['n2'] else 0)
    x0 = int((W - c1.width) / 2)
    if v >= 0.10:
        a = c1 if v < p['n2'] else Image.blend(c1, c1, 1)
        img.alpha_composite(c1, (x0, int(y1)))
    if v >= p['n2']:
        y2 = -c2.height + (y_slot + c2.height) * k2
        sc = 1.0
        if v >= p['tap']: sc = 1 - 0.04 * np.sin(min(1.0, (v - p['tap']) / 0.25) * np.pi)
        cc = c2.resize((int(c2.width * sc), int(c2.height * sc)), Image.LANCZOS) if sc != 1 else c2
        img.alpha_composite(cc, (int((W - cc.width) / 2), int(y2 + (c2.height - cc.height) / 2)))
        if v >= p['tap']:                                         # the tap
            a = (v - p['tap']) / 0.45; r = W * 0.03 + W * 0.12 * ease(a)
            ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); od = ImageDraw.Draw(ov)
            cx, cy = W * 0.6, y2 + c2.height * 0.55
            od.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255, int(110 * max(0, 1 - a))))
            img = Image.alpha_composite(img, ov)
    # haptics: a small shake as each one lands; a slow push in; the zoom into the card as it opens
    jit = 0
    for t in (0.10, p['n2']):
        if 0 <= v - t < 0.18: jit = int(W * 0.004 * np.sin((v - t) * 120))
    z = 1.0 + 0.04 * v / p['dive']
    if v >= p['tap'] + 0.3: z += 1.6 * ease((v - p['tap'] - 0.3) / (p['dive'] - p['tap'] - 0.3)) ** 2
    cw, ch = W / z, H / z; cx, cy = W * 0.5 + jit, (H * 0.5 if v < p['tap'] else H * 0.5 + (H * 0.42 + c2.height * 0.5 - H * 0.5) * ease((v - p['tap']) / 0.6))
    img = img.crop((int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))).resize((W, H), Image.LANCZOS)
    if v >= p['dive'] - 0.15:
        wf = ease((v - (p['dive'] - 0.15)) / 0.15); img = Image.blend(img.convert('RGB'), Image.new('RGB', (W, H), (255, 250, 255)), wf).convert('RGBA')
    if v < 0.06: img = Image.blend(Image.new('RGBA', (W, H), (0, 0, 0, 255)), img, v / 0.06)
    return img.convert('RGB')


def notif(W, H, p):
    img = Image.new('RGBA', (W, H), (0, 0, 0, 255)); v = p['v']; c = cards(W)[2]
    if v < p['t0']: return img.convert('RGB')
    k = spring((v - p['t0']) / 0.45); y = -c.height + (H * 0.06 + c.height) * k
    if v >= p['t1']: y -= (H * 0.06 + c.height) * ease((v - p['t1']) / 0.3)        # swiped away
    img.alpha_composite(c, (int((W - c.width) / 2), int(y)))
    return img.convert('RGB')


# ---------------------------------------------------------------- the caption at the tap
def caption(img, C, W, H):
    if not C: return img
    v = C['v']; age = v - C['t0']
    if age < 0: return img
    pop = float(spring(age / 0.22)); sc = 0.55 + 0.45 * pop
    fs = W * 0.082 * sc; f = font('InterDisplay-ExtraBold.otf', fs)
    words = ['The', 'End', 'Update']; ov = Image.new('RGBA', (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(ov)
    sp = d.textlength(' ', font=f); ws = [d.textlength(w, font=f) for w in words]; tot = sum(ws) + sp * 2
    x = (W - tot) / 2; y = H * 0.80 - fs / 2
    for i, (w, ww) in enumerate(zip(words, ws)):
        col = (255, 255, 255, 255) if i != 1 else (205, 150, 255, 255)
        d.text((x, y), w, font=f, fill=col, stroke_width=max(3, int(fs * 0.09)), stroke_fill=(8, 0, 18, 255)); x += ww + sp
    glow = ov.filter(ImageFilter.GaussianBlur(float(fs * 0.22))); ga = np.array(glow); ga[..., :3] = [150, 70, 255]; ga[..., 3] = (ga[..., 3] * 0.55).astype(np.uint8)
    img = Image.alpha_composite(img.convert('RGBA'), Image.fromarray(ga)); img = Image.alpha_composite(img, ov)
    return img.convert('RGB')


# ---------------------------------------------------------------- the title: blocky extruded letters (an original pixel face)
GLYPH = {'T': ['#####', '..#..', '..#..', '..#..', '..#..', '..#..', '..#..'], 'H': ['#...#', '#...#', '#...#', '#####', '#...#', '#...#', '#...#'],
         'E': ['#####', '#....', '#....', '####.', '#....', '#....', '#####'], 'N': ['#...#', '##..#', '#.#.#', '#..##', '#...#', '#...#', '#...#'],
         'D': ['####.', '#...#', '#...#', '#...#', '#...#', '#...#', '####.'], 'U': ['#...#', '#...#', '#...#', '#...#', '#...#', '#...#', '.###.'],
         'P': ['####.', '#...#', '#...#', '####.', '#....', '#....', '#....'], 'A': ['.###.', '#...#', '#...#', '#####', '#...#', '#...#', '#...#']}


def word_img(text, b, top=(246, 232, 255), bot=(160, 96, 236), seed=0):
    cells = []
    for i, ch in enumerate(text):
        for r_, row in enumerate(GLYPH[ch]):
            for c_, q in enumerate(row):
                if q == '#': cells.append((i * 6 + c_, r_))
    cw = len(text) * 6 - 1; dep = int(b * 0.75); pad = int(b * 0.5)
    Wd, Hd = int(cw * b) + dep + 2 * pad, int(7 * b) + dep + 2 * pad
    im = Image.new('RGBA', (Wd, Hd), (0, 0, 0, 0)); d = ImageDraw.Draw(im); rng = np.random.default_rng(seed)
    ol = max(2, int(b * 0.12))
    for (cx, cy) in cells:                                   # the outline, around face and depth
        x0, y0 = pad + cx * b, pad + cy * b; d.rectangle([x0 - ol, y0 - ol, x0 + b + ol + dep * 0.45, y0 + b + ol + dep], fill=(10, 3, 20, 255))
    for k in range(dep, 0, -1):                              # the extrusion, down and a little right
        t = k / dep; colr = (int(70 - 45 * t), int(30 - 22 * t), int(118 - 70 * t), 255)
        for (cx, cy) in cells:
            x0, y0 = pad + cx * b + k * 0.45, pad + cy * b + k; d.rectangle([x0, y0, x0 + b - 1, y0 + b - 1], fill=colr)
    for (cx, cy) in cells:                                   # the faces: a gradient down the letter, each block bevelled
        x0, y0 = pad + cx * b, pad + cy * b; t = (cy + 0.5) / 7; j = 1 + rng.uniform(-0.05, 0.05)
        col = tuple(int(min(255, (top[i] * (1 - t) + bot[i] * t) * j)) for i in range(3)) + (255,)
        d.rectangle([x0, y0, x0 + b - 1, y0 + b - 1], fill=col)
        e = max(1, int(b * 0.12))
        d.rectangle([x0, y0, x0 + b - 1, y0 + e - 1], fill=tuple(min(255, int(c * 1.12) + 10) for c in col[:3]) + (255,))
        d.rectangle([x0, y0 + b - e, x0 + b - 1, y0 + b - 1], fill=tuple(int(c * 0.78) for c in col[:3]) + (255,))
        d.rectangle([x0 + b - e, y0, x0 + b - 1, y0 + b - 1], fill=tuple(int(c * 0.86) for c in col[:3]) + (255,))
    return im


_TITLE = {}
def title_img(W):
    if W not in _TITLE:
        a = word_img('THE', int(W * 0.016), seed=1); bimg = word_img('END', int(W * 0.040), seed=2); c = word_img('UPDATE', int(W * 0.0195), seed=3)
        gap = int(W * 0.012); Wt = max(a.width, bimg.width, c.width); Ht = a.height + bimg.height + c.height + 2 * gap
        T = Image.new('RGBA', (Wt, Ht), (0, 0, 0, 0)); y = 0
        for im in (a, bimg, c): T.alpha_composite(im, ((Wt - im.width) // 2, y)); y += im.height + gap
        g = T.filter(ImageFilter.GaussianBlur(float(W * 0.03))); ga = np.array(g); ga[..., :3] = [165, 80, 255]
        _TITLE[W] = (T, Image.fromarray(ga))
    return _TITLE[W]


def title(img, P, W, H):
    if not P: return img
    v = P['v']; T, G = title_img(W)
    age = v - P['t0']; sc = 1.0 + 0.16 * np.exp(-age / 0.10) + 0.035 * min(age / 4.0, 1.0)
    gl = 0.55
    for t in P['beats']:
        d = v - t
        if 0 <= d < 0.6: sc += 0.06 * np.exp(-d / 0.11); gl += 0.6 * np.exp(-d / 0.15)
    a = min(1.0, age / 0.06)
    jx = jy = 0
    if v >= P['tape']:
        k = (v - P['tape']) / (P['sil'] - P['tape']); a *= 1 - ease(k); sc *= 1 - 0.05 * k
        jx = int(W * 0.01 * np.sin(v * 90) * k); jy = int(W * 0.006 * k * k * 10)
    if a <= 0.01: return img
    tw, th = int(T.width * sc), int(T.height * sc); Ti = T.resize((tw, th), Image.LANCZOS); Gi = G.resize((tw, th), Image.BILINEAR)
    x = (W - tw) // 2 + jx; y = int(H * 0.47 - th / 2) + jy
    base = img.convert('RGBA'); lay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    ga = np.array(Gi); ga[..., 3] = np.clip(ga[..., 3] * gl * a, 0, 255).astype(np.uint8); lay.alpha_composite(Image.fromarray(ga), (x, y))
    base = Image.alpha_composite(base, lay)
    Ta = np.array(Ti); Ta[..., 3] = (Ta[..., 3] * a).astype(np.uint8)
    if v >= P['tape']:                                       # the tape stop tears the colour apart
        k = (v - P['tape']) / (P['sil'] - P['tape']); off = int(W * 0.012 * k)
        r_ = Ta.copy(); r_[..., 1:3] = 0; bl = Ta.copy(); bl[..., 0:2] = 0
        l2 = Image.new('RGBA', (W, H), (0, 0, 0, 0)); l2.alpha_composite(Image.fromarray(r_), (x - off, y)); l2.alpha_composite(Image.fromarray(bl), (x + off, y))
        base = Image.alpha_composite(base, l2)
    l3 = Image.new('RGBA', (W, H), (0, 0, 0, 0)); l3.alpha_composite(Image.fromarray(Ta), (x, y))
    return Image.alpha_composite(base, l3).convert('RGB')


def render(ks, out_dir, w, h, label=False):
    os.makedirs(out_dir, exist_ok=True)
    need3d = any(A.frame(k)['render'] for k in ks)
    pw = b = pg = None
    if need3d:
        from playwright.sync_api import sync_playwright
        import end2_view as V
        V.ensure_server(); pw = sync_playwright().start(); b, pg = V.open_page(pw)
    t0 = time.time()
    for i, k in enumerate(ks):
        f = A.frame(k)
        if f['render']:
            r = dict(f['render']); r['w'] = w; r['h'] = h; im = V.shot(pg, r)
        elif f['phone']: im = phone(w, h, f['phone'])
        elif f['notif']: im = notif(w, h, f['notif'])
        else: im = Image.new('RGB', (w, h), (0, 0, 0))
        im = caption(im, f.get('caption'), w, h); im = title(im, f.get('title'), w, h)
        if label: ImageDraw.Draw(im).text((6, 6), 'v %.2f' % (k / A.FPS), fill=(255, 255, 0))
        im.save(os.path.join(out_dir, 'f_%05d.jpg' % k), quality=95) if out_dir.endswith('frames') else im.save(os.path.join(out_dir, 'f_%05d.png' % k))
        if i % 60 == 0: print('frame %d (%d/%d) %.0fs' % (k, i, len(ks), time.time() - t0), flush=True)
    if b: b.close(); pw.stop()


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        ts = [float(x) for x in sys.argv[2].split(',')]
        w, h = (int(x) for x in sys.argv[3].split('x')) if len(sys.argv) > 3 else (360, 640)
        ks = [int(round(t * A.FPS)) for t in ts]
        d = os.path.join(HERE, 'test'); render(ks, d, w, h, label=True)
        cols = 6; rows = (len(ks) + cols - 1) // cols; W_, H_ = 300, 533
        s = Image.new('RGB', (cols * W_, rows * H_))
        for i, k in enumerate(ks): s.paste(Image.open(os.path.join(d, 'f_%05d.png' % k)).resize((W_, H_)), ((i % cols) * W_, (i // cols) * H_))
        out = sys.argv[4] if len(sys.argv) > 4 else 'test_sheet.png'; s.save(os.path.join(HERE, out)); print('sheet', out)
    else:
        a, b_ = (int(x) for x in sys.argv[2].split(':'))
        w, h = (int(x) for x in sys.argv[3].split('x')) if len(sys.argv) > 3 else (720, 1280)
        render(list(range(a, b_)), os.path.join(HERE, 'frames'), w, h)
        print('frames %d..%d done' % (a, b_))
