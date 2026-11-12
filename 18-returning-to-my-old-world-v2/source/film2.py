"""v2 film: the beta-style menu and loading screens, then the shots (rendered through web/ow2.html, crossfaded), a
cinematic grade (soft bloom, warmth, lifted shadows, vignette, fine grain) and kinetic serif typography in the style
of a lyric edit.  python3 film2.py test [t,...] | render a:b"""
import math, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from playwright.sync_api import sync_playwright
import ow2_view as V
import anim2 as A
import pixfont
import tex2 as TX

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
CINZEL = os.path.join(HERE, 'fonts', 'Cinzel.ttf')
_FONTS = {}


def cinzel(size, weight='Bold'):
    key = (size, weight)
    if key not in _FONTS:
        f = ImageFont.truetype(CINZEL, size); f.set_variation_by_name(weight); _FONTS[key] = f
    return _FONTS[key]


# ------------------------------------------------------------------ the menu, in the old style
def dirt_bg(w, h, s, dark=0.25):
    t = TX.tile('dirt')[..., :3] * dark
    big = np.repeat(np.repeat(t, s, 0), s, 1)
    return np.tile(big, (h // big.shape[0] + 1, w // big.shape[1] + 1, 1))[:h, :w].astype(np.float32)


def ptext(img, s, x, y, scale, col, center=False):
    m = pixfont.text_mask(s); m = np.repeat(np.repeat(m, scale, 0), scale, 1); h, w = m.shape
    if center:
        x = int(x - w / 2)
    for dx, dy, c in ((scale, scale, np.array(col) * 0.25), (0, 0, np.array(col))):
        reg = img[y + dy:y + dy + h, x + dx:x + dx + w]; reg[m[:reg.shape[0], :reg.shape[1]]] = c
    return w


def button(img, x, y, w, h, label, scale, state):
    base = {0: 112, 1: 128, 2: 92}[state]
    img[y:y + h, x:x + w] = 0
    inner = img[y + scale:y + h - scale, x + scale:x + w - scale]
    inner[:] = np.linspace(base * 1.15, base * 0.85, inner.shape[0])[:, None, None]
    if state == 1:
        for sl in (np.s_[y:y + scale, x:x + w], np.s_[y + h - scale:y + h, x:x + w], np.s_[y:y + h, x:x + scale], np.s_[y:y + h, x + w - scale:x + w]):
            img[sl] = (190, 200, 255)
    m = pixfont.text_mask(label); tw = m.shape[1] * scale
    ptext(img, label, x + (w - tw) // 2, y + (h - 7 * scale) // 2, scale, (255, 255, 160) if state == 1 else (224, 224, 224))


def cursor(img, x, y, s=5):
    shape = ["#          ", "##         ", "#o#        ", "#oo#       ", "#ooo#      ", "#oooo#     ", "#ooooo#    ", "#oooooo#   ",
             "#ooooooo#  ", "#oooo#####", "#oo#oo#   ", "#o# #oo#  ", "##  #oo#  ", "     #oo# ", "      ##  "]
    for j, row in enumerate(shape):
        for i, ch in enumerate(row):
            if ch != ' ':
                img[y + j * s:y + (j + 1) * s, x + i * s:x + (i + 1) * s] = (0, 0, 0) if ch == '#' else (255, 255, 255)


def menu_frame(t):
    S = 6
    img = dirt_bg(W, H, S)
    if t < 1.1:
        ptext(img, 'Select world', W // 2, 300, S, (255, 255, 255), center=True)
        img[400:1300] *= 0.4
        y = 450
        for name, l2, sel in (('My First World', 'my first world (11/11/16 4:12 PM)', True), ('New World', 'new world (2/6/17 7:40 PM)', False),
                              ('creative', 'creative (8/22/16 1:03 PM)', False)):
            if sel:
                img[y - 14:y + 132, 90:990] = (128, 128, 128); img[y - 8:y + 126, 96:984] = 0
            ptext(img, name, 130, y + 10, 5, (255, 255, 255))
            ptext(img, l2, 130, y + 68, 4, (128, 128, 128))
            y += 190
        cx, cy = 300, 1530
        u = A.smooth(t, 0.25, 0.92)
        mx = 830 + (cx - 830) * u + 30 * math.sin(u * math.pi); my = 1220 + (cy - 1220) * u
        hov = abs(mx - cx) < 240 and abs(my - cy) < 50
        button(img, 60, 1480, 480, 100, 'Play selected world', 4, 2 if 0.98 <= t < 1.08 else (1 if hov else 0))
        button(img, 560, 1480, 460, 100, 'Create new world', 4, 0)
        button(img, 60, 1600, 300, 100, 'Rename', 4, 0); button(img, 380, 1600, 300, 100, 'Delete', 4, 0); button(img, 700, 1600, 320, 100, 'Cancel', 4, 0)
        cursor(img, int(mx), int(my))
    elif t < 2.2:
        ptext(img, 'Loading level', W // 2, 820, S, (255, 255, 255), center=True)
        ptext(img, 'Building terrain', W // 2, 920, 5, (255, 255, 255), center=True)
        p = A.smooth(t, 1.2, 2.1); x0, x1, y0 = 240, 840, 1010
        img[y0:y0 + 14, x0:x1] = (128, 128, 128); img[y0:y0 + 14, x0:x0 + int((x1 - x0) * p)] = (128, 255, 128)
    else:
        img[:] = 0
    img *= A.smooth(t, 0.0, 0.15)
    return img / 255.0


# ------------------------------------------------------------------ the grade
_VIG = None


def grade(img, k):
    global _VIG
    h, w = img.shape[:2]
    lum = img @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    # bloom from the bright parts (sky, sun, the doorway), soft and warm
    br = np.clip((img - 0.66) / 0.34, 0, 1) ** 1.5
    small = Image.fromarray((br * 255).astype(np.uint8)).resize((w // 4, h // 4), Image.BILINEAR)
    b1 = np.asarray(small.filter(ImageFilter.GaussianBlur(5)).resize((w, h), Image.BILINEAR), np.float32) / 255
    b2 = np.asarray(small.filter(ImageFilter.GaussianBlur(18)).resize((w, h), Image.BILINEAR), np.float32) / 255
    img = img + (0.30 * b1 + 0.28 * b2) * np.array([1.0, 0.86, 0.70], np.float32)
    # warmth, lifted violet shadows, a gentle S-curve, a little more colour
    img = img * np.array([1.035, 1.0, 0.95], np.float32) + np.array([0.030, 0.018, 0.050], np.float32) * ((1 - lum) ** 2)[..., None]
    img = np.clip(img, 0, 1.2)
    img = 0.82 * img + 0.18 * (img * img * (3 - 2 * np.clip(img, 0, 1)))
    l2 = (img @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None]
    img = l2 + (img - l2) * 1.07
    if _VIG is None or _VIG.shape[:2] != (h, w):
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot((xx - w / 2) / (w * 0.6), (yy - h / 2) / (h * 0.6))
        _VIG = (1 - 0.32 * np.clip(d, 0, 1.2) ** 2.4)[..., None].astype(np.float32)
    img = img * _VIG
    g = np.random.default_rng(k).standard_normal((h, w), dtype=np.float32) * 0.011
    return img + g[..., None]


# ------------------------------------------------------------------ kinetic typography
def text_layer(phrase, t, w, h):
    """RGBA layer (float) with the phrase's words as they land, plus their glow"""
    sc = w / W
    size = int(phrase['size'] * sc)
    f = cinzel(size, 'Bold')
    end = phrase['end']; t_first = phrase['times'][0][0]
    out_k = A.smooth(t, end - 0.4, end)
    life = A.smooth(t, t_first, end)
    layer = Image.new('L', (w, h), 0)
    d = ImageDraw.Draw(layer)
    nlines = len(phrase['lines'])
    lh = size * 1.12
    y0 = h * phrase['y'] - lh * nlines / 2
    for li, (words, times) in enumerate(zip(phrase['lines'], phrase['times'])):
        line = ' '.join(words)
        lw = d.textlength(line, font=f)
        x = (w - lw) / 2
        for word, ta in zip(words, times):
            ww = d.textlength(word, font=f)
            if t >= ta:
                e = A.smooth(t, ta, ta + 0.32)
                s = 1 + 0.42 * (1 - e) + 0.06 * out_k
                a = int(255 * e * (1 - out_k))
                # each word lands from a little larger and above, settling into place
                wi = Image.new('L', (int(ww * 1.6) + 20, int(size * 2.0)), 0)
                ImageDraw.Draw(wi).text((int(ww * 0.3) + 10, int(size * 0.35)), word, font=f, fill=a)
                if s != 1:
                    wi = wi.resize((max(1, int(wi.width * s)), max(1, int(wi.height * s))), Image.BICUBIC)
                if e < 0.999:
                    wi = wi.filter(ImageFilter.GaussianBlur(4 * sc * (1 - e)))
                cx = x + ww / 2; cy = y0 + li * lh + size * 0.55 - 14 * sc * (1 - e)
                layer.paste(wi, (int(cx - wi.width / 2), int(cy - wi.height / 2 + size * 0.15)), wi)
            x += ww + d.textlength(' ', font=f)
    # a slow push and a slight turn in depth over the phrase's life
    ang = math.radians(-9 + 7 * life)
    push = 1.0 + 0.05 * life
    layer = perspective(layer, ang, push)
    m = np.asarray(layer, np.float32) / 255
    if m.max() <= 0:
        return None
    small = layer.resize((w // 2, h // 2), Image.BILINEAR)
    g1 = np.asarray(small.filter(ImageFilter.GaussianBlur(9 * sc)).resize((w, h), Image.BILINEAR), np.float32) / 255
    g2 = np.asarray(small.filter(ImageFilter.GaussianBlur(28 * sc)).resize((w, h), Image.BILINEAR), np.float32) / 255
    sh = np.asarray(small.filter(ImageFilter.GaussianBlur(6 * sc)).resize((w, h), Image.BILINEAR), np.float32) / 255
    return m, g1, g2, sh


def perspective(img, ang, push):
    """turn a flat layer about the vertical axis through its centre (a cheap 3D tilt) and push it in a little"""
    w, h = img.size
    f = w * 1.4
    def proj(x, y):
        X = (x - w / 2) * push; Y = (y - h / 2) * push
        xr = X * math.cos(ang); zr = X * math.sin(ang)
        s = f / (f + zr)
        return (w / 2 + xr * s, h / 2 + Y * s)
    src = [(0, 0), (w, 0), (w, h), (0, h)]
    dst = [proj(*p) for p in src]
    # coefficients mapping output (dst) to input (src)
    A_ = []; B_ = []
    for (x, y), (u, v) in zip(dst, src):
        A_.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); B_.append(u)
        A_.append([0, 0, 0, x, y, 1, -v * x, -v * y]); B_.append(v)
    coef = np.linalg.solve(np.array(A_, float), np.array(B_, float))
    return img.transform((w, h), Image.PERSPECTIVE, tuple(coef), Image.BICUBIC)


def typography(img, t):
    h, w = img.shape[:2]
    for ph in A.PHRASES:
        if ph['times'][0][0] <= t <= ph['end']:
            r = text_layer(ph, t, w, h)
            if r is None:
                continue
            m, g1, g2, sh = r
            img = img * (1 - 0.45 * sh[..., None])                                   # a soft shadow under the words
            img = img + g2[..., None] * np.array([1.0, 0.55, 0.22], np.float32) * 0.55   # wide warm glow
            img = img + g1[..., None] * np.array([1.0, 0.78, 0.45], np.float32) * 0.65   # tight glow
            grad = (1 - ((np.arange(h) / h - ph['y']) * 3).clip(-0.5, 0.5)[:, None, None] * 0.25)
            col = np.array([1.0, 0.95, 0.84], np.float32) * grad
            img = img * (1 - m[..., None]) + col * m[..., None]
    o = A.OUTRO
    if o['t0'] <= t <= o['t1']:
        a = A.smooth(t, o['t0'], o['t0'] + 0.8) * (1 - A.smooth(t, o['t1'] - 0.6, o['t1']))
        sc = w / W
        f = cinzel(int(44 * sc), 'Regular')
        lay = Image.new('L', (w, h), 0); d = ImageDraw.Draw(lay)
        tw = d.textlength(o['text'], font=f)
        d.text(((w - tw) / 2, h * 0.70), o['text'], font=f, fill=int(255 * a))
        m = np.asarray(lay, np.float32)[..., None] / 255
        gl = np.asarray(lay.filter(ImageFilter.GaussianBlur(8 * sc)), np.float32)[..., None] / 255
        img = img + gl * np.array([1.0, 0.75, 0.45], np.float32) * 0.5
        img = img * (1 - m) + np.array([1.0, 0.95, 0.86], np.float32) * m
    return img


def finish(img, t, k):
    """img float 0..1 (world or menu) -> graded, titled, faded uint8"""
    if t >= A.T_WORLD:
        img = grade(img, k)
        img = img * A.smooth(t, A.T_WORLD, A.T_WORLD + 0.45)
        img = img * (1 - A.smooth(t, 29.0, 29.95))
    img = typography(img, t)
    return Image.fromarray((np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8))


def render_frames(ks, out_dir, w=W, h=H):
    os.makedirs(out_dir, exist_ok=True)
    world = [k for k in ks if k / A.FPS >= A.T_WORLD]
    for k in ks:
        if k / A.FPS < A.T_WORLD:
            im = menu_frame(k / A.FPS)
            if w != W:
                im = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((w, h), Image.NEAREST), np.float32) / 255
            finish(im, k / A.FPS, k).save(os.path.join(out_dir, 'f_%05d.png' % k))
    if not world:
        return
    V.ensure_server()
    with sync_playwright() as pw:
        b, pg = V.open_page(pw)
        t0 = time.time()
        for i, k in enumerate(world):
            acc = None
            for v, wt in A.views(k):
                v = dict(v); v['w'] = w; v['h'] = h
                im = np.asarray(V.shot(pg, v), np.float32) / 255
                acc = im * wt if acc is None else acc + im * wt
            finish(acc, k / A.FPS, k).save(os.path.join(out_dir, 'f_%05d.png' % k))
            if i % 60 == 0:
                print('frame %d (%d/%d) %.0fs' % (k, i, len(world), time.time() - t0), flush=True)
        b.close()


def sheet(paths, out, cols, w, h):
    rows = (len(paths) + cols - 1) // cols
    s = Image.new('RGB', (cols * w, rows * h))
    for i, p in enumerate(paths):
        s.paste(Image.open(p).resize((w, h)), ((i % cols) * w, (i // cols) * h))
    s.save(out)


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if mode == 'test':
        ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else \
            [0.5, 1.6, 3.2, 5.0, 6.4, 7.3, 9.0, 10.2, 12.9, 13.6, 15.0, 16.2, 18.0, 19.2, 20.6, 22.2, 23.6, 24.8, 27.0, 28.6]
        ks = [int(round(t * A.FPS)) for t in ts]
        d = os.path.join(HERE, 'test2'); render_frames(ks, d, 540, 960)
        sheet([os.path.join(d, 'f_%05d.png' % k) for k in ks], os.path.join(HERE, 'test2_sheet.png'), 5, 360, 640)
        print('test sheet written')
    else:
        a, b_ = (int(x) for x in sys.argv[2].split(':'))
        render_frames(list(range(a, b_)), os.path.join(HERE, 'frames2'))
        print('frames %d..%d done' % (a, b_))
