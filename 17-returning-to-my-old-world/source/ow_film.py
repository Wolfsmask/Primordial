"""The old world video: the world-select and loading screens, then the world through web/ow.html, with captions,
a soft classic vignette and the fades.  python3 ow_film.py test [t,t,...] | render a:b | thumb"""
import math, os, sys, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
import ow_view as V
import ow_anim as A
import pixfont
import tex_ow as TX

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920
FONT_P = '/usr/share/fonts/opentype/inter/Inter-Bold.otf'
CAPTIONS = [(0.12, 6.2, 'opening my first world\nafter 10 years…'), (20.4, 23.6, 'he never stopped waiting.')]


def font(sz):
    return ImageFont.truetype(FONT_P, sz)


# ------------------------------------------------------------------ menus (drawn in the game's pixel style)
def dirt_bg(w, h, s, dark=0.27):
    t = TX.tile('dirt')[..., :3] * dark
    big = np.repeat(np.repeat(t, s, 0), s, 1)
    reps = (h // big.shape[0] + 1, w // big.shape[1] + 1, 1)
    return np.tile(big, reps)[:h, :w].astype(np.float32)


def ptext(img, s, x, y, scale, col, shadow=True, center=False):
    m = pixfont.text_mask(s)
    m = np.repeat(np.repeat(m, scale, 0), scale, 1)
    h, w = m.shape
    if center:
        x = int(x - w / 2)
    for (dx, dy, c) in (((scale, scale, np.array(col) * 0.25),) if shadow else ()) + ((0, 0, np.array(col)),):
        reg = img[y + dy:y + dy + h, x + dx:x + dx + w]
        mm = m[:reg.shape[0], :reg.shape[1]]
        reg[mm] = c
    return w


def button(img, x, y, w, h, label, scale, state):
    """state 0 normal, 1 hovered, 2 pressed"""
    base = {0: 112, 1: 128, 2: 92}[state]
    img[y:y + h, x:x + w] = 0
    inner = img[y + scale:y + h - scale, x + scale:x + w - scale]
    grad = np.linspace(base * 1.15, base * 0.85, inner.shape[0])[:, None, None]
    inner[:] = grad
    if state == 1:
        img[y:y + scale, x:x + w] = [190, 200, 255]; img[y + h - scale:y + h, x:x + w] = [190, 200, 255]
        img[y:y + h, x:x + scale] = [190, 200, 255]; img[y:y + h, x + w - scale:x + w] = [190, 200, 255]
    col = (255, 255, 160) if state == 1 else (224, 224, 224)
    m = pixfont.text_mask(label); tw = m.shape[1] * scale; th = 7 * scale
    ptext(img, label, x + (w - tw) // 2, y + (h - th) // 2, scale, col)


def cursor(img, x, y, s=5):
    shape = ["#          ", "##         ", "#o#        ", "#oo#       ", "#ooo#      ", "#oooo#     ", "#ooooo#    ", "#oooooo#   ",
             "#ooooooo#  ", "#oooo#####", "#oo#oo#   ", "#o# #oo#  ", "##  #oo#  ", "     #oo# ", "      ##  "]
    for j, row in enumerate(shape):
        for i, ch in enumerate(row):
            if ch == ' ':
                continue
            img[y + j * s:y + (j + 1) * s, x + i * s:x + (i + 1) * s] = [0, 0, 0] if ch == '#' else [255, 255, 255]


THUMB = None


def menu_frame(t):
    global THUMB
    if THUMB is None:
        THUMB = np.asarray(Image.open(os.path.join(HERE, 'thumb.png')).convert('RGB').resize((176, 176), Image.NEAREST)).astype(np.float32)
    S = 6
    img = dirt_bg(W, H, S)
    if t < 1.15:
        ptext(img, 'Select World', W // 2, 340, S, (255, 255, 255), center=True)
        # the list
        img[420:1330] *= 0.45
        entries = [('My First World', 'my first world (11/11/2016 4:12 PM)', 'Survival Mode', True),
                   ('New World', 'new world (2/6/2017 7:40 PM)', 'Creative Mode', False),
                   ('build ideas', 'build ideas (8/22/2016 1:03 PM)', 'Creative Mode', False)]
        y = 470
        for name, l2, l3, sel in entries:
            if sel:
                img[y - 12:y + 188, 60:1020] = [140, 140, 140]
                img[y - 6:y + 182, 66:1014] = [0, 0, 0]
                img[y:y + 176, 84:260] = THUMB
            else:
                img[y:y + 176, 84:260] = [40, 40, 40]
                img[y + 70:y + 106, 160:184] = [70, 70, 70]
            ptext(img, name, 290, y + 14, 5, (255, 255, 255))
            ptext(img, l2, 290, y + 74, 4, (150, 150, 150))
            ptext(img, l3, 290, y + 124, 4, (150, 150, 150))
            y += 270
        # buttons
        cx, cy = 300, 1530
        u = A.smooth(t, 0.30, 0.98)
        mx = 830 + (cx - 830) * u + 30 * math.sin(u * math.pi)
        my = 1240 + (cy - 1240) * u
        hov = abs(mx - cx) < 240 and abs(my - cy) < 50
        state = 2 if 1.02 <= t < 1.12 else (1 if hov else 0)
        button(img, 60, 1480, 480, 100, 'Play Selected World', 4, state)
        button(img, 560, 1480, 460, 100, 'Create New World', 4, 0)
        button(img, 60, 1600, 225, 100, 'Edit', 4, 0); button(img, 300, 1600, 240, 100, 'Delete', 4, 0)
        button(img, 560, 1600, 225, 100, 'Re-Create', 3, 0); button(img, 800, 1600, 220, 100, 'Cancel', 4, 0)
        cursor(img, int(mx), int(my))
    elif t < 2.28:
        ptext(img, 'Loading world', W // 2, 820, S, (255, 255, 255), center=True)
        ptext(img, 'Building terrain', W // 2, 920, 5, (255, 255, 255), center=True)
        p = A.smooth(t, 1.25, 2.2)
        x0, x1, y0 = 240, 840, 1010
        img[y0:y0 + 14, x0:x1] = [128, 128, 128]
        img[y0:y0 + 14, x0:x0 + int((x1 - x0) * p)] = [128, 255, 128]
    else:
        img[:] = 0
    # fade in from black at the very start
    img *= A.smooth(t, 0.0, 0.18)
    return img


# ------------------------------------------------------------------ captions and the grade
_VIG = None


def vignette(w, h):
    global _VIG
    if _VIG is None or _VIG.shape[:2] != (h, w):
        yy, xx = np.mgrid[0:h, 0:w]
        d = np.hypot((xx - w / 2) / (w * 0.62), (yy - h / 2) / (h * 0.62))
        _VIG = (1 - 0.30 * np.clip(d, 0, 1) ** 2.2)[..., None].astype(np.float32)
    return _VIG


def caption(img, t):
    for a, b, txt in CAPTIONS:
        if a <= t <= b:
            ca = A.smooth(t, a, a + 0.35) * (1 - A.smooth(t, b - 0.4, b))
            sz = int(round(58 * img.shape[1] / W)); k = img.shape[1] / W
            im = Image.fromarray(img.astype(np.uint8)).convert('RGBA'); ov = Image.new('RGBA', im.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov); f = font(sz)
            for li, line in enumerate(txt.split('\n')):
                tw = d.textlength(line, font=f); x0, y0 = (im.width - tw) / 2, im.height * 0.085 + li * sz * 1.22
                for dx, dy in ((3, 3), (-2, 2), (2, -2), (-2, -2), (0, 4)):
                    d.text((x0 + dx * k, y0 + dy * k), line, font=f, fill=(10, 8, 6, int(150 * ca)))
                d.text((x0, y0), line, font=f, fill=(255, 252, 245, int(240 * ca)))
            img = np.asarray(Image.alpha_composite(im, ov).convert('RGB')).astype(np.float32)
    return img


def post(img, t):
    img = img.astype(np.float32)
    if t >= A.T_WORLD:
        img *= vignette(img.shape[1], img.shape[0])
        img *= A.smooth(t, A.T_WORLD, A.T_WORLD + 0.35)                 # the world fades in
        img *= 1 - A.smooth(t, 22.6, 23.9)                               # and out at the end
    img = caption(img, t)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def view_at(k, w, h):
    v = A.frame(k); v['w'] = w; v['h'] = h
    return v


def run(ks, out_dir, w=W, h=H, prefix='f_%05d.png'):
    os.makedirs(out_dir, exist_ok=True)
    world = [k for k in ks if k / A.FPS >= A.T_WORLD]
    for k in ks:
        if k / A.FPS < A.T_WORLD:
            im = menu_frame(k / A.FPS)
            if w != W:
                im = np.asarray(Image.fromarray(im.astype(np.uint8)).resize((w, h), Image.NEAREST)).astype(np.float32)
            post(im, k / A.FPS).save(os.path.join(out_dir, prefix % k))
    if not world:
        return
    V.ensure_server()
    with sync_playwright() as pw:
        b, pg = V.open_page(pw)
        t0 = time.time()
        for i, k in enumerate(world):
            img = np.asarray(V.shot(pg, view_at(k, w, h)))
            post(img, k / A.FPS).save(os.path.join(out_dir, prefix % k))
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
    if mode == 'thumb':
        V.ensure_server()
        with sync_playwright() as pw:
            b, pg = V.open_page(pw)
            im = V.shot(pg, dict(w=256, h=256, pos=[134.0, 67.2, 131.0], yaw=22, pitch=8, fov=60, time=3.0, door=0))
            im.save(os.path.join(HERE, 'thumb.png')); b.close()
    elif mode == 'test':
        ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [0.5, 1.05, 1.8, 2.6, 4.0, 5.6, 7.5, 9.8, 12.5, 13.3, 14.6, 15.9, 16.6, 17.3, 18.2, 19.4, 20.4, 21.3, 22.5, 23.5]
        ks = [int(round(t * A.FPS)) for t in ts]
        d = os.path.join(HERE, 'test'); run(ks, d, 360, 640)
        sheet([os.path.join(d, 'f_%05d.png' % k) for k in ks], os.path.join(HERE, 'test_sheet.png'), 5, 360, 640)
        print('test sheet written')
    else:
        a, b_ = (int(x) for x in sys.argv[2].split(':'))
        run(list(range(a, b_)), os.path.join(HERE, 'frames'))
        print('frames %d..%d done' % (a, b_))
