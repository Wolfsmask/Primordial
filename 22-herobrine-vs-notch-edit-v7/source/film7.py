"""Render v7: python3 film2.py test t1,t2,... [WxH] | render a:b [WxH]   (frames3/f_%05d.png at 60 fps)"""
import os, sys, time
from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright
import v2_view as V
V.PAGE = 'edit6.html'
import anim7 as A
HERE = os.path.dirname(os.path.abspath(__file__))


def render(ks, out_dir, w, h, label=False):
    os.makedirs(out_dir, exist_ok=True)
    V.ensure_server()
    with sync_playwright() as pw:
        b, pg = V.open_page(pw)
        t0 = time.time()
        for i, k in enumerate(ks):
            f = A.frame(k); f['w'] = w; f['h'] = h
            im = V.shot(pg, f)
            if label: ImageDraw.Draw(im).text((6, 6), 'v %.2f  s %.2f' % (k / A.FPS, f['t']), fill=(255, 255, 0))
            im.save(os.path.join(out_dir, 'f_%05d.png' % k))
            if i % 60 == 0: print('frame %d (%d/%d) %.0fs' % (k, i, len(ks), time.time() - t0), flush=True)
        b.close()


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        ts = [float(x) for x in sys.argv[2].split(',')]
        w, h = (int(x) for x in sys.argv[3].split('x')) if len(sys.argv) > 3 else (360, 640)
        ks = [int(round(t * A.FPS)) for t in ts]
        d = os.path.join(HERE, 'test7'); render(ks, d, w, h, label=True)
        cols = 6; rows = (len(ks) + cols - 1) // cols; W_, H_ = 300, 533
        s = Image.new('RGB', (cols * W_, rows * H_))
        for i, k in enumerate(ks): s.paste(Image.open(os.path.join(d, 'f_%05d.png' % k)).resize((W_, H_)), ((i % cols) * W_, (i // cols) * H_))
        out = sys.argv[4] if len(sys.argv) > 4 else 'test7_sheet.png'; s.save(os.path.join(HERE, out)); print('sheet', out)
    else:
        a, b_ = (int(x) for x in sys.argv[2].split(':'))
        w, h = (int(x) for x in sys.argv[3].split('x')) if len(sys.argv) > 3 else (720, 1280)
        render(list(range(a, b_)), os.path.join(HERE, 'frames7'), w, h)
        print('frames %d..%d done' % (a, b_))
