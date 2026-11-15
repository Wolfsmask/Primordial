"""Render the edit: python3 edit_film.py test [t,...] | render a:b  (frames -> frames/f_%05d.png, 1080 x 1920, 60 fps)"""
import os, sys, time
from PIL import Image
from playwright.sync_api import sync_playwright
import edit_view as V
import edit_anim as A

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1080, 1920


def render(ks, out_dir, w=W, h=H):
    os.makedirs(out_dir, exist_ok=True)
    V.ensure_server()
    with sync_playwright() as pw:
        b, pg = V.open_page(pw)
        t0 = time.time()
        for i, k in enumerate(ks):
            f = A.frame(k); f['w'] = w; f['h'] = h
            V.shot(pg, f).save(os.path.join(out_dir, 'f_%05d.png' % k))
            if i % 60 == 0:
                print('frame %d (%d/%d) %.0fs' % (k, i, len(ks), time.time() - t0), flush=True)
        b.close()


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'test'
    if mode == 'test':
        ts = [float(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else \
            [1.0, 3.9, 5.0, 5.6, 6.2, 6.6, 7.6, 9.0, 9.72, 11.0, 12.6, 13.2, 14.3, 15.2, 16.6, 17.75, 18.6, 20.4, 21.6, 22.45, 23.4, 25.0, 27.0, 28.6, 29.35, 30.2]
        ks = [int(round(t * A.FPS)) for t in ts]
        d = os.path.join(HERE, 'test'); render(ks, d, 360, 640)
        cols = 6; rows = (len(ks) + cols - 1) // cols
        s = Image.new('RGB', (cols * 300, rows * 533))
        for i, k in enumerate(ks):
            s.paste(Image.open(os.path.join(d, 'f_%05d.png' % k)).resize((300, 533)), ((i % cols) * 300, (i // cols) * 533))
        s.save(os.path.join(HERE, 'test_sheet.png')); print('sheet written')
    else:
        a, b_ = (int(x) for x in sys.argv[2].split(':'))
        w, h = (int(x) for x in sys.argv[3].split('x')) if len(sys.argv) > 3 else (W, H)
        render(list(range(a, b_)), os.path.join(HERE, 'frames'), w, h)
        print('frames %d..%d done' % (a, b_))
