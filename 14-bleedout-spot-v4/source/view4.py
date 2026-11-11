"""Render views of the v3 world through render4.html.  python3 view3.py name [name ...]"""
import base64, io, json, math, os, subprocess, sys, time, urllib.request
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web4'); PORT = 8792
EYE = [10.55, 252.0, 1.45]


def look_at(yaw, pitch, dist=100.0, eye=EYE):
    y, p = math.radians(yaw), math.radians(pitch)
    return [eye[0] - math.sin(y) * math.cos(p) * dist, eye[1] + math.sin(p) * dist, eye[2] - math.cos(y) * math.cos(p) * dist]


VIEWS = {
    'centre': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(79, -4)),
    'lake': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(90, -24)),
    'sky': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(40, 24)),
    'left': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(140, 3)),
    'right': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(38, -6)),
    'down': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(90, -62)),
    'trail': dict(w=540, h=960, fov=95, pos=[12.6, 252.45, -0.1], look=look_at(98, -60, 10, [12.6, 252.45, -0.1])),
    'over': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(84, -16)),
    'up': dict(w=540, h=960, fov=92, pos=EYE, look=look_at(74, 14)),
    'wide': dict(w=960, h=540, fov=70, pos=[60, 330, 120], look=look_at(85, -6, 300, [60, 330, 120])),
}


def ensure_server():
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{PORT}/render4.html', timeout=2)
    except Exception:
        subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)


def open_page(pw):
    b = pw.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': 720, 'height': 720})
    pg.on('console', lambda m: print('console:', m.text[:400]) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: print('pageerror:', str(e)[:800]))
    t0 = time.time()
    pg.goto(f'http://127.0.0.1:{PORT}/render4.html')
    pg.wait_for_function('window.READY === true', timeout=1800000)
    print('scene loaded %.0fs' % (time.time() - t0), flush=True)
    return b, pg


def shot(pg, v):
    url = pg.evaluate('(v) => renderView(v)', v)
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')


if __name__ == '__main__':
    names = sys.argv[1:] or ['centre', 'left', 'right', 'down']
    ensure_server()
    with sync_playwright() as p:
        b, pg = open_page(p)
        ims = []
        for k, n in enumerate(names):
            v = dict(VIEWS[n]); v['shadows'] = (k == 0); v['time'] = 3.0
            t1 = time.time(); im = shot(pg, v); im.save(os.path.join(HERE, f'v4_{n}.png')); ims.append(im)
            print(n, '%.0fs' % (time.time() - t1), flush=True)
        b.close()
    hh = max(i.height for i in ims); s = Image.new('RGB', (sum(i.width for i in ims), hh)); x = 0
    for i in ims:
        s.paste(i, (x, 0)); x += i.width
    s.save(os.path.join(HERE, 'v4_sheet.png'))
