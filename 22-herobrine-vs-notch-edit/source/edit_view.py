"""Render test stills of the edit through web/edit.html.  python3 edit_view.py name [name ...]"""
import base64, io, os, subprocess, sys, time, urllib.request
from PIL import Image
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web'); PORT = 8796
GUARD = dict(lean=8, crouch=1.2, head=[0, 0, 0], armR=[-55, 12, 55], armL=[-30, -10, 40], legR=[12, 4, 14], legL=[-18, -4, 22], sword=[-30, 0, 0], cape=14, eyes=1, glow=0.3)
H0 = dict(GUARD, pos=[-4.5, 0, 0], yaw=90)
N0 = dict(GUARD, pos=[4.5, 0, 0], yaw=-90)
VIEWS = {
    'wide': dict(cam=dict(pos=[0, 5, 21], look=[0, 1.2, 0], fov=50), H=H0, N=N0),
    'north': dict(cam=dict(pos=[2, 2.2, 10], look=[0, 7, -16], fov=55), H=H0, N=N0),
    'west': dict(cam=dict(pos=[8, 2.0, 1], look=[-30, 5, 0], fov=50), H=H0, N=N0),
    'east': dict(cam=dict(pos=[-8, 2.0, 1], look=[30, 3, 0], fov=50), H=H0, N=N0),
    'est': dict(cam=dict(pos=[22, 12, 30], look=[-3, 2.5, -4], fov=48), H=H0, N=N0),
    'h_close': dict(cam=dict(pos=[-2.6, 1.75, 1.2], look=[-4.5, 1.6, 0], fov=40), H=H0, N=N0),
    'n_close': dict(cam=dict(pos=[2.6, 1.75, 1.2], look=[4.5, 1.6, 0], fov=40), H=H0, N=N0),
    'med': dict(cam=dict(pos=[0, 1.9, 7.5], look=[0, 1.3, 0], fov=45), H=H0, N=N0),
    'title': dict(cam=dict(pos=[0, 3.0, 8], look=[0, 2.6, 0], fov=50), H=H0, N=N0, texts=[dict(id='VS', pos=[0, 2.8, 0], rot=[0, 0, 0], scale=1, glow=1.0)],
                  fx=dict(sparks=[[0, 1.6, 0, 0.12, 3, 1, 0.8, 0.4, 140, 10]], rings=[[0, 0.05, 0, 0.25, 6, 1, 0.6, 0.3, 0]])),
    'low': dict(cam=dict(pos=[-1.5, 0.35, 5.5], look=[-1.0, 1.6, 0], fov=60), H=H0, N=N0, lightning=0.0,
                fx=dict(bolts=[[8, 40, -10, 6, 0, -6, 4, 1.0, 'blue']], slashes=[[4.5, 1.2, 0, 1, 0.4, 0, 0, 1, 0, 1.4, -1.0, 2.2, 0.8, 1, 0.75, 0.3, 0.25]])),
}


def ensure_server():
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{PORT}/edit.html', timeout=2)
    except Exception:
        subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)


def open_page(pw):
    b = pw.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': 540, 'height': 960})
    pg.on('console', lambda m: print('console:', m.text[:300]) if m.type in ('error',) else None)
    pg.on('pageerror', lambda e: print('pageerror:', str(e)[:800]))
    t0 = time.time()
    pg.goto(f'http://127.0.0.1:{PORT}/edit.html')
    pg.wait_for_function('window.READY === true', timeout=600000)
    print('scene loaded %.0fs' % (time.time() - t0), flush=True)
    return b, pg


def shot(pg, v):
    url = pg.evaluate('(v) => renderFrame(v)', v)
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')


if __name__ == '__main__':
    names = sys.argv[1:] or list(VIEWS)
    ensure_server()
    with sync_playwright() as p:
        b, pg = open_page(p)
        ims = []
        for n in names:
            v = dict(VIEWS[n]); v.setdefault('t', 3.0); v['w'] = 540; v['h'] = 960
            t1 = time.time(); im = shot(pg, v); ims.append(im); print(n, '%.1fs' % (time.time() - t1), flush=True)
        b.close()
    s = Image.new('RGB', (540 * len(ims), 960))
    for i, im in enumerate(ims):
        s.paste(im, (540 * i, 0))
    s.save(os.path.join(HERE, 'edit_sheet.png'))
