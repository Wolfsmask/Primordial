"""Shared browser driver for web/edit2.html.  python3 v2_view.py  -> rig test sheet"""
import base64, io, os, subprocess, sys, time, urllib.request
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright
import rig as R
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web'); PORT = 8796
PAGE = 'edit2.html'


def ensure_server():
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{PORT}/{PAGE}', timeout=2)
    except Exception:
        subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)


def open_page(pw, query=''):
    b = pw.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': 540, 'height': 960})
    pg.on('console', lambda m: print('console:', m.text[:300]) if m.type in ('error',) else None)
    pg.on('pageerror', lambda e: print('pageerror:', str(e)[:800]))
    t0 = time.time()
    pg.goto(f'http://127.0.0.1:{PORT}/{PAGE}{query}')
    pg.wait_for_function('window.READY === true', timeout=600000)
    print('scene loaded %.0fs' % (time.time() - t0), flush=True)
    return b, pg


def shot(pg, v):
    url = pg.evaluate('(v) => renderFrame(v)', v)
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')


def stand(x, z, yaw_deg):
    Q = R.ident(); q = R.qaxis([0, 1, 0], np.radians(yaw_deg)); Q[:] = q
    return R.pose_dict([x, 12 * R.PX, z], Q)


def bent(x, z, yaw_deg):
    """a test pose: waist bent forward and sideways, elbows and knees bent"""
    yaw = R.qaxis([0, 1, 0], np.radians(yaw_deg))
    Q = R.ident(); Q[:] = yaw
    def setb(n, *rots):
        q = np.array([0, 0, 0, 1.0])
        for ax, ang in rots: q = R.qmul(q, R.qaxis(ax, np.radians(ang)))
        Q[R.BI[n]] = R.qmul(yaw, q)
    setb('chest', ([1, 0, 0], 25), ([0, 0, 1], 12))
    setb('head', ([1, 0, 0], 10), ([0, 1, 0], 20))
    setb('uarmR', ([1, 0, 0], -70), ([0, 0, 1], -15)); setb('farmR', ([1, 0, 0], -70), ([0, 0, 1], -15), ([1, 0, 0], -100))
    setb('uarmL', ([1, 0, 0], -30), ([0, 0, 1], 20)); setb('farmL', ([1, 0, 0], -30), ([0, 0, 1], 20), ([1, 0, 0], -60))
    setb('thighR', ([1, 0, 0], -50)); setb('shinR', ([1, 0, 0], -50), ([1, 0, 0], 80))
    setb('thighL', ([1, 0, 0], 15)); setb('shinL', ([1, 0, 0], 15), ([1, 0, 0], 30))
    return R.pose_dict([x, 11 * R.PX, z], Q)


if __name__ == '__main__':
    ensure_server()
    views = [dict(cam=dict(pos=[0, 1.4, 4.2], look=[0, 1.1, 0], fov=45), H=stand(-0.8, 0, 0), N=stand(0.8, 0, 0)),
             dict(cam=dict(pos=[0, 1.4, -4.2], look=[0, 1.1, 0], fov=45), H=stand(-0.8, 0, 0), N=stand(0.8, 0, 0)),
             dict(cam=dict(pos=[3.2, 1.6, 2.6], look=[0, 1.0, 0], fov=45), H=bent(-0.8, 0, 0), N=bent(0.8, 0, 0)),
             dict(cam=dict(pos=[-0.5, 1.9, 1.9], look=[-0.8, 1.6, 0], fov=40), H=stand(-0.8, 0, 0), N=stand(0.8, 0, 0)),
             dict(cam=dict(pos=[0.9, 1.9, 1.9], look=[0.8, 1.6, 0], fov=40), H=stand(-0.8, 0, 0), N=stand(0.8, 0, 0))]
    with sync_playwright() as p:
        b, pg = open_page(p)
        ims = []
        for v in views:
            v.update(t=3.0, w=540, h=960); t1 = time.time(); ims.append(shot(pg, v)); print('%.1fs' % (time.time() - t1))
        b.close()
    s = Image.new('RGB', (360 * len(ims), 640))
    for i, im in enumerate(ims): s.paste(im.resize((360, 640)), (i * 360, 0))
    s.save(os.path.join(HERE, 'test', 'rig_test.png')); print('ok')
