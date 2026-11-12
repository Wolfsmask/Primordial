"""Render test stills of the old world through web/ow2.html.  python3 ow_view.py name [name ...]"""
import base64, io, json, math, os, subprocess, sys, time, urllib.request
from PIL import Image
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web'); PORT = 8794
SPAWN = [128.5, 65.0, 128.5]
EYE = 1.62
WS = dict(x=199.6, y=65.0, z=191.4, yaw=90, sit=1, phase=0, walkAmt=0, headYaw=0, headPitch=8, headRoll=0, wag=0, wagPhase=0, tailUp=0.2, ears=0.4)
PL = dict(x=196.6, y=65.0, z=191.5, yaw=90, phase=0, walkAmt=0, sneak=0, headPitch=10)
VIEWS = {
    'vista': dict(pos=[232, 96, 204], yaw=97, pitch=14, fov=60),
    'vista2': dict(pos=[212, 84, 196], yaw=92, pitch=9, fov=60),
    'path': dict(pos=[182.5, 66.62, 191.5], yaw=-90, pitch=6, fov=70),
    'door': dict(pos=[193.4, 66.62, 191.5], yaw=-90, pitch=8, fov=70, door=1.0, wolf=WS),
    'dog_close': dict(pos=[201.3, 65.75, 192.9], yaw=125, pitch=6, fov=50, door=1.0, wolf=WS, dof=dict(focus=2.2, aperture=6)),
    'rev': dict(pos=[201.4, 66.2, 191.6], yaw=90, pitch=3, fov=62, door=1.0, wolf=dict(WS, yaw=-90, sit=0, tailUp=1, wag=0.5, wagPhase=1), player=PL),
    'tilt': dict(pos=[197.6, 65.45, 191.0], yaw=-74, pitch=-6, fov=55, door=1.0, wolf=dict(WS, x=198.4, z=191.5, yaw=-90, sit=1, headPitch=-26, headRoll=24, tailUp=1, wag=0.5, ears=1),
                 player=dict(PL, x=197.0, sneak=1), dof=dict(focus=1.0, aperture=5)),
    'wolf_side': dict(pos=[199.6, 65.8, 194.4], yaw=180, pitch=5, fov=50, door=1.0, wolf=dict(WS, yaw=90, sit=0, tailUp=1)),
    'wolf_side_sit': dict(pos=[199.6, 65.8, 194.4], yaw=180, pitch=5, fov=50, door=1.0, wolf=dict(WS, yaw=90, sit=1)),
    'wolf_front': dict(pos=[201.8, 65.8, 191.4], yaw=90, pitch=6, fov=50, door=1.0, wolf=dict(WS, yaw=90, sit=0, tailUp=1, headPitch=0)),
    'wolf_front_sit': dict(pos=[201.8, 65.8, 191.4], yaw=90, pitch=6, fov=50, door=1.0, wolf=dict(WS, yaw=90, sit=1, tailUp=1, headPitch=-12, headRoll=20)),
    'dusk': dict(pos=[176, 72, 186], yaw=-78, pitch=6, fov=60, dusk=0.6, sunElev=1.5),
}


def ensure_server():
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{PORT}/ow2.html', timeout=2)
    except Exception:
        subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)


def open_page(pw):
    b = pw.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': 540, 'height': 960})
    pg.on('console', lambda m: print('console:', m.text[:400]) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: print('pageerror:', str(e)[:800]))
    t0 = time.time()
    pg.goto(f'http://127.0.0.1:{PORT}/ow2.html')
    pg.wait_for_function('window.READY === true', timeout=600000)
    print('scene loaded %.0fs' % (time.time() - t0), flush=True)
    return b, pg


def shot(pg, v):
    url = pg.evaluate('(v) => renderView(v)', v)
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')


if __name__ == '__main__':
    names = sys.argv[1:] or ['spawn', 'north', 'door', 'inside']
    ensure_server()
    with sync_playwright() as p:
        b, pg = open_page(p)
        ims = []
        for n in names:
            v = dict(VIEWS[n]); v.setdefault('time', 3.0); v['w'] = 540; v['h'] = 960
            t1 = time.time(); im = shot(pg, v); im.save(os.path.join(HERE, f'ow2_{n}.png')); ims.append(im)
            print(n, '%.1fs' % (time.time() - t1), flush=True)
        b.close()
    s = Image.new('RGB', (540 * len(ims), 960)); x = 0
    for i in ims:
        s.paste(i, (x, 0)); x += 540
    s.save(os.path.join(HERE, 'ow2_sheet.png'))
