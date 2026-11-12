"""Render test stills of the old world through web/ow.html.  python3 ow_view.py name [name ...]"""
import base64, io, json, math, os, subprocess, sys, time, urllib.request
from PIL import Image
from playwright.sync_api import sync_playwright
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web'); PORT = 8794
SPAWN = [128.5, 65.0, 128.5]
EYE = 1.62
WOLF_SIT = dict(x=128.5, y=65.0, z=141.6, yaw=180, sit=1, phase=0, walkAmt=0, headYaw=0, headPitch=5, headRoll=0, wag=0, wagPhase=0, tailUp=0, ears=0.5)
VIEWS = {
    'spawn': dict(pos=[128.5, 65 + EYE, 128.5], yaw=0, pitch=4, fov=70),
    'west': dict(pos=[128.5, 65 + EYE, 128.5], yaw=70, pitch=2, fov=70),
    'north': dict(pos=[128.5, 65 + EYE, 128.5], yaw=180, pitch=-4, fov=70),
    'east': dict(pos=[128.5, 65 + EYE, 128.5], yaw=-70, pitch=0, fov=70),
    'door': dict(pos=[128.5, 65 + EYE, 135.6], yaw=0, pitch=8, fov=70, door=0.0),
    'inside': dict(pos=[128.5, 65 + EYE, 139.3], yaw=0, pitch=22, fov=70, door=1.0, wolf=WOLF_SIT),
    'wolf_close': dict(pos=[128.5, 65 + 1.27, 139.4], yaw=0, pitch=30, fov=70, door=1.0,
                       wolf=dict(WOLF_SIT, z=140.6, yaw=180, sit=1, headPitch=-25, headRoll=14, tailUp=1, wag=0.5, wagPhase=1.0, ears=1)),
    'wolf_stand': dict(pos=[128.5, 65 + EYE, 139.3], yaw=0, pitch=25, fov=70, door=1.0,
                       wolf=dict(WOLF_SIT, z=140.8, yaw=180, sit=0, headPitch=-10, tailUp=1, wag=0.5, wagPhase=0.5, ears=1, walkAmt=1, phase=0.8)),
    'wolf_side': dict(pos=[131.0, 65 + 1.0, 141.6], yaw=90, pitch=10, fov=60, door=1.0, wolf=dict(WOLF_SIT, yaw=180)),
    'wolf_side_stand': dict(pos=[131.0, 65 + 1.0, 141.6], yaw=90, pitch=10, fov=60, door=1.0, wolf=dict(WOLF_SIT, yaw=180, sit=0, tailUp=1, wag=0.4, wagPhase=0.3)),
    'house': dict(pos=[133.5, 67.5, 130.5], yaw=25, pitch=10, fov=70),
}


def ensure_server():
    try:
        urllib.request.urlopen(f'http://127.0.0.1:{PORT}/ow.html', timeout=2)
    except Exception:
        subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)


def open_page(pw):
    b = pw.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': 540, 'height': 960})
    pg.on('console', lambda m: print('console:', m.text[:400]) if m.type in ('error', 'warning') else None)
    pg.on('pageerror', lambda e: print('pageerror:', str(e)[:800]))
    t0 = time.time()
    pg.goto(f'http://127.0.0.1:{PORT}/ow.html')
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
            v.setdefault('arm', dict(bx=0, by=0))
            t1 = time.time(); im = shot(pg, v); im.save(os.path.join(HERE, f'ow_{n}.png')); ims.append(im)
            print(n, '%.1fs' % (time.time() - t1), flush=True)
        b.close()
    s = Image.new('RGB', (540 * len(ims), 960)); x = 0
    for i in ims:
        s.paste(i, (x, 0)); x += 540
    s.save(os.path.join(HERE, 'ow_sheet.png'))
