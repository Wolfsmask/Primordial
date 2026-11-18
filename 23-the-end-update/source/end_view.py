"""Browser driver for web/end.html (headless Chromium + swiftshader)."""
import base64, io, os, subprocess, sys, time, urllib.request
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web'); PORT = 8797; PAGE = 'end.html'


def ensure_server():
    try: urllib.request.urlopen(f'http://127.0.0.1:{PORT}/{PAGE}', timeout=2)
    except Exception:
        subprocess.Popen([sys.executable, '-m', 'http.server', str(PORT), '--bind', '127.0.0.1'], cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)


def open_page(pw):
    b = pw.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'])
    pg = b.new_page(viewport={'width': 540, 'height': 960})
    pg.on('console', lambda m: print('console:', m.text[:300]) if m.type in ('error',) else None)
    pg.on('pageerror', lambda e: print('pageerror:', str(e)[:800]))
    t0 = time.time(); pg.goto(f'http://127.0.0.1:{PORT}/{PAGE}')
    pg.wait_for_function('window.READY === true', timeout=600000); print('scene loaded %.0fs' % (time.time() - t0))
    return b, pg


def shot(pg, v):
    url = pg.evaluate('(v) => renderView(v)', v)
    return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))).convert('RGB')
