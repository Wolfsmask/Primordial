"""The edit's timeline on the song's 140 BPM grid (the drop at 5.26 s, a bar of silence at 28.0-29.26 s): poses and
keyframes for both fighters, the camera of every shot, the effects, the 3D titles and the edit pass (zoom punches on
kicks, flashes, shakes, chromatic aberration, radial blur, grade), with slow motion where it counts.  60 fps."""
import json, math, os
import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 60
T_END = 31.0
NF = int(T_END * FPS)
BEAT = 0.4295
B = lambda n: 5.26 + n * BEAT
KICKS = [k for k in json.load(open(os.path.join(HERE, 'beats.json')))['kicks'] if 5.0 < k < 31.2]


def ss(u):
    u = min(1.0, max(0.0, u)); return u * u * (3 - 2 * u)


def ease(t, a, b):
    return ss((t - a) / (b - a)) if b > a else float(t >= b)


def track(keys):
    ts = np.array([k[0] for k in keys], float); vs = np.array([k[1] for k in keys], float)
    if len(ts) == 1:
        return lambda t: vs[0]
    f = PchipInterpolator(ts, vs, axis=0, extrapolate=False)
    return lambda t: vs[0] if t <= ts[0] else (vs[-1] if t >= ts[-1] else f(t))


# ------------------------------------------------------------------ time: slow motion and the freeze
SLOW = [(9.70, 10.40, 0.35), (10.40, 12.11, 0.55), (20.10, 22.32, 0.16), (28.00, 29.26, 0.015)]


def speed(t):
    for a, b, s in SLOW:
        if a <= t < b:
            edge = min(t - a, b - t)
            return s + (1 - s) * max(0.0, 1 - edge / 0.08) * 0.0
    return 1.0


_TS = np.arange(0, T_END + 0.05, 1 / 600)
_WT = np.concatenate([[0], np.cumsum([speed(t) / 600 for t in _TS[:-1]])])


def wtime(t):
    """world time (ambient motion, effect ages) under the slow-motion map"""
    return float(np.interp(t, _TS, _WT))


# ------------------------------------------------------------------ poses (degrees; arms [pitch, roll, elbow, yaw], legs [pitch, roll, knee])
P = {
    'idle': dict(lean=4, twist=0, crouch=0.4, head=[0, 0, 0], armR=[-20, 8, 25, 0], armL=[-8, -6, 15, 0], legR=[3, 2, 4], legL=[-3, -2, 4], sword=[-10, 0, 0], cape=10),
    'guard': dict(lean=8, twist=-8, crouch=1.4, head=[-2, 6, 0], armR=[-58, 14, 55, 0], armL=[-32, -10, 40, 0], legR=[14, 4, 16], legL=[-20, -4, 24], sword=[-28, 0, 0], cape=14),
    'windup': dict(lean=-6, twist=-42, crouch=1.8, head=[0, 30, 0], armR=[-196, 24, 34, 0], armL=[-62, -22, 34, 0], legR=[22, 5, 22], legL=[-26, -5, 16], sword=[-30, 0, 0], cape=20),
    'strike': dict(lean=24, twist=38, crouch=2.8, head=[6, -28, 0], armR=[-72, -24, 6, 0], armL=[12, -32, 10, 0], legR=[32, 5, 36], legL=[-42, -5, 12], sword=[8, 0, 0], cape=32),
    'block': dict(lean=4, twist=-6, crouch=1.8, head=[-8, 0, 0], armR=[-122, 30, 62, 0], armL=[-112, -30, 62, 0], legR=[20, 6, 26], legL=[-24, -6, 20], sword=[-6, 0, 80], cape=18),
    'lock': dict(lean=22, twist=4, crouch=2.2, head=[-4, 0, 0], armR=[-96, 10, 38, 0], armL=[-96, -12, 44, 0], legR=[36, 4, 30], legL=[-32, -4, 10], sword=[-24, 0, 0], cape=28),
    'leap': dict(lean=10, twist=-20, crouch=0, head=[-6, 10, 0], armR=[-176, 14, 22, 0], armL=[-70, -30, 40, 0], legR=[-62, 5, 100], legL=[-28, -5, 82], sword=[-20, 0, 0], cape=48),
    'land': dict(lean=36, twist=0, crouch=9.0, head=[24, 0, 0], armR=[-26, 34, 12, 0], armL=[-44, -12, 6, 0], legR=[-86, 8, 120], legL=[24, 0, 118], sword=[10, 0, 0], cape=-6),
    'hit': dict(lean=-30, twist=12, crouch=1.0, head=[-22, 0, 0], armR=[36, 44, 8, 0], armL=[30, -46, 8, 0], legR=[-32, 0, 10], legL=[-52, 0, 32], sword=[0, 0, 0], cape=60),
    'cast': dict(lean=-10, twist=10, crouch=0.6, head=[-24, 0, 0], armR=[-24, 12, 22, 0], armL=[-176, -6, 0, 0], legR=[10, 6, 8], legL=[-10, -6, 8], sword=[-10, 0, 0], cape=26),
    'kneel': dict(lean=30, twist=0, crouch=6.0, head=[18, 0, 0], armR=[-22, 22, 30, 0], armL=[-46, -10, 22, 0], legR=[-72, 5, 96], legL=[12, 0, 100], sword=[0, 0, 0], cape=4),
    'raise': dict(lean=-4, twist=-30, crouch=1.2, head=[-6, 20, 0], armR=[-170, 30, 20, 0], armL=[-90, -30, 40, 0], legR=[18, 6, 16], legL=[-22, -6, 20], sword=[-20, 0, 0], cape=30),
}
KEYS_P = ['lean', 'twist', 'crouch', 'cape']


def blend(a, b, u):
    out = {}
    for k in a:
        if isinstance(a[k], list):
            out[k] = [x + (y - x) * u for x, y in zip(a[k], b[k])]
        else:
            out[k] = a[k] + (b[k] - a[k]) * u
    return out


def run_overlay(p, phase, amt):
    s = math.sin(phase); c = math.cos(phase)
    p = dict(p)
    p['legR'] = [p['legR'][0] + 48 * s * amt, p['legR'][1], p['legR'][2] + 50 * max(0, -c) * amt + 10 * amt]
    p['legL'] = [p['legL'][0] - 48 * s * amt, p['legL'][1], p['legL'][2] + 50 * max(0, c) * amt + 10 * amt]
    p['armL'] = [p['armL'][0] + 40 * s * amt, p['armL'][1], p['armL'][2] + 20 * amt, 0]
    p['lean'] += 22 * amt; p['crouch'] += 0.8 * amt + 0.5 * abs(c) * amt; p['cape'] += 40 * amt
    return p


class Fighter:
    """keys: (t, pose, pos (x, y, z), yaw, opts) with opts: run=True (run cycle while moving), arc=h (a jump arc), eyes, glow, visible"""
    def __init__(self, keys):
        self.keys = keys

    def at(self, t):
        K = self.keys
        if t <= K[0][0]:
            i, u = 0, 0.0
        elif t >= K[-1][0]:
            i, u = len(K) - 2, 1.0
        else:
            i = max(j for j in range(len(K) - 1) if K[j][0] <= t)
            u = (t - K[i][0]) / (K[i + 1][0] - K[i][0])
        a, b = K[i], K[i + 1]
        oa, ob = a[4], b[4]
        e = u if ob.get('linear') else ss(u)
        p = blend(P[a[1]], P[b[1]], e)
        pa, pb = np.array(a[2], float), np.array(b[2], float)
        mv = u if ob.get('linear') or ob.get('run') else ss(u)
        pos = pa + (pb - pa) * mv
        if ob.get('arc'):
            pos[1] += ob['arc'] * 4 * mv * (1 - mv)
        if ob.get('fall'):
            pos = pa + (pb - pa) * (u ** 2)
        ya, yb = a[3], b[3]
        dy = (yb - ya + 180) % 360 - 180
        yaw = ya + dy * e
        if ob.get('run'):
            dist = float(np.linalg.norm(pb - pa)) * u
            amt = min(1.0, 4 * u * (1 - u) * 2.5)
            p = run_overlay(p, dist / 1.1 * math.pi, amt)
        p.update(pos=[float(x) for x in pos], yaw=float(yaw))
        p['eyes'] = oa.get('eyes', 1) + (ob.get('eyes', oa.get('eyes', 1)) - oa.get('eyes', 1)) * e
        p['glow'] = oa.get('glow', 0.2) + (ob.get('glow', oa.get('glow', 0.2)) - oa.get('glow', 0.2)) * e
        p['visible'] = bool(oa.get('visible', True)) if u < 1 else bool(ob.get('visible', True))
        p['wind'] = 0.6 + 0.6 * min(1.0, p['cape'] / 40)
        return p


def K(t, pose, pos, yaw, **o):
    return (t, pose, pos, yaw, o)


# Herobrine starts on the west half of the floor facing east (+x); Notch arrives on the east half facing west.
HB = Fighter([
    K(0.0, 'idle', [-6, 0, 0], 90, eyes=0, glow=0.0), K(3.2, 'idle', [-6, 0, 0], 90, eyes=0, glow=0.0), K(3.6, 'idle', [-6, 0, 0], 90, eyes=1, glow=0.2),
    K(5.26, 'idle', [-6, 0, 0], 90, glow=0.3), K(6.0, 'guard', [-5.2, 0, 0], 90, glow=0.3), K(8.68, 'guard', [-4.5, 0, 0], 90, glow=0.3),
    K(9.30, 'windup', [-1.4, 0, 0], 90, run=True), K(9.68, 'lock', [-0.78, 0, 0], 90, glow=0.9),
    K(12.11, 'lock', [-0.70, 0, 0], 90, glow=1.0), K(12.40, 'guard', [-1.8, 0, 0], 90),
    K(12.54, 'guard', [-1.8, 0, 0], 90, visible=False), K(12.97, 'windup', [3.3, 0, 0], -90, visible=True, glow=0.8),
    K(13.17, 'strike', [3.1, 0, 0], -90, glow=1.0), K(13.83, 'guard', [3.4, 0, 0], -90),
    K(14.26, 'block', [3.5, 0, 0], -90), K(14.69, 'block', [3.6, 0, 0], -90), K(15.12, 'hit', [3.8, 0, 0], -90),
    K(15.40, 'hit', [8.6, 0.4, 0], -90, arc=1.2), K(15.60, 'kneel', [9.5, 0, 0], -90, linear=True),
    K(16.40, 'kneel', [9.5, 0, 0], -90, eyes=1), K(16.45, 'kneel', [9.5, 0, 0], -90, eyes=2.4, glow=1.2), K(17.20, 'guard', [9.5, 0, 0], -90, eyes=2.2),
    K(17.50, 'cast', [9.5, 0, 0], -90, eyes=2.4, glow=1.4), K(19.30, 'cast', [9.5, 0, 0], -90, eyes=2.4),
    K(19.83, 'windup', [8.2, 0, 0], -90, run=True), K(22.30, 'leap', [6.6, 3.0, 0], -90, glow=1.6),
    K(22.42, 'hit', [6.9, 3.1, 0], -90), K(23.25, 'land', [7.4, 0, 0], -90, arc=1.5),
    K(24.11, 'land', [7.4, 0, 0], -90), K(24.90, 'guard', [7.4, 0, 0], -90, eyes=2.6, glow=1.4),
    K(26.05, 'raise', [7.4, 0, 0], -90, eyes=3.0, glow=2.0), K(27.54, 'raise', [7.4, 0, 0], -90, glow=2.2),
    K(28.00, 'strike', [6.75, 0, 0], -90, run=True, glow=2.4), K(29.26, 'strike', [6.72, 0, 0], -90, glow=2.4),
    K(29.60, 'hit', [9.8, 0.6, 0], -90, arc=1.0), K(31.0, 'hit', [10.6, 0, 0], -90),
])
NB = Fighter([
    K(0.0, 'leap', [4.5, 30, 0], -90, visible=False, glow=0.0), K(5.26, 'leap', [4.5, 15, 0], -90, visible=True, glow=0.6),
    K(6.12, 'land', [4.5, 0, 0], -90, fall=True, glow=0.8), K(6.97, 'land', [4.5, 0, 0], -90),
    K(7.60, 'guard', [4.5, 0, 0], -90, glow=0.4), K(8.68, 'guard', [4.5, 0, 0], -90),
    K(9.30, 'windup', [1.4, 0, 0], -90, run=True), K(9.68, 'lock', [0.78, 0, 0], -90, glow=0.9),
    K(12.11, 'lock', [0.70, 0, 0], -90, glow=1.0), K(12.40, 'guard', [1.8, 0, 0], -90),
    K(12.97, 'guard', [1.8, 0, 0], -90), K(13.17, 'block', [1.9, 0, 0], 90),
    K(13.83, 'guard', [1.9, 0, 0], 90), K(14.10, 'windup', [2.0, 0, 0], 90), K(14.26, 'strike', [2.2, 0, 0], 90, glow=1.2),
    K(14.50, 'windup', [2.2, 0, 0], 90), K(14.69, 'strike', [2.4, 0, 0], 90, glow=1.2), K(14.95, 'windup', [2.4, 0, 0], 90),
    K(15.12, 'strike', [2.8, 0, 0], 90, glow=1.6), K(15.80, 'guard', [2.6, 0, 0], 90), K(17.25, 'guard', [2.4, 0, 0], 90),
    K(17.60, 'block', [2.4, 0, 0], 90, glow=1.4), K(19.30, 'block', [2.4, 0, 0], 90, glow=1.4),
    K(19.83, 'windup', [3.6, 0, 0], 90, run=True), K(22.30, 'leap', [5.4, 3.0, 0], 90, glow=1.8),
    K(22.42, 'hit', [5.1, 3.1, 0], 90), K(23.25, 'land', [4.6, 0, 0], 90, arc=1.5),
    K(24.11, 'land', [4.6, 0, 0], 90), K(24.90, 'guard', [4.6, 0, 0], 90, glow=1.4),
    K(26.05, 'raise', [4.6, 0, 0], 90, glow=2.0), K(27.54, 'raise', [4.6, 0, 0], 90, glow=2.2),
    K(28.00, 'strike', [5.25, 0, 0], 90, run=True, glow=2.4), K(29.26, 'strike', [5.28, 0, 0], 90, glow=2.4),
    K(29.60, 'hit', [2.2, 0.6, 0], 90, arc=1.0), K(31.0, 'hit', [1.4, 0, 0], 90),
])


# ------------------------------------------------------------------ cameras, shot by shot
class Cam:
    def __init__(self, t0, t1, pos, look, fov=45, roll=None):
        self.t0, self.t1 = t0, t1
        self.pos = track(pos); self.look = track(look)
        self.fov = track(fov) if isinstance(fov, list) else (lambda t, f=fov: f)
        self.roll = track(roll) if roll else (lambda t: 0.0)

    def at(self, t):
        return dict(pos=[float(x) for x in self.pos(t)], look=[float(x) for x in self.look(t)], fov=float(self.fov(t)), roll=float(self.roll(t)))


def orbit(t0, t1, c, r, h, a0, a1, n=12, look_dy=0.0):
    ks = []
    for i in range(n + 1):
        u = i / n; a = math.radians(a0 + (a1 - a0) * u)
        ks.append((t0 + (t1 - t0) * u, [c[0] + math.sin(a) * r, h, c[2] + math.cos(a) * r]))
    return ks


CAMS = [
    Cam(0.0, 2.6, [(0, [24, 13, 30]), (2.6, [15, 6.5, 19])], [(0, [-2, 1.5, 0]), (2.6, [-5, 1.4, 0])], 42),
    Cam(2.6, 4.6, [(2.6, [-4.3, 1.74, -0.45]), (4.6, [-4.7, 1.72, -0.25])], [(2.6, [-6, 1.62, 0]), (4.6, [-6, 1.66, 0])], 40),
    Cam(4.6, 5.26, [(4.6, [-4.7, 1.72, -0.25]), (5.26, [-4.95, 1.70, -0.15])], [(4.6, [-6, 1.66, 0]), (5.26, [-6, 1.66, 0])], [(4.6, 40), (5.26, 30)], roll=[(4.6, 0), (5.26, 4)]),
    Cam(5.26, 6.97, [(5.26, [9.5, 0.9, 7.5]), (6.12, [8.6, 1.0, 6.4]), (6.97, [7.6, 1.2, 5.4])],
        [(t, [4.5, max(1.2, NB.at(t)['pos'][1] + 0.4), 0]) for t in np.linspace(5.26, 6.12, 9)] + [(6.97, [4.5, 1.2, 0])], 50),
    Cam(6.97, 8.68, [(6.97, [7.6, 2.3, 1.6]), (8.68, [1.2, 2.5, 0.6])], [(6.97, [-4.5, 1.6, 0]), (8.68, [-4.5, 1.8, 0])], [(6.97, 48), (8.68, 58)]),
    Cam(8.68, 10.4, [(8.68, [0.3, 1.0, 13.5]), (9.68, [0.2, 1.0, 5.6]), (10.4, [0.1, 1.1, 4.8])], [(8.68, [0, 1.4, 0]), (10.4, [0, 1.6, 0])], [(8.68, 56), (9.68, 46), (10.4, 42)]),
    Cam(10.4, 12.11, *zip(*[(k, (k[0], [0, 1.75, 0])) for k in orbit(10.4, 12.11, [0, 0, 0], 2.3, 1.9, 30, 200)]), 42) if False else
    Cam(10.4, 12.11, orbit(10.4, 12.11, [0, 0, 0], 3.4, 1.85, -50, 35), [(10.4, [0, 1.7, 0]), (12.11, [0, 1.75, 0])], 48),
    Cam(12.11, 13.83, [(12.11, [0.4, 2.3, 9.5]), (12.9, [0.8, 2.2, 8.6]), (13.0, [2.0, 2.1, 7.2]), (13.83, [2.4, 1.9, 6.0])],
        [(12.11, [0, 1.3, 0]), (12.9, [0.2, 1.3, 0]), (13.0, [2.5, 1.5, 0]), (13.83, [2.6, 1.5, 0])], [(12.11, 46), (13.83, 42)], roll=[(12.11, 0), (12.92, 0), (13.0, -6), (13.3, 0)]),
    Cam(13.83, 15.55, [(13.83, [1.1, 1.7, 3.4]), (15.12, [1.8, 1.5, 2.9]), (15.55, [4.5, 1.6, 4.6])], [(13.83, [2.8, 1.4, 0]), (15.12, [3.3, 1.4, 0]), (15.55, [7.5, 1.0, 0])], [(13.83, 44), (15.55, 50)]),
    Cam(15.55, 17.25, [(15.55, [7.3, 0.35, 1.2]), (17.25, [7.8, 0.45, 0.9])], [(15.55, [9.5, 1.5, 0]), (17.25, [9.5, 1.9, 0])], [(15.55, 46), (17.25, 40)]),
    Cam(17.25, 19.4, [(17.25, [-1.6, 2.5, 2.8]), (19.4, [-0.9, 2.4, 2.4])], [(17.25, [8.5, 1.9, 0]), (19.4, [8.5, 2.0, 0])], 48),
    Cam(19.4, 20.72, [(19.4, [12.4, 2.0, 1.5]), (20.72, [11.2, 2.1, 1.1])], [(19.4, [5.0, 1.5, 0]), (20.72, [5.5, 1.7, 0])], 50),
    Cam(20.72, 22.4, [(20.72, [6.0, 0.9, 8.2]), (22.4, [6.0, 3.5, 8.8])], [(20.72, [6.0, 1.3, 0]), (22.4, [6.0, 3.1, 0])], [(20.72, 54), (22.4, 50)]),
    Cam(22.4, 24.11, [(22.4, [6.0, 3.5, 9.8]), (24.11, [6.0, 2.4, 9.2])], [(22.4, [6.0, 2.4, 0]), (24.11, [6.0, 1.4, 0])], 52),
    Cam(24.11, 28.0, [(24.11, [6.0, 1.5, 10.0]), (27.9, [6.0, 1.75, 7.0]), (28.0, [6.0, 1.8, 6.8])], [(24.11, [6.0, 1.3, 0]), (28.0, [6.0, 1.7, 0])], [(24.11, 52), (28.0, 50)]),
    Cam(28.0, 29.26, orbit(28.0, 29.26, [6.0, 0, 0], 3.0, 1.95, 0, 24, 4), [(28.0, [6.0, 1.9, 0]), (29.26, [6.0, 1.95, 0])], [(28.0, 40), (29.26, 34)]),
    Cam(29.26, 31.0, [(29.26, [7.24, 1.95, 2.8]), (29.7, [6.6, 4.5, 16.0]), (31.0, [6.4, 5.4, 19.0])], [(29.26, [6.0, 1.95, 0]), (31.0, [6.0, 1.6, 0])], [(29.26, 34), (29.7, 55), (31.0, 58)]),
]


def camera(t):
    for c in CAMS:
        if c.t0 <= t < c.t1:
            return c.at(t)
    return CAMS[-1].at(t)


# ------------------------------------------------------------------ effects (in world time for their ages)
EV = []


def ev(kind, t, **kw):
    EV.append(dict(kind=kind, t=t, **kw))


GOLD, RED, WHITE, BLUE = (1, 0.78, 0.35), (1, 0.25, 0.12), (1, 1, 1), (0.7, 0.8, 1.0)
# intro storm
for i, t in enumerate((1.14, 2.01)):
    ev('bolt', t, a=[-40 + 30 * i, 70, -60], b=[-30 + 25 * i, -10, -50], seed=10 + i, inten=0.9, col='blue', life=0.22)
ev('smoke', 3.54, p=[-6.3, 1.9, -0.2], col=(0.08, 0.08, 0.1), cnt=6, size=0.5, dark=True)
# Notch's descent and landing
ev('beam', 5.26, x=4.5, z=0, life=1.6)
ev('ring', 6.12, p=[4.5, 0.06, 0], r=7, col=GOLD); ev('sparks', 6.12, p=[4.5, 0.3, 0], col=GOLD, cnt=150, spd=11)
ev('smoke', 6.12, p=[4.5, 0.1, 0], col=(0.35, 0.3, 0.28), cnt=26, size=1.5, dark=True)
ev('bolt', B(6), a=[30, 70, -40], b=[22, -6, -34], seed=21, inten=0.8, col='blue', life=0.2)
# the clash and the lock
ev('sparks', 9.68, p=[0, 1.75, 0], col=WHITE, cnt=160, spd=12); ev('ring', 9.68, p=[0, 1.75, 0], r=4, col=WHITE, vert=0.0001)   # facing the camera (the lock orbit would see it edge-on)
for t in (10.83, 11.26, 11.69):
    ev('sparks', t, p=[0, 1.75, 0], col=GOLD if t != 11.26 else RED, cnt=70, spd=6)
# teleport
ev('smoke', 12.54, p=[-1.8, 0.8, 0], col=(0.05, 0.05, 0.06), cnt=30, size=1.3, dark=True)
ev('smoke', 12.97, p=[3.3, 0.8, 0], col=(0.05, 0.05, 0.06), cnt=26, size=1.2, dark=True)
ev('slash', 13.0, c=[3.1, 1.4, 0], A=[-1, 0.6, 0], Bv=[0, 0, 1], r=1.3, a0=-0.6, a1=2.6, col=RED)
ev('sparks', 13.17, p=[2.6, 1.8, 0], col=RED, cnt=120, spd=9)
# the counter: three cuts
for i, t in enumerate((14.26, 14.69, 15.12)):
    ev('slash', t - 0.12, c=[2.3 + 0.2 * i, 1.4, 0], A=[1, 0.5 - i * 0.6, 0], Bv=[0, 0, 1 if i % 2 else -1], r=1.4, a0=-0.8, a1=2.4, col=GOLD)
    ev('sparks', t, p=[3.2 + 0.2 * i, 1.6, 0], col=GOLD if i < 2 else WHITE, cnt=110 if i < 2 else 160, spd=9 if i < 2 else 13)
ev('ring', 15.12, p=[3.8, 1.5, 0], r=3.5, col=GOLD, vert=1.5708)
ev('smoke', 15.6, p=[9.7, 0.1, -0.3], col=(0.12, 0.11, 0.11), cnt=8, size=0.6, dark=True)
# he rises; red lightning answers
ev('bolt', 16.4, a=[14, 60, -8], b=[11.5, 0, -2.5], seed=33, inten=1.0, col='red', life=0.3)
ev('aura', 16.45, who='H', col=RED, life=12.0)
for i, t in enumerate((17.68, 18.11, 18.54, 18.97)):
    off = [(-0.8, 1.6), (1.2, -1.4), (-1.6, -1.0), (1.4, 1.2)][i]
    ev('bolt', t, a=[2.4 + off[0] * 3, 60, off[1] * 3], b=[2.4 + off[0], 0, off[1]], seed=40 + i, inten=1.0, col='red', life=0.25)
    ev('sparks', t, p=[2.4 + off[0], 0.2, off[1]], col=RED, cnt=90, spd=8); ev('smoke', t, p=[2.4 + off[0], 0.1, off[1]], col=(0.25, 0.2, 0.2), cnt=10, size=0.9, dark=True)
ev('shield', 17.6, who='N', life=1.75)
# mid-air collision and the fall
ev('sparks', 22.40, p=[6.0, 3.6, 0], col=WHITE, cnt=160, spd=14); ev('ring', 22.40, p=[6.0, 3.6, 0], r=7, col=WHITE, vert=1.5708)
ev('ring', 23.25, p=[7.4, 0.06, 0], r=3, col=RED); ev('ring', 23.25, p=[4.6, 0.06, 0], r=3, col=GOLD)
ev('smoke', 23.25, p=[7.4, 0.1, 0], col=(0.3, 0.27, 0.25), cnt=14, size=1.0, dark=True); ev('smoke', 23.25, p=[4.6, 0.1, 0], col=(0.3, 0.27, 0.25), cnt=14, size=1.0, dark=True)
ev('aura', 24.9, who='N', col=GOLD, life=4.4)
for t in (25.83, 26.05):
    ev('bolt', t, a=[10, 55, -6], b=[8.2, 0, -1.3], seed=int(t * 10), inten=0.8, col='red', life=0.2)
# the last clash (frozen) and the blast
ev('sparks', 27.98, p=[6.0, 1.95, 0], col=WHITE, cnt=160, spd=12)
ev('ring', 29.26, p=[6.0, 1.9, 0], r=8, col=WHITE, vert=1.5708); ev('ring', 29.26, p=[6.0, 0.06, 0], r=9, col=GOLD)
ev('sparks', 29.26, p=[6.0, 1.9, 0], col=GOLD, cnt=160, spd=18)
ev('smoke', 29.26, p=[6.0, 0.2, 0], col=(0.3, 0.27, 0.25), cnt=30, size=2.0, dark=True)

LIFE = dict(sparks=0.75, smoke=1.4, ring=0.7, slash=0.45)


def effects(t, H, N):
    w = wtime(t)
    fx = dict(sparks=[], smoke=[], rings=[], bolts=[], slashes=[], auras=[])
    lightning = 0.0
    for e in EV:
        age = w - wtime(e['t'])
        if age < 0:
            continue
        k = e['kind']
        if k == 'sparks' and age < LIFE['sparks']:
            fx['sparks'].append([*e['p'], age, e['t'] * 7.13, *e['col'], e['cnt'], e['spd']])
        elif k == 'smoke' and age < LIFE['smoke']:
            fx['smoke'].append([*e['p'], age, e['t'] * 3.7, *e['col'], e['cnt'], e['size'], e['dark']])
        elif k == 'ring' and age < LIFE['ring']:
            fx['rings'].append([*e['p'], age, e['r'], *e['col'], e.get('vert', 0)])
        elif k == 'bolt' and age < e['life']:
            fl = 1 - age / e['life']
            flick = 1.0 if int(age * 40) % 3 != 1 else 0.35
            fx['bolts'].append([*e['a'], *e['b'], e['seed'], e['inten'] * fl * flick, e['col']])
            lightning = max(lightning, e['inten'] * fl * flick)
        elif k == 'slash' and age < LIFE['slash']:
            prog = age / 0.14
            fx['slashes'].append([*e['c'], *e['A'], *e['Bv'], e['r'], e['a0'], e['a1'], prog, *e['col'], 0.24])
        elif k == 'beam' and age < e['life']:
            fx['beam'] = [e['x'], e['z'], min(1, age / 0.15) * (1 - ease(age, e['life'] - 0.5, e['life']))]
        elif k == 'shield' and age < e['life']:
            q = N['pos']; fx['shield'] = [q[0], q[1] + 1.1, q[2], min(1, age / 0.12) * (1 - ease(age, e['life'] - 0.25, e['life'])) * 0.8, 1.0]
        elif k == 'aura' and age < e['life']:
            q = (H if e['who'] == 'H' else N)['pos']
            fx['auras'].append([q[0], q[1] + 1.1, q[2], 2.6 + 0.2 * math.sin(w * 6), *[c * 0.8 for c in e['col']], 0.14 * min(1, age / 0.4)])
    # the dark, constant flicker of the storm
    lightning = max(lightning, 0.05 * max(0.0, math.sin(w * 2.3) * math.sin(w * 5.1)))
    return fx, lightning


# ------------------------------------------------------------------ titles
TEXTS = [  # id, t_in, t_out, camera-space position, rotation, scale, style
    ('HEROBRINE', 4.62, 5.26, [0, -0.06, -0.55], [0, 14, 0], 0.84, 'slam'),
    ('NOTCH', 6.40, 6.97, [0, 0.5, -2.6], [0, 10, 0], 0.72, 'slam'),
    ('T1', 29.48, 31.0, [0, 0.55, -3.4], [0, -8, 0], 0.86, 'slam'),
    ('T2', 29.68, 31.0, [0, 0.0, -3.4], [0, 0, 0], 0.26, 'slam'),
    ('T3', 29.90, 31.0, [0, -0.55, -3.4], [0, 8, 0], 0.66, 'slam'),
]


def texts(t, cam):
    out = []
    for (tid, t0, t1, pos, rot, sc, style) in TEXTS:
        if t0 <= t <= t1:
            u = (t - t0) / 0.14
            s_in = 1 + 2.2 * (1 - ss(u)) ** 2
            life = (t - t0) / (t1 - t0)
            push = 1 + 0.06 * life
            out_k = ease(t, t1 - 0.2, t1) if tid not in ('T1', 'T2', 'T3') else 0.0
            out.append(dict(id=tid, cam=True, pos=pos, rot=[rot[0], rot[1] * (1 - 0.5 * life), rot[2]], fit=sc, scale=s_in * push * (1 + 0.6 * out_k),
                            opacity=min(1.0, u * 1.4) * (1 - out_k), glow=0.5 + 1.2 * math.exp(-(t - t0) / 0.18)))
    # VS in the world, between them, the camera flies past it
    if 6.97 <= t < 8.68:
        u = ease(t, 6.97, 7.12)
        o = u * (1 - ease(t, 8.15, 8.6))
        out.append(dict(id='VS', pos=[0, 2.3, 0], rot=[0, 90, 0], scale=0.55 * (0.4 + 0.6 * u), opacity=o, glow=0.6 + 1.0 * math.exp(-(t - 6.97) / 0.2)))
    return out


# ------------------------------------------------------------------ the edit pass
HITS = [6.12, 9.68, 13.17, 15.12, 16.4, 22.40, 23.25, 29.26]          # big ones: flash, shake, chromatic aberration


def post(t):
    zoom, shake, flash, ca, rb = 1.0, [0.0, 0.0], 0.0, 0.0018, 0.0
    if t >= 5.2:
        for k in KICKS:
            if k <= t < k + 0.3 and not (28.0 <= k < 29.2):
                d = t - k
                zoom = max(zoom, 1 + 0.055 * math.exp(-d / 0.07))
                ca = max(ca, 0.0018 + 0.006 * math.exp(-d / 0.06))
    for h in HITS:
        d = t - h
        if 0 <= d < 0.6:
            big = 1.8 if h == 29.26 else 1.0
            flash = max(flash, ((0.4 if h == 29.26 else 0.55) if h in (6.12, 9.68, 22.40, 29.26) else 0.3) * math.exp(-d / (0.06 * big)))
            amp = 0.012 * big * math.exp(-d / 0.18)
            shake = [amp * math.sin(d * 97 + h), amp * math.cos(d * 83 + 2 * h)]
            ca = max(ca, 0.012 * big * math.exp(-d / 0.12)); zoom = max(zoom, 1 + 0.08 * big * math.exp(-d / 0.1))
    # zoom-through into the drop, the whip pan of the teleport, the blast
    rb = max(rb, 0.18 * ease(t, 4.95, 5.26) * (t < 5.26))
    rb = max(rb, 0.10 * math.exp(-abs(t - 12.97) / 0.05))
    if 29.26 <= t < 29.8:
        rb = max(rb, 0.22 * math.exp(-(t - 29.26) / 0.15))
    sat, contrast, tint, vig, black, exposure = 1.08, 1.1, [1.0, 1.0, 1.0], 0.42, 0.0, 1.0
    if t < 2.6:
        black = 1 - ease(t, 0.0, 1.0)                       # out of black
    if 28.0 <= t < 29.26:
        sat, contrast, vig = 0.3, 1.25, 0.6                  # the freeze: the colour drains, the glows stay
    if t >= 30.7:
        black = ease(t, 30.7, 31.0)
    if 2.6 <= t < 5.26:
        sat, tint = 0.45, [0.88, 0.96, 1.14]                # the reveal: cold and drained, colour snaps in on the drop
    if 4.6 <= t < 5.26:
        exposure = 0.9
    return dict(zoom=zoom, shake=shake, flash=flash, ca=ca, rblur=rb, sat=sat, contrast=contrast, tint=tint, vig=vig, grain=0.04, black=black, exposure=exposure)


def frame(k):
    t = k / FPS
    H = HB.at(t); N = NB.at(t)
    fx, lightning = effects(t, H, N)
    cam = camera(t)
    return dict(t=wtime(t), cam=cam, H=H, N=N, fx=fx, lightning=lightning, texts=texts(t, cam), post=post(t))


if __name__ == '__main__':
    for t in (0.5, 3.6, 5.0, 6.12, 7.5, 9.68, 11.0, 12.97, 15.12, 17.68, 21.0, 22.4, 26.0, 28.5, 29.6):
        f = frame(int(t * FPS))
        print(t, 'cam', [round(x, 1) for x in f['cam']['pos']], 'H', [round(x, 1) for x in f['H']['pos']], f['H']['visible'], 'N', [round(x, 1) for x in f['N']['pos']],
              'fx', {k: len(v) for k, v in f['fx'].items() if isinstance(v, list) and v}, 'texts', [x['id'] for x in f['texts']], 'zoom %.2f flash %.2f' % (f['post']['zoom'], f['post']['flash']))
