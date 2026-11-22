"""The End Update v2: the timeline.  video time v -> renderView params + 2D overlay state.  Cut to the beat:
  0      - 4.904   the phone: GTA VI at midnight, then the End Update lands; tap -> caption "The End Update"
  4.904  - 9.30    THE END UPDATE in the middle (pulsing on the beat) while the camera slowly pushes in on the Rift Gate;
                   the tape stop freezes it, black
  9.605  - 28.41   the drop: a shot per bar or half bar, every cut on a beat, punch-ins on every beat, whips between places
  28.41  - 29.52   the break, under water: slow motion, the black hole
  29.52  - 34.10   "Girl, tell me now ... it's about to start": the crystals charge the gate; on "start" into the Rift
  34.10  - 36.6    black; GTA VI: "your game is ready to play"
frame(k) -> dict"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 60
SC = json.load(open(os.path.join(HERE, 'web', 'scene.json')))
T_END = 36.6
NFRAMES = int(round(T_END * FPS))
DEG = np.pi / 180
UP = np.array([0, 1.0, 0])
BHC = np.array([0, np.sin(19 * DEG), np.cos(19 * DEG)])


def song_to_v(t): return t - 21.40 + 9.55 if t >= 21.40 else t - 12.0
_b = np.load(os.path.join(HERE, 'beats.npy'))
BEATS = np.array([song_to_v(t) for t in _b if 12.0 < t < 46.2 and not (20.9 < t < 21.40)])
DB = BEATS[BEATS >= 9.6]                     # the drop's beats: DB[0] = 9.605
def kb(k): return float(DB[k])
T_TAP = song_to_v(16.254); T_OPEN = song_to_v(16.904); T_N2 = song_to_v(14.977)
T_TAPE = 8.90; T_SIL = 9.30; T_DROP = kb(0)
T_BREAK = song_to_v(40.26); T_GIRL = song_to_v(41.37); T_START = song_to_v(45.04); T_CUT = song_to_v(45.95)
GC = np.array(SC['gate']['c']); CRY = np.array(SC['crystals']); CHEV = np.array(SC['gate']['chevrons'])


def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def eout(x, p=2.2): x = np.clip(x, 0, 1); return 1 - (1 - x) ** p
def ein(x, p=2.0): x = np.clip(x, 0, 1); return x ** p
def lerp(a, b, u): return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u
def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-9)
def ramp(u, a=0.35, k=2.6):
    """a speed ramp: fast in, then gliding (the edit's 'slow-mo after the hit')"""
    u = np.clip(u, 0, 1); return a * eout(u / 0.25, k) * (u < 0.25) + (a + (1 - a) * (u - 0.25) / 0.75) * (u >= 0.25)


def spline(P, u):
    P = np.asarray(P, float); n = len(P) - 1; u = float(np.clip(u, 0, 1)) * n; i = min(int(u), n - 1); t = u - i
    p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def orbit(c, r, a_deg, y): a = a_deg * DEG; return np.array([c[0] + r * np.cos(a), y, c[2] + r * np.sin(a)])


# ---------------------------------------------------------------- shots: (t0, t1, name, fn(u) -> dict)
def S_intro(u):              # the slow push in on the Rift, under the title
    p = lerp((-4, 15, -215), (0, 25, -100), u ** 0.85); return dict(pos=p, look=lerp((0, 37, 0), (0, 34, 0), u), fov=44 - 4 * u,
        shadowC=(0, 15, -10), shadowR=95, fog=dict(mist=0.007), motes=0.6)
def S_rise(u):               # up past the rim, the light fall beside us, to the gate across the island
    w = ramp(u, 0.45)
    p = spline([(-70, -75, -150), (-46, -32, -132), (-24, 8, -112), (-14, 24, -100)], w)
    l = spline([(-50, -10, -40), (-28, 8, -22), (-5, 28, -5), (0, 33, 0)], w)
    return dict(pos=p, look=l, fov=64 - 14 * w, roll=-10 * (1 - w), shadowC=(10, 10, -20), shadowR=100)
def S_dash(u):               # low and fast down the causeway between the lamps
    p = lerp((2.0, 6.2, -78), (-0.8, 7.8, -30), u ** 0.9)
    return dict(pos=p, look=lerp((0, 15, 0), (0, 24, 0), u), fov=68, roll=-5 + 8 * u, shadowC=(0, 5, -45), shadowR=50, motes=1.4)
def S_orbit_cry(u):          # round a crystal as it fires into the gate
    c = CRY[5]; a = 200 + 95 * ramp(u, 0.4); p = orbit(c, 8.5, a, c[1] - 2 + 4 * u)
    return dict(pos=p, look=c + [0, 0.5, 0], fov=52, shadowC=c - [0, 10, 0], shadowR=40, dof=0.010, focus=8.5)
def S_city(u):               # up the great tower, the skyline and the hole behind it
    w = ramp(u, 0.4)
    p = spline([(-158, 6, -10), (-154, 40, -16), (-149, 78, -22), (-144, 106, -26)], w)
    l = spline([(-175, 22, 30), (-175, 50, 30), (-175, 82, 30), (-176, 96, 36)], w)
    return dict(pos=p, look=l, fov=56 - 4 * w, roll=4 * (1 - w), shadowC=(-175, 50, 30), shadowR=80)
def S_city_orbit(u):         # sweep round the city at the height of its bridges
    a = -128 + 62 * ramp(u, 0.45); p = orbit(SC['city']['c'], 66, a, 62)
    return dict(pos=p, look=(-176, 56, 30), fov=50, roll=-3, shadowC=(-175, 45, 30), shadowR=80)
def S_spires_skim(u):        # low between the crystals, toward the great one
    w = ramp(u, 0.45)
    p = spline([(128, 20, -20), (142, 23, -6), (154, 28, 8)], w)
    return dict(pos=p, look=lerp((176, 44, 40), (177, 56, 40), w), fov=60, roll=6 - 10 * w, shadowC=(160, 25, 20), shadowR=60, motes=1.2)
def S_spires_orbit(u):       # round and up the great crystal, the shards hanging, the hole behind
    w = ramp(u, 0.4); a = -150 + 85 * w; p = orbit((175, 0, 40), 58 - 10 * w, a, 32 + 38 * w)
    return dict(pos=p, look=lerp((176, 52, 40), (178, 74, 40), w), fov=54, shadowC=(175, 40, 40), shadowR=80)
def S_grove(u):              # under the elder tree, the vines, the canopy alight
    w = ramp(u, 0.4); p = lerp((40, -20, -215), (39, -16.5, -203), w)
    return dict(pos=p, look=lerp((29, 0, -172), (30, 5, -172), w), fov=58, roll=-5 + 5 * w, shadowC=(34, -10, -180), shadowR=60, dof=0.004, focus=35,
                fog=dict(mist=0.012, mistY=-30))
def S_fall(u):               # face to face with the light fall, then down with it into the void
    w = ramp(u, 0.35); f = SC['lightfalls'][-1]; lip = np.array(f[:3]); o = np.array([f[3], 0, f[4]])
    p = lip + o * lerp(30, 36, w) + [0, lerp(4, -16, ein(w, 1.2)), 0]
    l = lip + o * lerp(0, 3, w) + [0, lerp(-8, -42, w), 0]
    return dict(pos=p, look=l, fov=58, roll=-4 + 8 * w, shadowC=lip, shadowR=50, fog=dict(mist=0.010))
def S_wide(u):               # everything: the archipelago, the far End, the hole
    w = ease(u) * 0.7 + 0.3 * u
    p = spline([(-60, 40, -120), (-95, 62, -185), (-130, 85, -250)], w)
    return dict(pos=p, look=lerp((0, 45, 30), (20, 70, 90), w), fov=56 + 6 * w, shadowC=(0, 10, 0), shadowR=130, motes=0.0, fog=dict(mist=0.008))
def S_punch(u):              # the hit before the break: straight at the Rift
    p = lerp((0, 31, -110), (0, 31, -52), eout(u, 3)); return dict(pos=p, look=(0, 31, 0), fov=50, shadowC=(0, 15, -20), shadowR=80)
def S_break(u):              # under water: the black hole, slow, soft
    p = lerp((132, 46, -52), (136, 49, -46), u)
    return dict(pos=p, look=p + BHC * 100 + np.array([22, -12, 0]), fov=50, shadowC=(175, 40, 40), shadowR=80, dof=0.012, focus=60, motes=1.5)
def S_charge(u):             # the crystals charge the gate, from the foot of the stairs
    w = ramp(u, 0.3); p = lerp((13, 6.5, -40), (9, 10, -33), w)
    return dict(pos=p, look=lerp((0, 30, 0), (0, 36, 0), w), fov=58, roll=-4 * w, shadowC=(0, 15, -10), shadowR=60)
def S_approach(u):           # rising to the gate's eye, faster and faster
    w = ein(u, 1.7); p = spline([(-20, 12, -120), (-8, 22, -80), (0, 30, -44)], w)
    return dict(pos=p, look=lerp((0, 30, 0), (0, 31, 0), w), fov=56 - 4 * w, roll=-6 + 6 * w, shadowC=(0, 15, -30), shadowR=90)
def S_dive(u):               # in
    k = ein(u, 1.6); p = lerp((0, 31, -44), (0, 31, 8), k); return dict(pos=p, look=(0, 31, 40), fov=56 + 18 * k, roll=25 * k * k, shadowC=(0, 15, -10), shadowR=60)


SHOTS = [(T_OPEN, T_SIL, 'intro', S_intro),
         (kb(0), kb(4), 'rise', S_rise), (kb(4), kb(6), 'dash', S_dash), (kb(6), kb(8), 'cry', S_orbit_cry),
         (kb(8), kb(12), 'city', S_city), (kb(12), kb(14), 'cityorbit', S_city_orbit),
         (kb(14), kb(16), 'skim', S_spires_skim), (kb(16), kb(20), 'spires', S_spires_orbit),
         (kb(20), kb(22), 'grove', S_grove), (kb(22), kb(24), 'fall', S_fall), (kb(24), kb(28), 'wide', S_wide),
         (kb(28), T_BREAK, 'punch', S_punch), (T_BREAK, T_GIRL, 'break', S_break),
         (T_GIRL, BEATS[np.searchsorted(BEATS, T_GIRL) + 2], 'charge', S_charge), (BEATS[np.searchsorted(BEATS, T_GIRL) + 2], T_START, 'approach', S_approach),
         (T_START, T_CUT, 'dive', S_dive)]
# whip pans across these cuts (direction +1 / -1)
WHIPS = {kb(8): 1, kb(14): -1, kb(20): 1, kb(24): -1, kb(12): 1}
WHIP_PRE, WHIP_POST, WHIP_DEG = 0.13, 0.17, 60.0


def shot_index(v):
    for i, (a, b, _, _) in enumerate(SHOTS):
        if a <= v < b: return i
    return None


def rot_y(vec, deg):
    a = deg * DEG; c, s = np.cos(a), np.sin(a); return np.array([vec[0] * c + vec[2] * s, vec[1], -vec[0] * s + vec[2] * c])


def handheld(v, amp):
    return np.array([np.sin(v * 1.7 + 0.3) + 0.5 * np.sin(v * 4.1 + 1.1), 0.7 * np.sin(v * 2.3 + 2.0) + 0.3 * np.sin(v * 5.7), 0.0]) * amp


def cam(v, si=None):
    si = shot_index(v) if si is None else si
    if si is None: return None
    a, b, name, fn = SHOTS[si]
    tv = v
    if name == 'intro' and v > T_TAPE:                       # the tape stop: everything slows to a halt
        x = min(v - T_TAPE, 0.4); tv = T_TAPE + x - x * x / 0.8
    c = fn((tv - a) / (b - a)); c = dict(c)
    pos = np.array(c['pos'], float); look = np.array(c['look'], float); d = look - pos
    yaw = 0.0; roll = c.get('roll', 0.0)
    for tc, dr in WHIPS.items():
        if tc - WHIP_PRE <= v < tc and abs(b - tc) < 1e-6: x = (v - (tc - WHIP_PRE)) / WHIP_PRE; yaw += dr * WHIP_DEG * ein(x, 2.4); roll += dr * 10 * ein(x, 2)
        if tc <= v < tc + WHIP_POST and abs(a - tc) < 1e-6: x = (v - tc) / WHIP_POST; yaw -= dr * WHIP_DEG * (1 - eout(x, 2.4)); roll -= dr * 10 * (1 - eout(x, 2))
    d = rot_y(d, yaw)
    hh = handheld(v, 0.0025 * np.linalg.norm(d)) if name not in ('intro',) else handheld(v * 0.5, 0.0012 * np.linalg.norm(d))
    c['pos'] = pos; c['look'] = pos + d + hh; c['roll'] = roll; c['name'] = name; c['si'] = si
    return c


# ---------------------------------------------------------------- beats -> pulses
DROP_BEATS = [t for t in BEATS if T_DROP - 0.01 <= t < T_BREAK]
LATE_BEATS = [t for t in BEATS if T_GIRL - 0.3 <= t < T_CUT]
INTRO_BEATS = [t for t in BEATS if T_OPEN - 0.01 <= t < T_TAPE]


def pulse(v, beats, tau):
    best = 0.0; down = False
    for i, t in enumerate(beats):
        d = v - t
        if 0 <= d < 6 * tau:
            p = np.exp(-d / tau)
            if p > best: best = p; down = (i % 4 == 0)
    return best, down


def lights_for(c, v, gate):
    look = np.array(c['look'], float); pos = np.array(c['pos'], float); pts = []
    def score(p, w=1.0): p = np.asarray(p, float); return (np.linalg.norm(p - look) * 0.6 + np.linalg.norm(p - pos)) / w
    pts.append((score(GC, 3.0), [*GC, 1.1 * gate, 0.75, 0.3, 1.2, 120]))
    pts.append((score(GC + [0, -14, -14], 2.0), [*(GC + [0, -14, -14]), 0.6 * gate, 0.6, 0.3, 1.2, 50]))
    for k in CRY: pts.append((score(k), [*k, 0.7, 1.0, 0.35, 1.0, 40]))
    for i, f in enumerate(SC['lightfalls']):
        q = np.array(f[:3]) + [f[3] * 3, -4, f[4] * 3]; col = (0.3, 1.0, 1.1) if i == len(SC['lightfalls']) - 1 else (0.9, 0.4, 1.2)
        pts.append((score(q, 1.5), [*q, 0.25, *col, 40]))
    for g in SC['glows']: pts.append((score(g[:3], 1.2), [*g[:3], 0.45, g[4], g[5], g[6], 45]))
    for l in SC['lamps']: pts.append((score(l, 0.8), [*l, 0.22, 1.0, 0.85, 1.1, 22]))
    pts.sort(key=lambda x: x[0]); return [p for _, p in pts[:6]]


def beams_for(v, name):
    out = []
    if name == 'cry':                                         # the orbit crystal fires on the next beat
        t = kb(7); k = ease((v - t) / 0.08)
        if k > 0: out.append([*CRY[5], *CHEV[5], 0.30, k * (1.4 + 0.4 * np.sin(v * 40))])
    if name in ('rise', 'dash'):                              # on the bar, every crystal fires into the gate
        for t in (kb(0), kb(4)):
            d = v - t
            if 0 <= d < 1.2:
                k = ease(d / 0.06) * (1 - ease((d - 0.6) / 0.6))
                for i, c in enumerate(CRY[2:]): out.append([*c, *CHEV[i % len(CHEV)], 0.32, k * (1.2 + 0.3 * np.sin(v * 33 + i))])
    if name in ('charge', 'approach', 'dive'):                # the charge: crystals join in, beat by beat
        n = sum(1 for t in LATE_BEATS if t <= v) * 2
        for i, c in enumerate(CRY[:min(n, len(CRY))]):
            tgt = CHEV[i % len(CHEV)] if i >= 2 else GC + [(-1) ** i * 6, -12, -3]
            out.append([*c, *tgt, 0.34, 1.1 + 0.35 * np.sin(v * 37 + i)])
    return out


def post_at(v, c):
    e = dict(exposure=1.0, vig=0.38, ca=0.0012, grain=0.035, zoom=1.0, zblur=0.0, shake=[0, 0], rot=0.0, flash=0.0, sat=1.0, contrast=1.0,
             tint=[1, 1, 1], bloom=0.08, bars=0.0, streak=0.35, dirt=2.5)
    name = c['name'] if c else None
    if name == 'intro':
        d = v - T_OPEN
        if d < 0.35: e['flash'] = float(np.exp(-d / 0.08))
        p, _ = pulse(v, INTRO_BEATS, 0.09); e['zoom'] += 0.012 * p; e['exposure'] += 0.08 * p
        if v >= T_TAPE:
            k = ease((v - T_TAPE) / (T_SIL - T_TAPE)); e['sat'] = 1 - 0.85 * k; e['exposure'] *= 1 - 0.9 * k; e['zoom'] += 0.05 * k
        return e
    if T_DROP <= v < T_BREAK:
        p, down = pulse(v, DROP_BEATS, 0.085)
        e['zoom'] += (0.055 if down else 0.03) * p; e['zblur'] = (0.22 if down else 0.12) * p ** 1.5; e['ca'] += 0.006 * p
        e['exposure'] += 0.14 * p; e['rot'] = (1.2 if down else -0.7) * p
        sh = handheld(v * 9, 0.0035 * p); e['shake'] = [float(sh[0]), float(sh[1])]
        e['contrast'] = 1.15; e['sat'] = 1.05
    if T_GIRL <= v < T_CUT:
        p, down = pulse(v, LATE_BEATS, 0.09); e['zoom'] += 0.03 * p; e['zblur'] = 0.12 * p ** 1.5; e['exposure'] += 0.12 * p; e['ca'] += 0.004 * p
    for a, b, nm, _ in SHOTS[1:]:                            # a hard cut gets a white pop
        d = v - a
        if 0 <= d < 0.16 and nm not in ('break',): e['flash'] = max(e['flash'], (0.5 if a == T_DROP else 0.18) * (1 - d / 0.16))
    if T_BREAK <= v < T_GIRL:
        d = v - T_BREAK; e.update(bloom=0.13, sat=0.8, vig=0.6, exposure=0.85, tint=[0.85, 0.92, 1.18], contrast=0.9)
        if d < 0.25: e['flash'] = 0.45 * (1 - d / 0.25)
    if v >= T_START:
        k = ein((v - T_START) / (T_CUT - T_START), 2); e['zoom'] += 0.12 * k; e['zblur'] = 0.35 * k; e['lines'] = 0.6 * k; e['lineSeed'] = float(int(v * 30)); e['ca'] += 0.01 * k
        if v > T_CUT - 0.14: e['flash'] = (v - (T_CUT - 0.14)) / 0.14
    return e


def params_at(v, c):
    name = c['name']; r = dict(pos=[float(x) for x in c['pos']], look=[float(x) for x in c['look']], fov=float(c['fov']), roll=float(c.get('roll', 0.0)))
    t = v
    if name == 'intro' and v > T_TAPE: x = min(v - T_TAPE, 0.4); t = T_TAPE + x - x * x / 0.8
    if name == 'break': t = T_BREAK + 0.25 * (v - T_BREAK)
    r['time'] = float(t)
    gate = 0.85; spin = 1.0
    if T_DROP <= v < T_BREAK: p, down = pulse(v, DROP_BEATS, 0.12); gate += 0.45 * p
    if name == 'intro': p, _ = pulse(v, INTRO_BEATS, 0.12); gate += 0.3 * p
    if name in ('charge', 'approach', 'dive'):
        k = (v - T_GIRL) / (T_CUT - T_GIRL); p, _ = pulse(v, LATE_BEATS, 0.12); gate = 0.9 + 1.0 * k + 0.4 * p; spin = 1 + 3.5 * ein(k, 2)
    r['gate'] = float(gate); r['spin'] = float(spin)
    emk = 1.0
    if T_DROP <= v < T_BREAK: p, _ = pulse(v, DROP_BEATS, 0.1); emk += 0.5 * p
    r['emk'] = float(emk)
    r['shadowC'] = [float(x) for x in c.get('shadowC', c['look'])]; r['shadowR'] = float(c.get('shadowR', 80))
    r['fog'] = c.get('fog', {}); r['motes'] = float(c.get('motes', 1.0))
    if c.get('dof'): r['dof'] = float(c['dof']); r['focus'] = float(c['focus'])
    r['lights'] = lights_for(c, v, gate); r['beams'] = beams_for(v, name); r['post'] = post_at(v, c)
    if name == 'dive':
        k = ein((v - (T_START + 0.30)) / 0.30, 1.5); r['dive'] = float(k); r['diveFwd'] = [0, 0, 1.0]; r['diveSpeed'] = 3.0
    r['god'] = 0.3 if name != 'break' else 0.2
    if name == 'intro' and v >= T_SIL: r['post']['black'] = 1.0
    return r


def frame(k):
    v = k / FPS
    out = dict(v=v, phone=None, notif=None, render=None, title=None, caption=None)
    if v < T_OPEN:
        out['phone'] = dict(v=v, tap=T_TAP, dive=T_OPEN, n2=T_N2)
        if v >= T_TAP - 0.02: out['caption'] = dict(v=v, t0=T_TAP, t1=T_OPEN)
        return out
    if v >= T_CUT:
        out['notif'] = dict(v=v, t0=T_CUT + 0.55, t1=T_END - 0.75); return out
    c = cam(v)
    if c is None:                                            # silence before the drop
        out['render'] = None; out['black'] = True; return out
    r = params_at(v, c)
    if c['name'] != 'intro' or v < T_TAPE:                   # camera motion blur (same shot, one frame back)
        pc = cam(v - 1.0 / FPS, c['si'])
        r['prev'] = dict(pos=[float(x) for x in pc['pos']], look=[float(x) for x in pc['look']], fov=float(pc['fov']), roll=float(pc.get('roll', 0.0)))
        r['shutter'] = 0.5
        if any(tc - WHIP_PRE <= v < tc + WHIP_POST for tc in WHIPS): r['shutter'] = 0.9
    out['render'] = r
    if c['name'] == 'intro' and v < T_SIL:
        out['title'] = dict(v=v, t0=T_OPEN, tape=T_TAPE, sil=T_SIL, beats=INTRO_BEATS)
    return out


if __name__ == '__main__':
    print('open %.3f tape %.2f drop %.3f break %.3f girl %.3f start %.3f cut %.3f frames %d' % (T_OPEN, T_TAPE, T_DROP, T_BREAK, T_GIRL, T_START, T_CUT, NFRAMES))
    for a, b, n, _ in SHOTS: print('%-10s %.3f - %.3f (%.2fs)' % (n, a, b, b - a))
