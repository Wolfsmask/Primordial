"""The End Update v2: the timeline.  video time v -> renderView params + 2D overlay state.  Cut to the beat:
  0      - 4.904   the phone: GTA VI at midnight, then the End Update lands; tap -> caption "The End Update"
  4.904  - 9.30    THE END UPDATE in the middle (pulsing on the beat) while the camera slowly pushes in on the Rift Gate;
                   the tape stop freezes it, black
  9.605  - 23.23   the drop (two bars shorter than the song's): a new shot on every kick; a place per bar (the Rift Gate,
                   the End City, the crystal spires, the grove, everything) with a whip pan into each, a punch-in on every beat
  23.23  - 24.34   the break, under water: slow motion, the black hole
  24.34  - 28.92   "Girl, tell me now ... it's about to start": a crystal joins the charge on every beat, a new angle on each;
                   on "start" into the Rift
  28.92  - 30.92   black; GTA VI: "your game is ready to play"
frame(k) -> dict"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 60
SC = json.load(open(os.path.join(HERE, 'web', 'scene.json')))
CUT_A, CUT_B = 24.055873015873015, 29.23392290249433      # song [CUT_A, CUT_B) is spliced out of the drop (end_audio.py)
DEG = np.pi / 180
UP = np.array([0, 1.0, 0])
BHC = np.array([0, np.sin(19 * DEG), np.cos(19 * DEG)])


def song_to_v(t):
    if t >= CUT_B: return t - (CUT_B - CUT_A) - 21.40 + 9.55
    return t - 21.40 + 9.55 if t >= 21.40 else t - 12.0
_b = np.load(os.path.join(HERE, 'beats.npy'))
BEATS = np.array([song_to_v(t) for t in _b if 12.0 < t < 46.2 and not (20.9 < t < 21.40) and not (CUT_A - 0.01 < t < CUT_B - 0.01)])
DB = BEATS[BEATS >= 9.6]                     # the drop's beats: DB[0] = 9.605
def kb(k): return float(DB[k])
T_TAP = song_to_v(16.254); T_OPEN = song_to_v(16.904); T_N2 = song_to_v(14.977)
T_TAPE = 8.90; T_SIL = 9.30; T_DROP = kb(0)
T_BREAK = song_to_v(40.26); T_GIRL = song_to_v(41.37); T_START = song_to_v(45.04); T_CUT = song_to_v(45.95)
T_END = T_CUT + 2.0
NFRAMES = int(round(T_END * FPS))
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
def mv(p0, p1, l0, l1, u, a=0.4, fov=(60, 60), roll=(0, 0), **kw):
    """one beat's move: most of it on the hit, then a glide"""
    w = ramp(u, a); return dict(pos=lerp(p0, p1, w), look=lerp(l0, l1, w), fov=float(lerp(*fov, w)), roll=float(lerp(*roll, w)), **kw)


def S_intro(u):              # the slow push in on the Rift, under the title
    p = lerp((-4, 15, -215), (0, 25, -100), u ** 0.85); return dict(pos=p, look=lerp((0, 37, 0), (0, 34, 0), u), fov=44 - 4 * u,
        shadowC=(0, 15, -10), shadowR=95, fog=dict(mist=0.007), motes=0.6)
# -- bar 1: the Rift Gate
def S_rise(u):               # up over the rim: the gate across the island
    return mv((-40, -4, -118), (-30, 17, -104), (-8, 22, 0), (0, 31, 0), u, 0.5, (60, 52), (-9, 0), shadowC=(0, 12, -30), shadowR=100)
def S_dash(u):               # low and fast down the causeway between the lamps
    return mv((2.0, 6.4, -72), (-0.6, 7.6, -50), (0, 15, 0), (0, 21, 0), u, 0.45, (68, 66), (-5, 2), shadowC=(0, 5, -45), shadowR=50, motes=1.4)
def S_orbit_cry(u):          # round a crystal as it fires into the gate
    c = CRY[5]; w = ramp(u, 0.45); p = orbit(c, 8.5, 205 + 45 * w, c[1] - 2 + 3 * w)
    return dict(pos=p, look=c + [0, 0.5, 0], fov=52, shadowC=c - [0, 10, 0], shadowR=40, dof=0.010, focus=8.5)
def S_gate_low(u):           # under the ring, looking up through it at the hole
    return mv((9, 8.5, -34), (5, 9.0, -27), (0, 58, 10), (0, 62, 12), u, 0.45, (74, 70), (8, 2), shadowC=(0, 20, -10), shadowR=60)
# -- bar 2: the End City
def S_city(u):               # up the great tower
    return mv((-157, 22, -12), (-152, 52, -18), (-175, 36, 30), (-175, 66, 30), u, 0.45, (56, 54), (5, 2), shadowC=(-175, 50, 30), shadowR=80)
def S_city_orbit(u):         # round the city at the height of its bridges
    w = ramp(u, 0.45); p = orbit(SC['city']['c'], 66, -122 + 26 * w, 60)
    return dict(pos=p, look=(-176, 54, 30), fov=50, roll=-3, shadowC=(-175, 45, 30), shadowR=80)
def S_city_street(u):        # from the island's edge, up the great tower between the others
    return mv((-128, 21, 30), (-134, 22.5, 31), (-175, 84, 30), (-175, 93, 30), u, 0.45, (64, 62), (9, 4), shadowC=(-165, 40, 30), shadowR=70)
def S_city_top(u):           # past the crown of the great tower, the hole behind it
    return mv((-166, 96, -14), (-170, 99, -6), (-175, 104, 40), (-176, 107, 50), u, 0.45, (58, 56), (-6, -2), shadowC=(-175, 80, 30), shadowR=60)
# -- bar 3: the crystal spires
def S_spires_skim(u):        # low between the crystals, toward the great one
    return mv((128, 20, -20), (140, 23, -8), (176, 44, 40), (177, 50, 40), u, 0.45, (60, 60), (6, -2), shadowC=(160, 25, 20), shadowR=60, motes=1.2)
def S_spires_orbit(u):       # round the great crystal
    w = ramp(u, 0.45); p = orbit((175, 0, 40), 57 - 3 * w, -150 + 28 * w, 34 + 12 * w)
    return dict(pos=p, look=lerp((176, 52, 40), (177, 60, 40), w), fov=54, shadowC=(175, 40, 40), shadowR=80)
def S_spire_up(u):           # from its foot, up the great crystal to the hole
    return mv((177, 21, -12), (178.5, 22.5, -6), (181, 66, 36), (181, 74, 36), u, 0.45, (70, 66), (-8, -3), shadowC=(178, 40, 30), shadowR=70)
def S_shard_pass(u):         # over the great crystal's tip, down onto the spires
    return mv((198, 106, 12), (193, 101, 19), (177, 32, 44), (176, 27, 44), u, 0.45, (62, 60), (6, 2), shadowC=(175, 30, 40), shadowR=70)
# -- bar 4: the elder grove
def S_grove(u):              # under the elder tree, the vines, the canopy alight
    return mv((40, -20, -214), (39.4, -18, -207), (29, 0, -172), (30, 4, -172), u, 0.45, (58, 58), (-5, 0), shadowC=(34, -10, -180), shadowR=60,
              dof=0.004, focus=35, fog=dict(mist=0.012, mistY=-30))
def S_fall(u):               # face to face with the light fall, then down with it
    w = ramp(u, 0.4); f = SC['lightfalls'][-1]; lip = np.array(f[:3]); o = np.array([f[3], 0, f[4]])
    p = lip + o * lerp(30, 34, w) + [0, lerp(2, -10, w), 0]; l = lip + o * lerp(0, 2, w) + [0, lerp(-10, -30, w), 0]
    return dict(pos=p, look=l, fov=58, roll=float(-4 + 6 * w), shadowC=lip, shadowR=50, fog=dict(mist=0.010))
def S_pool(u):               # low over the pool, to the tree
    return mv((-3, -25.6, -192), (1.5, -25.2, -188), (34, 0, -172), (34, 5, -172), u, 0.45, (68, 66), (6, 1), shadowC=(20, -20, -178), shadowR=50,
              fog=dict(mist=0.012, mistY=-30))
def S_canopy(u):             # the grove from off its edge: the tree, the light fall
    return mv((-30, 0, -235), (-22, 3, -228), (32, -8, -175), (32, -4, -175), u, 0.45, (54, 52), (-6, -2), shadowC=(20, -10, -180), shadowR=70,
              fog=dict(mist=0.010, mistY=-30))
# -- bar 5: everything
def S_wide(u):               # the archipelago, the far End, the hole
    return mv((-60, 40, -120), (-76, 50, -150), (0, 45, 30), (7, 51, 48), u, 0.5, (56, 58), (0, 0), shadowC=(0, 10, 0), shadowR=130, motes=0.0,
              fog=dict(mist=0.008))
def S_tele(u):               # the gate, small, against the hole (long lens)
    return mv((12, 22, -232), (8, 23, -226), (0, 50, 0), (0, 52, 0), u, 0.4, (31, 29), (0, 0), shadowC=(0, 15, -20), shadowR=110, fog=dict(mist=0.006))
def S_gate_side(u):          # round the gate, the crystals on their pillars
    w = ramp(u, 0.45); p = orbit(GC, 92, -38 + 12 * w, 58 + 2 * w)
    return dict(pos=p, look=GC + [0, -2, 0], fov=54, roll=float(-4 + 3 * w), shadowC=(0, 20, 0), shadowR=100)
def S_under(u):              # under the island, looking up the light fall
    f = SC['lightfalls'][0]; lip = np.array(f[:3]); o = np.array([f[3], 0, f[4]])
    return mv(lip + o * 26 + [0, -58, 0], lip + o * 24 + [0, -49, 0], lip + o * 3 + [0, -6, 0], lip + o * 2 + [0, -2, 0], u, 0.45, (72, 70), (10, 4),
              shadowC=lip + [0, -20, 0], shadowR=70, fog=dict(mist=0.008))
def S_punch(u):              # the hit before the break: straight at the Rift
    p = lerp((0, 31, -110), (0, 31, -52), eout(u, 3)); return dict(pos=p, look=(0, 31, 0), fov=50, shadowC=(0, 15, -20), shadowR=80)
def S_break(u):              # under water: the black hole, slow, soft
    p = lerp((132, 46, -52), (136, 49, -46), u)
    return dict(pos=p, look=p + BHC * 100 + np.array([22, -12, 0]), fov=50, shadowC=(175, 40, 40), shadowR=80, dof=0.012, focus=60, motes=1.5)
# -- the charge: a crystal joins on every beat
def S_charge(u):             # from the foot of the stairs
    w = ramp(u, 0.3); p = lerp((13, 6.5, -40), (10, 9, -35), w)
    return dict(pos=p, look=lerp((0, 30, 0), (0, 35, 0), w), fov=58, roll=-4 * w, shadowC=(0, 15, -10), shadowR=60)
def S_charge_c3(u):          # round the crystal that joins now
    c = CRY[3]; w = ramp(u, 0.45); p = orbit(c, 9.0, 115 + 40 * w, c[1] - 1.5 + 3 * w)
    return dict(pos=p, look=c + [0, 0.5, 0], fov=52, shadowC=c - [0, 10, 0], shadowR=40, dof=0.010, focus=9.0)
def S_charge_c5(u):          # the next pair joins: across the pillars to the ring
    return mv((24, 22, -78), (22, 25, -70), (19, 35, -4), (17, 36, -4), u, 0.45, (64, 62), (4, 0), shadowC=(15, 25, -20), shadowR=60)
def S_charge_wide(u):        # all of them into the ring, from above
    return mv((-20, 70, -90), (-16, 72, -82), GC + [0, -1, 0], GC + [0, 1, 0], u, 0.45, (56, 54), (-5, -2), shadowC=(0, 20, -10), shadowR=90)
def S_approach(u):           # rising to the gate's eye, faster and faster
    w = ein(u, 1.5); p = spline([(-12, 18, -100), (-5, 25, -72), (0, 30, -44)], w)
    return dict(pos=p, look=lerp((0, 30, 0), (0, 31, 0), w), fov=56 - 4 * w, roll=-6 + 6 * w, shadowC=(0, 15, -30), shadowR=90)
def S_dive(u):               # in
    k = ein(u, 1.6); p = lerp((0, 31, -44), (0, 31, 8), k); return dict(pos=p, look=(0, 31, 40), fov=56 + 18 * k, roll=25 * k * k, shadowC=(0, 15, -10), shadowR=60)


DROP_SHOTS = [('rise', S_rise), ('dash', S_dash), ('cry', S_orbit_cry), ('gatelow', S_gate_low),
              ('city', S_city), ('cityorbit', S_city_orbit), ('citystreet', S_city_street), ('citytop', S_city_top),
              ('skim', S_spires_skim), ('spires', S_spires_orbit), ('spireup', S_spire_up), ('shards', S_shard_pass),
              ('grove', S_grove), ('fall', S_fall), ('pool', S_pool), ('grovewide', S_canopy),
              ('wide', S_wide), ('tele', S_tele), ('gateside', S_gate_side), ('under', S_under)]
_lb = [t for t in BEATS if T_GIRL < t < T_START]          # the charge's beats: the kick comes back on the 2nd
CB = _lb[1:5]
SHOTS = [(T_OPEN, T_SIL, 'intro', S_intro)]
SHOTS += [(kb(i), kb(i + 1), n, f) for i, (n, f) in enumerate(DROP_SHOTS)]
SHOTS += [(kb(len(DROP_SHOTS)), T_BREAK, 'punch', S_punch), (T_BREAK, T_GIRL, 'break', S_break),
          (T_GIRL, CB[0], 'charge', S_charge), (CB[0], CB[1], 'charge3', S_charge_c3), (CB[1], CB[2], 'charge5', S_charge_c5),
          (CB[2], CB[3], 'chargewide', S_charge_wide), (CB[3], T_START, 'approach', S_approach), (T_START, T_CUT, 'dive', S_dive)]
GATE_DROP = ('rise', 'dash', 'gatelow', 'gateside', 'punch')
CHARGE = ('charge', 'charge3', 'charge5', 'chargewide', 'approach', 'dive')
# whip pans into each new place (direction +1 / -1)
WHIPS = {kb(4): 1, kb(8): -1, kb(12): 1, kb(16): -1}
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
    si = shot_index(v); t0 = SHOTS[si][0] if si is not None else v
    if name == 'cry':                                         # the crystal fires just after the cut
        k = ease((v - t0 - 0.05) / 0.08)
        if k > 0: out.append([*CRY[5], *CHEV[5], 0.30, k * (1.4 + 0.4 * np.sin(v * 40))])
    if name in GATE_DROP:                                     # on the gate's beats every crystal fires into it
        d = v - t0
        if 0 <= d < 0.65:
            k = ease(d / 0.06) * (1 - ease((d - 0.35) / 0.3))
            for i, c in enumerate(CRY[2:]): out.append([*c, *CHEV[i % len(CHEV)], 0.32, k * (1.2 + 0.3 * np.sin(v * 33 + i))])
    if name in CHARGE:                                        # the charge: crystals join in, beat by beat
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
        if 0 <= d < 0.16 and nm not in ('break',):
            f0 = 0.5 if a == T_DROP else (0.2 if any(abs(a - tc) < 1e-6 for tc in WHIPS) else 0.1)
            e['flash'] = max(e['flash'], f0 * (1 - d / 0.16))
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
    if name in CHARGE:
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
        out['notif'] = dict(v=v, t0=T_CUT + 0.45, t1=T_END - 0.55); return out
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
