"""Herobrine VS Notch v6 = v5 with the anime line-teleport, held fight shots he leaves and re-enters, glows on the models.  v5 = v4 with a bar inserted at 4.347 s (the casual straight).  v4 code below runs in v4 time ("old");
the inserted bar runs in v5 time ("new").  v4: video time -> story time (hit-stops), the shot list, effects, the anime edit pass.   frame(k) -> renderFrame params."""
import json, os
import numpy as np
import rig as R

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 60
DUR = 32.60                 # v4 time
T_BLACK = 32.283            # v4 time
SPLIT = 4.347; SH = 4 * 0.4295; T_GO = SPLIT + SH      # v5: [SPLIT, T_GO) is new; v5 time = v4 time + SH after it
DUR5 = DUR + SH
MODE = 'old'                # how story times are read: 'old' = v4 time (mapped), 'new' = v5 time
_d = np.load(os.path.join(HERE, 'story7.npz'))
TS = _d['ts']; DT = TS[1] - TS[0]
POSE = {'H': (_d['HP'], _d['HQ']), 'N': (_d['NP'], _d['NQ'])}
VIS = {'H': _d['Hvis'], 'N': _d['Nvis']}; EYES = {'H': _d['Heyes'], 'N': _d['Neyes']}
_E = json.load(open(os.path.join(HERE, 'story7_events.json')))
SHELL = _E['shell']; XS = _E['xs']


def _old(t): return t - SH if t >= T_GO - 1e-6 else t


def _split(items):
    old, new = [], []
    for e in items:
        if SPLIT - 1e-6 <= e['t'] < T_GO - 1e-6: new.append(dict(e))
        else:
            e = dict(e); e['t'] = _old(e['t'])
            if 't1' in e: e['t1'] = _old(e['t1'])
            old.append(e)
    return old, new


EV, EV5 = _split(_E['events']); HITS, HITS5 = _split(_E['hits'])
HIT = {h['id']: h for h in HITS}
PT = {'kneeR': 'shinR'}
OPP = {'H': 'N', 'N': 'H'}
COL = {'H': (1.0, 0.16, 0.08), 'N': (1.0, 0.78, 0.32)}
UP = np.array([0, 1.0, 0])


def V(m): return 1.77 + 0.4295 * m + (0.0185 if m >= 64 else 0.0)
KICKS = [V(m) for m in range(0, 71) if m < 44 or m >= 56 or m % 4 == 0]


def evs(kind, who=None):
    return [e for e in EV if e['kind'] == kind and (who is None or e.get('who') == who)]


def ev1(kind, who=None):
    e = evs(kind, who); return e[0]['t'] if e else None


def ease(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


def lerp(a, b, u): return np.asarray(a) * (1 - u) + np.asarray(b) * u


# ---------------------------------------------------------------- poses
def s5(s):
    """story time as v5 time"""
    return s if MODE == 'new' else (s + SH if s >= SPLIT else s)


def idx(s):
    i = np.clip(s5(s) / DT, 0, len(TS) - 1.001); i0 = int(i); return i0, i - i0


def pose(F, s):
    P, Q = POSE[F]; i0, w = idx(s)
    q = R.qslerp(Q[i0], Q[i0 + 1], np.full((11, 1), w))
    return P[i0] * (1 - w) + P[i0 + 1] * w, q


_fkc = {}
def fk(F, s):
    k = (F, round(s5(s), 4))
    if k not in _fkc:
        if len(_fkc) > 6000: _fkc.clear()
        _fkc[k] = R.fk(*pose(F, s))
    return _fkc[k]


def pt(F, name, s, w=0.0):
    name = PT.get(name, name)
    if w <= 0: return fk(F, s)[name]
    return (fk(F, s - w)[name] + 2 * fk(F, s)[name] + fk(F, s + w)[name]) / 4


def vis(F, s): return bool(VIS[F][idx(s)[0]])
def eyes(F, s):
    i0, w = idx(s); e = float(EYES[F][i0] * (1 - w) + EYES[F][i0 + 1] * w)
    return float(3.6 * np.tanh(e / 3.6))


def heading(F, s):
    q = pose(F, s)[1][R.BI['pelvis']]; f = R.qrot(q, np.array([0, 0, 1.0])); f[1] = 0
    return f / (np.linalg.norm(f) + 1e-9)


def axis(s, w=0.1):
    d = pt('N', 'chestC', s, w) - pt('H', 'chestC', s, w); d[1] = 0; L = np.linalg.norm(d) + 1e-9
    return d / L, L


def side_of(a): return np.cross(UP, a)


# ---------------------------------------------------------------- time remap: hit-stops (video, story)
def _stops():
    out = []
    big = {'i3': (0.14, 0.42), 'f6': (0.08, 0.26), 't1': (0.12, 0.32), 'palm': (0.07, 0.22), 'bh': (0.12, 0.36), 'd1': (0.20, 0.42),
           'catch': (0.18, 0.44)}
    for h in HITS:
        if h.get('shatter'): continue
        if h['id'] in big: hold, rec = big[h['id']]
        elif h['id'].startswith('f') or h['id'] in ('b1', 'b8', 'b9'): hold, rec = 0.015, 0.055
        elif h['kind'] == 'shell': hold, rec = 0.03, 0.10
        else: hold, rec = 0.025, 0.075
        out.append((h['t'], hold, rec))
    out += [(7.783, 0.05, 0.18), (12.508, 0.08, 0.26), (12.937, 0.06, 0.22), (18.95, 0.05, 0.20), (31.42, 0.04, 0.16)]
    return sorted(out)


STOPS = _stops()


def build_map():
    k = [(0.0, 0.0)]
    for t, hold, rec in STOPS:
        k += [(t, t), (t + hold, t + hold * 0.1), (t + rec, t + rec)]
    k += [(28.399, 28.399), (28.62, 28.44), (28.80, 28.80)]               # the barrier shatters, slowed
    k += [(DUR, DUR)]
    return sorted(k)


SMAP = build_map()


def S(v):
    k = SMAP
    for i in range(len(k) - 1, -1, -1):
        if v >= k[i][0]:
            if i == len(k) - 1: return k[i][1]
            a, b = k[i], k[i + 1]
            if b[0] - a[0] < 1e-9: continue
            return a[1] + (b[1] - a[1]) * (v - a[0]) / (b[0] - a[0])
    return 0.0


# ---------------------------------------------------------------- cameras
def C(pos, look, fov=45, roll=0.0):
    return dict(pos=[float(x) for x in pos], look=[float(x) for x in look], fov=float(fov), roll=float(roll))


def shake(v, amp, f=9.0):
    return np.array([np.sin(v * f * 2.1 + 1.3) + 0.5 * np.sin(v * f * 4.7), 0.6 * np.sin(v * f * 1.7 + 0.4) + 0.4 * np.sin(v * f * 5.3), 0]) * amp


def hand(c, v, amp=0.015, f=1.3):
    c = dict(c); c['pos'] = (np.array(c['pos']) + shake(v, amp, f)).tolist(); return c


def face(F, s, dist=1.5, h=0.0, fov=36, yaw=0.0, tgt='headC', w=0.06):
    hd = pt(F, tgt, s, w); f = heading(F, s)
    c, sn = np.cos(yaw), np.sin(yaw); d = f * c + side_of(f) * sn
    return C(hd + d * dist + UP * h, hd, fov)


def ots(F, s, back=2.4, side=0.9, h=0.4, fov=38, tgt='headC', frm='headC'):
    o = OPP[F]; hd = pt(F, frm, s, 0.06); ot = pt(o, tgt, s, 0.05)
    d = ot - hd; d[1] = 0; d /= np.linalg.norm(d) + 1e-9
    return C(hd - d * back + side_of(d) * side + UP * h, ot, fov)


def two(s, dist=3.0, h=1.2, lookh=1.2, sign=1, fov=42, orbit=0.0, cen=None, w=0.1):
    a, L = axis(s, w)
    m = cen if cen is not None else (pt('H', 'chestC', s, w) + pt('N', 'chestC', s, w)) / 2
    m = np.array(m, float); base = m.copy(); base[1] = 0
    sd = side_of(a) * sign; c, sn = np.cos(orbit), np.sin(orbit); dv = sd * c + a * sn
    hf = np.arctan(np.tan(np.radians(fov) / 2) * 9 / 16)
    need = (0.5 * L * abs(c) + 0.6) / np.tan(hf) if cen is None else 0
    return C(base + dv * max(dist, need) + UP * h, base + UP * lookh if cen is None else m, fov)


def orbit(cen, r, h, ang, fov=45, lookh=0.0):
    return C(cen + np.array([np.cos(ang) * r, h, np.sin(ang) * r]), cen + UP * lookh, fov)


def avoid(c, s, rmin=0.55):
    p = np.array(c['pos'], float); look = np.array(c['look'], float)
    pts = []
    for F in ('H', 'N'):
        if not vis(F, s): continue
        f = fk(F, s); pts += [(f['headC'], 0.5), (f['chestC'], 0.42), (f['belly'], 0.38), (f['fistR'], 0.2), (f['fistL'], 0.2), (f['soleR'], 0.2), (f['soleL'], 0.2)]
    for b in float_blocks(s): pts.append((np.array(b[:3]), b[7] * 0.9 + 0.35))
    for _ in range(4):
        moved = False
        for q, r in pts:
            d = p - q; L = np.linalg.norm(d)
            if L < r and np.linalg.norm(look - q) > 0.05:
                p = q + d / (L + 1e-9) * r; moved = True
        if not moved: break
    p[1] = max(p[1], 0.12)
    c = dict(c); c['pos'] = p.tolist(); return c


NX = 1.0
WALLX = -7.6


def ncen(s): return pt('N', 'chestC', s)


def high(cen, ang, dist=5.2, elev=0.8, fov=50, lookh=-0.25):
    """looking down on Notch from high up: everything round him reads, near side low in the frame, far side high"""
    p = cen + np.array([np.cos(ang) * np.cos(elev), np.sin(elev), np.sin(ang) * np.cos(elev)]) * dist
    return C(p, cen + UP * lookh, fov)


def shot_raw(v, s):
    u = lambda a, b: float(ease((v - a) / (b - a)))
    if v < 1.77:     # behind Herobrine, low: the light coming down through the storm
        c = C(np.array([XS - 1.6 + 0.25 * u(0, 1.77), 0.3, 1.3]), np.array([XS + 2.6, 4.0 - 0.6 * u(0, 1.77), 0.0]), 64, -3)
        return hand(c, v, 0.01)
    if v < 2.63:     # his face, the eyes catching
        c = face('H', s, dist=1.35 - 0.3 * u(1.77, 2.63), h=0.05, fov=34, yaw=0.25); c['roll'] = 2 * (1 - u(1.77, 2.63)); return c
    if v < 3.49:     # Notch coming down: the camera sinks with him
        np_ = pt('N', 'headC', s)
        c = C(np_ + np.array([-2.2, -1.0 + 0.4 * u(2.63, 3.49), 1.3 - 0.3 * u(2.63, 3.49)]), np_ + [0, -0.2, 0], 38)
        return hand(c, v, 0.01)
    if v < 4.10:     # over Herobrine's shoulder: Notch settles in the air, a long way off
        return ots('H', s, back=1.9 - 0.25 * u(3.49, 4.10), side=0.75, h=0.2, fov=40, tgt='chestC')
    if v < 4.347:    # v6: Herobrine side on, all of him: he leans in - and goes
        return h_profile(v)
    if v < 5.206:    # he's gone - behind Notch, beside him, in front: high, round Notch, nothing lands
        return bcam('w1', v)
    if v < 5.80:     # the fist stops dead in front of his face: side view, a slow drift round it
        cen = (pt('H', 'pelvis', 5.206) + pt('N', 'headC', 5.206)) / 2
        a, _ = axis(5.206); base = np.arctan2(side_of(a)[2], side_of(a)[0])
        return orbit(cen, 3.2, 0.0, base + 0.3 * u(5.206, 5.8), 42, 0.0)
    if v < 6.15:     # jab, straight - from the side, low
        return two(s, dist=2.5, h=0.45, lookh=1.3, sign=1, fov=44, orbit=-0.45)
    if v < 7.29:     # everywhere at once: high, turning round Notch
        return bcam('w2a' if v < 6.83 else 'w2b', v)
    if v < 7.783:    # back in front of him; the open hand comes up
        return two(s, dist=2.4, h=0.4, lookh=1.3, sign=-1, fov=44, orbit=0.3)
    if v < 8.75:     # the push: he slides the length of the yard
        hp = pt('H', 'chestC', s, 0.06)
        return C(np.array([-1.2, 1.2, 5.0]), hp + UP * -0.2, 46)
    if v < 9.29:     # stops, low, glaring
        c = face('H', s, dist=2.2, h=-0.2, fov=38, yaw=-0.3, tgt='chestC'); c['look'][1] += 0.35; return hand(c, v, 0.01)
    if v < 10.145:   # flash steps: low, round Notch
        return bcam('w3a1' if v < 9.62 else 'w3a2', v)
    if v < 10.94:    # from above
        return bcam('w3b1' if v < 10.48 else 'w3b2', v)
    if v < 12.078:   # the front combo
        c = two(s, dist=2.3, h=0.3, lookh=1.3, sign=1, fov=46, orbit=-0.55); return hand(c, v, 0.02, 6)
    if v < 12.508:   # the big one; the finger
        return two(s, dist=2.5, h=0.75, lookh=1.55, sign=1, fov=42, orbit=-0.2)
    if v < 12.937:   # the flick: he flies the length of the yard
        hp = pt('H', 'pelvis', s, 0.04)
        return C(np.array([-3.4, 1.4, 6.2]), hp + UP * 0.3, 48)
    if v < 13.796:   # the wall comes apart
        return hand(C(np.array([WALLX + 3.4, 1.0, 3.6]), np.array([WALLX + 0.5, 1.1, 0.0]), 50), v, 0.02, 3)
    if v < 14.655:   # rising out of the rubble
        c = face('H', s, dist=3.0, h=-0.35, fov=40, yaw=0.5, tgt='chestC'); c['look'][1] += 0.3; return hand(c, v, 0.015)
    if v < 15.085:   # Notch, unbothered; Herobrine appears in his face
        return face('N', 15.0, dist=3.0, h=-0.15, fov=42, yaw=1.05, tgt='chestC')
    if v < 15.514:   # stopped in the air: the side view drifts
        cen = (pt('H', 'pelvis', 15.085) + pt('N', 'chestC', 15.085)) / 2
        a, _ = axis(15.085); base = np.arctan2(side_of(a)[2], side_of(a)[0])
        return orbit(cen, 3.4, 0.15, base - 0.3 + 0.4 * u(15.085, 15.514), 44, 0.0)
    if v < 15.944:   # the palm
        return two(s, dist=2.0, h=0.25, lookh=1.0, sign=-1, fov=46, orbit=0.15)
    if v < 16.803:   # the backhand sends him sliding away
        hp = pt('H', 'chestC', s, 0.05)
        return C(np.array([-6.8, 1.6, 1.2]), hp + UP * -0.2, 44)
    if v < 17.662:   # stops, low
        hc = pt('H', 'chestC', s, 0.06); a, _ = axis(s)
        return hand(C(hc + a * 2.5 + side_of(a) * 0.9 + UP * -0.25, hc + UP * 0.1, 42), v, 0.015)
    if v < 18.95:    # calls the storm into his hand
        hp = pt('H', 'pelvis', s, 0.08); f = heading('H', s)
        return C(hp + f * 2.4 + side_of(f) * 1.0 + UP * 0.05, pt('H', 'headC', s, 0.08) + UP * 0.7, 46, 3)
    if v < 19.809:   # the bolt
        hp = pt('H', 'pelvis', s, 0.08); f = heading('H', s)
        return hand(C(hp + f * 5.2 + side_of(f) * 3.0 + UP * 1.6, hp + UP * 2.0, 50), v, 0.03, 4)
    if v < 20.668:   # the roar
        c = face('H', s, dist=2.0, h=-0.1, fov=40, yaw=0.25, tgt='chestC'); c['look'][1] += 0.3; return hand(c, v, 0.03, 7)
    # ---- breakdown: slow, building
    if v < 22.386:
        cen = pt('H', 'chestC', s, 0.1); base = np.arctan2(*heading('H', s)[[2, 0]])
        return orbit(cen, 3.1 - 0.4 * u(20.668, 22.386), 0.1, base + 0.6 - 0.9 * u(20.668, 22.386), 42, 0.05)
    if v < 24.104:   # Notch waits, hands loose
        return face('N', s, dist=2.6 - 0.4 * u(22.386, 24.104), h=-0.3, fov=40, yaw=0.45, tgt='chestC')
    if v < 25.393:
        return ots('H', s, back=3.2, side=1.1, h=0.6, fov=40, tgt='chestC')
    if v < 25.60:
        c = face('H', s, dist=2.2, h=-0.1, fov=40, yaw=0.3, tgt='chestC'); c['look'][1] += 0.3; return c
    if v < 25.822:   # Notch: Herobrine appears in his face
        return face('N', 25.8, dist=3.0, h=-0.1, fov=42, yaw=1.15, tgt='chestC')
    # ---- drop 2
    if v < 26.0:     # the punch that cracks it
        cen = (pt('H', 'pelvis', 25.822) + pt('N', 'chestC', 25.822)) / 2
        a, _ = axis(25.822); base = np.arctan2(side_of(a)[2], side_of(a)[0])
        return orbit(cen, 3.6, 0.2, base + 0.12 * u(25.822, 26.0), 44, 0.0)
    if v < 26.36:
        return two(s, dist=3.0, h=0.6, lookh=1.3, sign=1, fov=44, orbit=-0.3)
    if v < 28.24:    # all round him, the barrier cracking
        return bcam('w5a' if v < 27.33 else 'w5b', v)
    if v < 28.829:   # the barrier shatters
        return two(28.4, dist=3.6, h=0.9, lookh=1.3, sign=1, fov=46, orbit=0.2 + 0.2 * u(28.24, 28.83))
    if v < 29.276:   # lands in front of him, coils
        return two(s, dist=2.6, h=0.3, lookh=1.15, sign=-1, fov=44, orbit=-0.45)
    if v < 29.706:   # caught: side on, both of them, the fist in his open hand in the middle
        c = two(29.3, dist=3.3, h=0.35, lookh=1.3, sign=1, fov=46, orbit=-0.1); return hand(c, v, 0.02, 8)
    if v < 30.565:   # over Herobrine's shoulder, low: his arm, the fist in the open hand, Notch's face above it
        c = ots('H', s, back=2.5 - 0.25 * u(29.706, 30.565), side=-0.8, h=0.05, fov=42, tgt='headC'); c['look'][1] -= 0.3; return hand(c, v, 0.006, 6)
    if v < 30.995:   # Notch's face: nothing
        return face('N', s, dist=2.9 - 0.2 * u(30.565, 30.995), h=-0.15, fov=38, yaw=0.5, tgt='headC')
    if v < 31.853:   # forced down onto his knee
        c = two(31.3, dist=3.6, h=0.3, lookh=1.0, sign=-1, fov=46, orbit=0.6); return hand(c, v, 0.012, 3)
    # his eyes
    return face('N', s, dist=2.6 - 0.35 * u(31.853, T_BLACK), h=-0.32, fov=38, yaw=0.3, tgt='headC')


def shot(v, s):
    return avoid(shot_raw(v, s), s)


CUTS = [1.77, 2.63, 3.49, 4.10, 4.347, 5.206, 5.80, 6.15, 6.83, 7.29, 7.783, 8.75, 9.29, 9.62, 10.145, 10.48, 10.94, 12.078, 12.508, 12.937, 13.796, 14.655, 15.085,
        15.514, 15.944, 16.803, 17.662, 18.95, 19.809, 20.668, 22.386, 24.104, 25.393, 25.60, 25.822, 26.36, 27.33, 28.24, 28.829, 29.276, 29.706,
        30.565, 30.995, 31.853, T_BLACK]


# ---------------------------------------------------------------- the barrier and the effects
def n_center(s):
    p = pt('N', 'pelvis', s); return p + UP * SHELL['dy'], SHELL['rx'], SHELL['ry']


_debris = None


def wall_blocks():
    out = []
    for row in range(3):
        for col in (-1, 0, 1):
            out.append(np.array([WALLX, row + 0.5, col * 1.0]))
    return out


def debris():
    global _debris
    if _debris is not None: return _debris
    z = np.load(os.path.join(HERE, 'debris3.npz')); _debris = (z['ts'], z['pos'], z['quat']); return _debris


T_WALL = ev1('wall_break')


def float_blocks(s):
    out = []
    if not (19.8 <= s < 26.6): return out
    rng = np.random.default_rng(11); c = pt('H', 'pelvis', 20.7); c[1] = 0
    for i in range(12):
        a = rng.uniform(0, 6.283); r = rng.uniform(1.3, 3.2); sz = rng.uniform(0.25, 0.55); t0 = 19.8 + rng.uniform(0, 3.5)
        hmax = rng.uniform(0.6, 2.6); spin = rng.normal(0, 0.7, 3); ph = rng.uniform(0, 6.3)
        k = ease((s - t0) / 1.8)
        if k <= 0: continue
        p = c + np.array([np.cos(a) * r, 0, np.sin(a) * r]) + UP * (sz * 0.5 + (hmax + 0.08 * np.sin(s * 1.3 + ph)) * k)
        if s > 25.822:
            dt = s - 25.822; dirv = np.array([np.cos(a), 0.3, np.sin(a)]); p = p + dirv * (9 * dt) + UP * (-4.9 * dt * dt)
        q = R.qmul(R.qaxis(spin, np.linalg.norm(spin) * (s - t0)), np.array([0, 0, 0, 1.0]))
        out.append([*p, *q, sz])
    return out


def blocks_at(s):
    out = []
    q0 = [0, 0, 0, 1.0]
    if T_WALL is None or s < T_WALL:
        for b in wall_blocks(): out.append([*b, *q0, 1.0])
    elif s < T_WALL + 2.8:
        ts, pos, quat = debris(); i = min(len(ts) - 1, int((s - T_WALL) * 240))
        sc = 1.0 if s < T_WALL + 2.3 else max(0.0, 1 - (s - T_WALL - 2.3) / 0.5)
        for j in range(pos.shape[1]): out.append([*pos[i, j], *quat[i, j], sc])
    out += float_blocks(s)
    return [[float(x) for x in b] for b in out]


def hit_point(h):
    """where the strike meets the barrier (or the body), and the outward normal"""
    t = h['t']; att = h['who']; p = pt(att, h['part'], t)
    if h['kind'] == 'shell':
        c, rx, ry = n_center(t); d = p - c; n = np.array([d[0] / rx ** 2, d[1] / ry ** 2, d[2] / rx ** 2]); n /= np.linalg.norm(n)
        return p, n
    if h['kind'] in ('barrier', 'catch'):
        a = pt('N', 'pelvis', t); b = pt('N', 'headC', t); ab = b - a
        q = a + ab * np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1); n = p - q; n /= np.linalg.norm(n) + 1e-9
        if h['kind'] == 'catch':
            arm = pt('H', 'fistR', t) - pt('H', 'farmR', t); n = -arm / (np.linalg.norm(arm) + 1e-9)
        return p, n
    tgt = pt(OPP[att], h.get('target', 'chestC'), t); n = p - tgt; n /= np.linalg.norm(n) + 1e-9
    return p, n


HP = [(h, *hit_point(h)) for h in HITS]


def streak_mid(a, b, c, rmin=1.1):
    """bend a flash step round Notch rather than through him"""
    m = (a + b) / 2; d = m - c; d[1] = 0; L = np.linalg.norm(d)
    ab = b - a; ab[1] = 0
    if L < rmin:
        if L < 1e-3:
            d = np.cross(UP, ab); L = np.linalg.norm(d) + 1e-9
        m = m + d / L * (rmin - L) * 2.0
    return m


def fx_at(s, v):
    fx = dict(sparks=[], smoke=[], rings=[], bolts=[], glows=[], flights=[], ripples=[], orbs=[], streaks=[], warp=[])
    fx['hand'] = {'H': np.zeros(3), 'N': np.zeros(3)}; fx['rim'] = {'H': np.zeros(3), 'N': np.zeros(3)}; fx['eyesN'] = 0.0
    lightning = 0.0
    shield = None; crack = 0.0
    for (h, p, n) in HP:
        age = s - h['t']; big = h.get('big', 1.0)
        if h['kind'] == 'shell' and age >= 0: crack = max(crack, min(1.0, h.get('crack', 0) * 0.15))
        if h['kind'] == 'barrier' and 0 <= age < 0.4:      # the invisible wall: a small ripple of light just in front of the fist
            fx['ripples'].append([*(p - n * 0.07), *n, age * 1.15, 0.24 + 0.2 * big, 1.0, 0.8, 0.38])
            if age < 0.3: fx['sparks'].append([*(p - n * 0.05), age, h['t'] * 7.3, 1.0, 0.78, 0.35, int(14 + 26 * big), 1.6 + 2.2 * big])
            if big >= 1.4 and age < 0.6:
                fx['rings'].append([*(p - n * 0.07), age, 0.6 * big, 1.0, 0.8, 0.45, float(np.arctan2(n[0], n[2]))])
            if big >= 1.5 and age < 1.0:
                c = pt('N', 'pelvis', h['t']); fx['rings'].append([c[0], 0.04, c[2], age, 2.4 * big, 1.0, 0.7, 0.35, 0])
                fx['smoke'].append([c[0], 0.05, c[2], age, 11.0 + h['t'], 0.13, 0.12, 0.11, 10, 0.7 + 0.15 * big, 1, 0.3])
        if h['kind'] == 'shell' and 0 <= age < 0.6:
            if not (h.get('shatter') and age > 0.06): fx['ripples'].append([*p, *n, age, 0.4 + 0.25 * big, 1.0, 0.8, 0.35])
            if age < 0.45 and (shield is None or age < shield[1]):
                c, rx, ry = n_center(s); shield = ([*c, (0.55 if big < 1.5 else 0.85) * np.exp(-age / 0.25) + 0.16, rx, 0.0, *p, age, ry], age)
            if age < 0.5: fx['sparks'].append([*p, age, h['t'] * 7.3, 1.0, 0.75, 0.3, int(40 + 40 * big), 3 + 2.5 * big])
            if big >= 1.8 and age < 0.6: fx['rings'].append([*p, age, 0.45 * big, 1.0, 0.8, 0.4, float(np.arctan2(n[0], n[2]))])
        if h['kind'] == 'body' and 0 <= age < 0.7:
            fx['sparks'].append([*p, age, h['t'] * 5.1, 1.0, 0.85, 0.45, 90, 7])
            fx['rings'].append([*p, age, 1.8, 1.0, 0.85, 0.5, float(np.arctan2(n[0], n[2]))])
            fx['glows'].append([*p, 0.7 * np.exp(-age / 0.15), 1.4, 1.1, 0.6, 0.6])
        if h['kind'] == 'catch' and 0 <= age < 1.4:          # the fist in his open hand: the air goes out of the yard
            q = pt('N', 'fistR', s)
            fx['rings'].insert(0, [*q, age * 0.9, 2.4, 1.0, 0.85, 0.5, float(np.arctan2(n[0], n[2]))])
            c = pt('N', 'pelvis', h['t']); fx['rings'].append([c[0], 0.04, c[2], age * 0.8, 7.0, 1.0, 0.75, 0.4, 0])
            fx['smoke'].append([c[0], 0.05, c[2], min(age, 1.2), 121.0, 0.13, 0.12, 0.11, 18, 1.2, 1, 0.38])
            if age < 0.7: fx['sparks'].insert(0, [*q, age, 123.0, 1.0, 0.85, 0.45, 140, 8])
            fx['glows'].append([*q, 1.6 * np.exp(-age / 0.12), 1.5, 1.2, 0.7, 0.55])
            fx['hand']['N'] = fx['hand']['N'] + GOLD * (1.2 * np.exp(-age / 0.15) + 0.3)
        if h.get('shatter') and 0 <= age < 1.2:
            fx['sparks'].insert(0, [*p, min(age, 0.69), 501.0, 1.0, 0.85, 0.4, 160, 12])
            fx['sparks'].insert(0, [*n_center(h['t'])[0], min(age, 0.69), 503.0, 1.0, 0.95, 0.7, 160, 8])
            fx['rings'].insert(0, [*p, age * 0.8, 2.2, 1, 0.9, 0.6, float(np.arctan2(n[0], n[2]))])
            fx['glows'].append([*p, 1.8 * max(0, 1 - age / 0.4), 1.6, 1.3, 0.8, 0.6])
    # drop 2: the barrier shows, faintly, cracking with every hit; gone when it shatters
    if 25.82 <= s < 28.399:
        c, rx, ry = n_center(s)
        if shield is None: shield = ([*c, 0.16, rx, crack, *c, 9.0, ry], 9)
        else: shield[0][5] = crack
    if 28.399 <= s < 28.46:
        c, rx, ry = n_center(s); shield = ([*c, 1.2 * (1 - (s - 28.399) / 0.06), rx, 1.0, *c, 0.0, ry], 0)
    if shield is not None: fx['shield'] = shield[0]
    # Notch: the light he comes down in, the dust he blows off the ground, his calm glow
    sn = s5(s)
    if sn < 4.5:
        k = 1.0 if sn < 3.6 else max(0.0, 1 - (sn - 3.6) / 0.9)
        np_ = pt('N', 'chestC', s); fx['rim']['N'] = fx['rim']['N'] + GOLD * (1.4 * k + 0.25)
        fx['sparks'].append([*np_, (sn * 1.5) % 0.7, 700.0 + int(sn * 1.5), 1, 0.8, 0.35, 40, 1.5])
        if vis('N', s): fx['flights'].append([*np_, 12.0 * k + 2, 1.0, 0.75, 0.35])
    if 3.6 <= sn < 5.8:
        age = sn - 3.6; fx['rings'].append([NX, 0.04, 0.0, age * 0.6, 3.5, 1.0, 0.75, 0.35, 0])
        fx['smoke'].append([NX, 0.05, 0.0, min(age, 1.3), 7.0, 0.13, 0.12, 0.11, 14, 0.8, 1, 0.3])
    if s >= 3.5 and vis('N', s):
        fx['rim']['N'] = fx['rim']['N'] + GOLD * 0.22
    # the push and the flick: a blast from his hand
    for kind in ('push', 'flick'):
        for e in evs(kind):
            age = s - e['t']
            if 0 <= age < 0.8:
                p = pt('N', 'fistR', e['t']); a, _ = axis(e['t'])
                fx['glows'].append([*p, (1.6 if kind == 'push' else 2.4) * np.exp(-age / 0.12), 1.5, 1.2, 0.6, 0.85])
                fx['rings'].insert(0, [*p, age, 3.5 if kind == 'push' else 5.0, 1, 0.85, 0.45, float(np.arctan2(a[0], a[2]))])
                fx['sparks'].append([*p, min(age, 0.69), 33.0 + e['t'], 1, 0.85, 0.4, 120, 9])
    # dashes and skids: dust
    for e in evs('skid', 'H'):
        if e['t'] <= s < e['t'] + e['dur'] + 0.6:
            for f in ('soleR', 'soleL'):
                c = pt('H', f, min(s, e['t'] + e['dur']))
                fx['smoke'].append([c[0], 0.05, c[2], (s - e['t']) % 0.5, 71.0 + int((s - e['t']) / 0.5) + (f == 'soleL'), 0.13, 0.12, 0.11, 4, 0.5, 1, 0.35])
            if s < e['t'] + e['dur']:
                c = pt('H', 'soleR', s); fx['sparks'].append([c[0], 0.05, c[2], (s * 3) % 0.4, 80.0 + int(s * 3), 1, 0.6, 0.3, 30, 3])
                c = pt('H', 'fistL', s)
                if c[1] < 0.3: fx['sparks'].append([c[0], 0.05, c[2], (s * 4) % 0.3, 85.0 + int(s * 4), 1, 0.55, 0.3, 20, 2])
    # the wall
    if T_WALL is not None and T_WALL <= s < T_WALL + 2.2:
        age = s - T_WALL
        fx['smoke'].append([WALLX + 0.3, 1.2, 0.0, age, 81.0, 0.12, 0.11, 0.10, 20, 1.2, 1, 0.42])
        fx['sparks'].append([WALLX + 0.5, 1.4, 0.0, min(age, 0.69), 83.0, 1, 0.8, 0.5, 120, 9])
        if age < 0.7: fx['rings'].append([WALLX + 0.5, 1.4, 0.0, age, 4.0, 1, 0.7, 0.4, float(np.pi / 2)])
    # rising from the rubble: red smoke, embers; red fists from here on
    for e in evs('rise', 'H'):
        if e['t'] - 0.2 <= s < e['t'] + e['dur'] + 0.5:
            c = pt('H', 'pelvis', s); age = (s - e['t'] + 0.2) % 0.9
            fx['smoke'].append([c[0], 0.1, c[2], age, 91.0 + int((s - e['t']) / 0.9), 0.05, 0.03, 0.05, 12, 1.1, 1])
            fx['sparks'].append([c[0], c[1] + 0.3, c[2], (s - e['t']) % 0.7, 93.0 + int((s - e['t']) / 0.7), 1, 0.1, 0.05, 70, 3])
    if s >= 13.8 and vis('H', s):
        k = min(1, (s - 13.8) / 0.5) * (1.0 if s < 18.95 else 1.4)
        fx['hand']['H'] = fx['hand']['H'] + RED * 1.6 * k
        fx['flights'].append([*pt('H', 'fistR', s), 0.6 * k, 1.0, 0.2, 0.1])
    # the summoning: crackle at the raised hand, the bolt, the aura; the roar
    ts_ = ev1('summon'); tb = ev1('bolt'); tr = ev1('roar')
    if ts_ is not None and ts_ <= s < tb:
        c = pt('H', 'fistR', s); k = (s - ts_) / (tb - ts_)
        fx['hand']['H'] = fx['hand']['H'] + RED * (2.0 + 4.0 * k) * (0.8 + 0.2 * np.sin(s * 41)); lightning = max(lightning, 0.2 * k * (0.5 + 0.5 * np.sin(s * 37)))
        if int(s * 9) % 3 == 0: fx['bolts'].append([c[0] + 4, 45, c[2] - 3, *c, 300.0 + int(s * 9), 0.08, 'red'])
    if tb is not None and tb <= s < tb + 0.32:
        c = pt('H', 'fistR', s); fx['bolts'].append([c[0] + 3, 60, c[2] - 4, *c, 7.0, 1.0 - (s - tb) / 0.32, 'red'])
        lightning = max(lightning, 1.0 - (s - tb) / 0.32)
    if tb is not None and tb <= s < 25.6:
        c = pt('H', 'chestC', s); k = min(1.0, (s - tb) / 0.4) * (1.0 + 0.6 * ease((s - 20.67) / 5))
        fx['rim']['H'] = fx['rim']['H'] + RED * 0.32 * k
        if int(s * 7) % 4 == 0 and s > 19.4:
            ang = (int(s * 7) * 2.4) % 6.283; p0 = c + np.array([np.cos(ang) * 0.9, 1.2, np.sin(ang) * 0.9])
            fx['bolts'].append([*p0, *(c + np.array([np.cos(ang + 1) * 0.3, -0.6, np.sin(ang + 1) * 0.3])), 400.0 + int(s * 7), 0.0, 'red'])
    if tr is not None and tr <= s < tr + 1.0:
        age = s - tr; c = pt('H', 'pelvis', tr)
        fx['rings'].append([c[0], 0.04, c[2], age, 5.0, 1, 0.25, 0.12, 0]); fx['smoke'].append([c[0], 0.05, c[2], age, 97.0, 0.13, 0.12, 0.11, 14, 1.0, 1, 0.3])
    if 20.668 <= s < 25.822:
        lightning = max(lightning, 0.12 * ease((s - 20.67) / 5) * (0.5 + 0.5 * np.sin(s * 23)))
        for kk in (22.386, 24.104):
            if kk <= s < kk + 0.25:
                c = pt('H', 'chestC', s); fx['bolts'].append([c[0] - 5, 60, c[2] + 3, c[0] + 0.6, 0.0, c[2] + 0.4, 11.0 + kk, 1 - (s - kk) / 0.25, 'red'])
                lightning = max(lightning, 0.9 * (1 - (s - kk) / 0.25))
    # the end: his knee hits the ground; Notch's eyes
    tk = ev1('kneel')
    if tk is not None and tk - 0.1 <= s < tk + 1.0:
        c = pt('H', 'shinR', tk); age = max(0.0, s - tk + 0.1)
        fx['smoke'].append([c[0], 0.05, c[2], age, 131.0, 0.13, 0.12, 0.11, 8, 0.6, 1, 0.35]); fx['rings'].append([c[0], 0.04, c[2], age, 2.2, 1, 0.7, 0.4, 0])
    if s >= 29.276:
        fx['hand']['N'] = fx['hand']['N'] + GOLD * (0.32 + 0.08 * np.sin(s * 9))
    tf = ev1('flare')
    if tf is not None and s >= tf - 0.1:
        k = ease((s - tf + 0.1) / 0.3); hd = pt('N', 'headC', s); q = pose('N', s)[1][R.BI['head']]
        fx['eyesN'] = 2.0 + 3.0 * k
        fx['flights'].insert(0, [*(hd + R.qrot(q, np.array([0.0, 0.02, 0.6]))), 0.45 * k, 1.0, 0.75, 0.35])
    blk = blocks_at(s)
    if blk: fx['blocks'] = blk
    t5 = s5(s); fx['warp'] = warp_lines(t5); st, uu = stretch5(t5)
    if st is not None: fx['str'] = st; fx['rim']['H'] = fx['rim']['H'] + np.array([2.2, 0.7, 0.6]) * uu
    fx['glows'] = fx['glows'][:10]; fx['sparks'] = fx['sparks'][:8]; fx['rings'] = fx['rings'][:4]; fx['flights'] = fx['flights'][:2]; fx['ripples'] = fx['ripples'][:8]
    return fx, lightning


# ---------------------------------------------------------------- the edit pass (video time)
KEY_IMP = [5.206, 12.508, 12.937, 15.085, 15.944, 25.822, 28.399, 29.276]
BIG_V = [1.77, 2.63, 3.49, 5.206, 7.783, 12.078, 12.508, 12.937, 15.085, 15.514, 15.944, 18.95, 19.809, 22.386, 24.104,
         25.822, 26.252, 26.681, 27.11, 27.54, 27.97, 28.399, 29.276, 31.42]
OWNER = {12.508: 'N', 15.514: 'N', 15.944: 'N', 29.276: 'N', 31.42: 'N', 7.783: 'N'}
SMALL_V = [h['t'] for h in HITS if h['kind'] == 'barrier' and h['t'] not in BIG_V]


def post_at(v, s):
    e = dict(zoom=1.0, rblur=0.0, ca=0.0015, shake=[0, 0], flash=0.0, sat=1.05, contrast=1.1, tint=[1, 1, 1], vig=0.4, grain=0.035, exposure=1.0)
    for k in KICKS:
        d = v - k
        if 0 <= d < 0.25:
            p = np.exp(-d / 0.06); e['zoom'] += 0.010 * p; e['ca'] += 0.003 * p
    for t in SMALL_V:                 # every strike that stops: a jolt
        d = v - t
        if 0 <= d < 0.2:
            p = np.exp(-d / 0.05); e['zoom'] += 0.018 * p; e['ca'] += 0.004 * p
            sh = shake(v, 0.004 * np.exp(-d / 0.1), 23); e['shake'] = [e['shake'][0] + sh[0], e['shake'][1] + sh[1]]
    for t in BIG_V:
        d = v - t
        if -0.02 <= d < 0.6:
            big = 1.6 if t in KEY_IMP else 1.0
            p = np.exp(-max(d, 0) / (0.07 * big))
            e['flash'] = max(e['flash'], (0.12 if t >= 25.8 else 0.2) * p)
            e['zoom'] += 0.05 * p * big; e['rblur'] = max(e['rblur'], 0.06 * p * big); e['ca'] += 0.008 * p * big
            sh = shake(v, 0.009 * big * np.exp(-max(d, 0) / 0.18), 23); e['shake'] = [e['shake'][0] + sh[0], e['shake'][1] + sh[1]]
            if t in KEY_IMP and 0 <= d < 3 / FPS + 1e-6:
                e['impact'] = 1.0; e['impInv'] = 1.0 if int(d * FPS) % 2 == 0 else 0.0; e['flash'] = 0.0
            if 0 <= d < 0.3:
                e['lines'] = max(e.get('lines', 0), 0.8 * (1 - d / 0.3)); e['lineSeed'] = float(int(v * 30))
    for a, b in ((4.347, 5.206), (6.15, 7.29), (9.29, 10.94), (12.508, 12.937), (14.655, 15.085), (25.60, 25.822), (26.36, 28.24)):
        if a <= v < b: e['lines'] = max(e.get('lines', 0), 0.4); e['lineSeed'] = float(int(v * 30))
    if v < 1.77: e['exposure'] = 0.7 + 0.25 * ease((v - 0.4) / 1.2); e['sat'] = 0.8; e['tint'] = [0.9, 0.94, 1.1]
    if 1.77 <= v < 5.206: e['tint'] = [1.02, 0.98, 0.98]
    if 18.95 <= v < 20.668: e['tint'] = [1.1, 0.88, 0.88]; e['contrast'] = 1.16
    if 20.668 <= v < 25.822:
        w = ease((v - 20.668) / 5.15)
        e['contrast'] = 1.2; e['sat'] = 0.95 + 0.2 * w; e['vig'] = 0.55; e['bars'] = 0.07; e['tint'] = [1.08, 0.92, 0.9]
        e['shake'] = list(np.array(e['shake']) + shake(v, 0.002 + 0.006 * w, 31)[:2])
        if v > 25.62: e['flash'] = max(e['flash'], (v - 25.62) / 0.2 * 0.45)
    if 25.822 <= v < T_BLACK:
        e['contrast'] = 1.18; e['sat'] = 1.15; e['bars'] = 0.055; e['ca'] += 0.002
        for t in BIG_V:
            d = v - t
            if t >= 25.8 and 0.03 <= d < 0.2:
                who = OWNER.get(t, 'H')
                e['duo'] = 0.35 * (1 - (d - 0.03) / 0.17)
                e['duoA'] = [0.02, 0.0, 0.0] if who == 'H' else [0.05, 0.03, 0.0]
                e['duoB'] = [1.0, 0.25, 0.15] if who == 'H' else [1.0, 0.85, 0.4]
    if 29.706 <= v < T_BLACK:         # the stand-off: quieter, colder, the gold of his hand
        w = ease((v - 29.706) / 0.4); e['sat'] = 1.15 - 0.2 * w; e['vig'] = 0.4 + 0.2 * w
        if v >= 31.853: e['exposure'] = 1.0 + 0.15 * ease((v - 31.853) / 0.43); e['vig'] = 0.7
    for c in CUTS:
        d = v - c
        if 0 <= d < 0.06 and c >= 25.8: e['rblur'] = max(e['rblur'], 0.1 * (1 - d / 0.06))
    if v >= T_BLACK: e['black'] = 1.0; e['bars'] = 0.055
    if v < 0.2: e['black'] = 1 - v / 0.2
    return e


def texts_at(v):
    out = []
    def slam(id_, t0, t1, pos, rot, fit, glow=1.0):
        if t0 <= v < t1:
            u = ease((v - t0) / 0.12); o = 1 - ease((v - (t1 - 0.12)) / 0.12)
            push = 1.0 + 0.04 * (v - t0)
            out.append(dict(id=id_, cam=True, pos=pos, rot=rot, fit=fit, scale=(1.6 - 0.6 * u) * push, opacity=float(u * o), glow=glow))
    slam('HEROBRINE', 1.92, 2.63, [0, -0.62, -2.6], [0, 12, 0], 0.84)
    slam('NOTCH', 2.70, 3.47, [0, 0.66, -2.6], [0, -10, 0], 0.7)
    slam('VS', 3.55, 4.33, [0, 0.45, -3.2], [0, 0, 0], 0.42)
    return out


# ---------------------------------------------------------------- afterimages on the fast moves
GHOST_WIN = [(4.47, 5.21, 'H'), (5.77, 7.56, 'H'), (9.44, 12.08, 'H'), (12.51, 12.94, 'H'), (14.99, 15.09, 'H'), (15.30, 15.96, 'N'),
             (25.74, 25.83, 'H'), (26.16, 28.45, 'H'), (29.10, 29.28, 'H')]


def ghosts(F, s, v):
    if not any(a <= s < b and F == f for a, b, f in GHOST_WIN) or not vis(F, s): return []
    out = []
    f0 = fk(F, s)
    for j, lag in enumerate((0.025, 0.05, 0.075)):
        if not vis(F, s - lag): continue
        f1 = fk(F, s - lag)
        if max(np.linalg.norm(f0[k] - f1[k]) for k in ('pelvis', 'fistR', 'fistL', 'soleR', 'soleL')) < 0.16: continue
        P, Q = pose(F, s - lag)
        out.append(dict(P=[float(x) for x in P], Q=[[float(x) for x in q] for q in Q], a=0.3 - 0.08 * j, c=list(COL[F])))
    return out


def clean(o):
    if isinstance(o, dict): return {k: clean(x) for k, x in o.items()}
    if isinstance(o, (list, tuple)): return [clean(x) for x in o]
    if isinstance(o, (np.floating, float)): return float(o)
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, np.ndarray): return clean(o.tolist())
    if isinstance(o, np.bool_): return bool(o)
    return o


# ---------------------------------------------------------------- v5: the inserted bar (v5 time): the casual straight
def V5(m): return 1.77 + 0.4295 * m + (0.0185 if m >= 68 else 0.0)


T_CAS = V5(7)
C0 = HITS5[0]
SK5 = [(SPLIT, SPLIT), (T_CAS, T_CAS), (T_CAS + 0.10, T_CAS + 0.012), (T_CAS + 0.40, T_CAS + 0.40), (T_GO, T_GO)]


def S5(v):
    for a, b in zip(SK5[:-1], SK5[1:]):
        if a[0] <= v < b[0]: return a[1] + (b[1] - a[1]) * (v - a[0]) / (b[0] - a[0])
    return v


def shot5(v, s):
    u = lambda a, b: float(ease((v - a) / (b - a)))
    if v < T_SW:             # v6: the over-the-shoulder holds: he turns into lines and goes
        return h_profile(v)
    if v < T_CAS + 0.10:     # side on, where he lands: the lines come in, he's there, the lazy straight goes out
        return hand(two(T_CAS, dist=3.2 + 0.25 * (1 - u(T_SW, T_CAS)), h=0.5, lookh=1.25, sign=-1, fov=42, orbit=0.1), v, 0.008)
    if v < 5.27:             # close: the fist stopped dead in front of him, the air rippling round it
        t = T_CAS; p = pt('H', 'fistR', t); a, _ = axis(t); sd = -side_of(a)
        cen = p + a * 0.12 + UP * 0.04
        dist = 3.3 - 0.4 * u(T_CAS + 0.10, 5.27); dv = sd * np.cos(0.72) - a * np.sin(0.72)
        return hand(C(cen + dv * dist + UP * 0.3, cen, 32), v, 0.005)
    if v < 5.86:             # he looks at his fist
        c = face('H', s, dist=2.9 - 0.25 * u(5.27, 5.86), h=-0.05, fov=36, yaw=-1.05, tgt='headC'); c['look'][1] -= 0.14; return hand(c, v, 0.005)
    c = face('H', s, dist=2.1 - 0.2 * u(5.86, T_GO), h=0.02, fov=34, yaw=-1.1, tgt='headC')      # and back up at him
    return hand(c, v, 0.006)


def casual_fx(fx, s):
    t = C0['t']; p = pt('H', 'fistR', t)
    a = pt('N', 'pelvis', t); b = pt('N', 'headC', t); ab = b - a
    q = a + ab * np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1); n = p - q; n /= np.linalg.norm(n) + 1e-9
    pn = pt('H', 'fistR', min(s, 5.22))
    for t0, sz, k in ((t, 0.46, 1.0), (4.96, 0.32, 0.65), (5.12, 0.28, 0.55)):
        age = s - t0
        if 0 <= age < 0.4: fx['ripples'].append([*(pn - n * 0.07), *n, age * 1.15, sz, 1.0 * k, 0.8 * k, 0.38 * k])
    age = s - t
    if 0 <= age < 0.3: fx['sparks'].append([*(p - n * 0.05), age, 41.7, 1.0, 0.78, 0.35, 22, 2.4])


def post5(v, s):
    e = dict(zoom=1.0, rblur=0.0, ca=0.0015, shake=[0, 0], flash=0.0, sat=1.05, contrast=1.1, tint=[1.02, 0.98, 0.98], vig=0.4, grain=0.035, exposure=1.0)
    e = open_post(e, v)
    for m in range(4, 11):
        d = v - V5(m)
        if 0 <= d < 0.25: p = np.exp(-d / 0.06); e['zoom'] += 0.010 * p; e['ca'] += 0.003 * p
    d = v - T_CAS
    if -0.02 <= d < 0.6:
        p = np.exp(-max(d, 0) / 0.09)
        e['zoom'] += 0.04 * p; e['rblur'] = 0.03 * p; e['ca'] += 0.007 * p; e['flash'] = 0.1 * p
        sh = shake(v, 0.006 * np.exp(-max(d, 0) / 0.15), 23); e['shake'] = [float(sh[0]), float(sh[1])]
    if v >= 5.86:            # his eyes: a push in, a little red
        w = float(ease((v - 5.86) / 0.2)); e['zoom'] += 0.03 * w; e['tint'] = [1.02 + 0.06 * w, 0.98 - 0.04 * w, 0.98 - 0.04 * w]; e['vig'] = 0.4 + 0.15 * w
    return e


def frame_new(v):
    s = float(S5(v))
    out = dict(t=s, v=v)
    out['cam'] = avoid(shot5(v, s), s)
    for F in ('H', 'N'):
        P, Q = pose(F, s)
        out[F] = dict(P=[float(x) for x in P], Q=[[float(x) for x in q] for q in Q], vis=vis(F, s), eyes=eyes(F, s))
    out['ghH'] = []; out['ghN'] = []
    fx, lightning = fx_at(s, v)
    glow_out(out, fx)
    casual_fx(fx, s)
    fx['ripples'] = fx['ripples'][:8]; fx['sparks'] = fx['sparks'][:8]; fx['rings'] = fx['rings'][:4]
    out['fx'] = fx; out['lightning'] = float(lightning)
    out['post'] = post5(v, s); out['texts'] = []
    out['fill'] = 0.6; out['fireK'] = 0.55
    return clean(out)


def frame(k):
    global MODE, EV, HP
    v5 = k / FPS
    if SPLIT <= v5 < T_GO:
        MODE = 'new'; evo, hpo = EV, HP; EV, HP = EV5, []
        try: return frame_new(v5)
        finally: MODE = 'old'; EV, HP = evo, hpo
    v = v5 - SH if v5 >= T_GO else v5
    s = float(S(v))
    out = dict(t=s5(s), v=v5)
    out['cam'] = shot(v, s)
    for F in ('H', 'N'):
        P, Q = pose(F, s)
        out[F] = dict(P=[float(x) for x in P], Q=[[float(x) for x in q] for q in Q], vis=vis(F, s), eyes=eyes(F, s))
    out['ghH'] = ghosts('H', s, v); out['ghN'] = ghosts('N', s, v)
    fx, lightning = fx_at(s, v)
    glow_out(out, fx)
    out['fx'] = fx; out['lightning'] = float(lightning)
    out['post'] = post_at(v, s)
    out['texts'] = texts_at(v)
    out['fill'] = 0.6
    out['fireK'] = 0.55
    return clean(out)


# ================================================================ v6
GOLD = np.array([1.0, 0.75, 0.35])
RED = np.array([1.0, 0.09, 0.035])
T_SW = 4.43                 # the opening: cut from the over-the-shoulder to the side-on two shot (he is gone in between)
T_ARR0 = 4.64               # ... and he is there (story7 T_A)


def h_profile(v):
    """the opening: Herobrine side on, full length, looking along the way he will go (+x = screen right)"""
    c = fk5('H', 4.2)['chestC']; k = float(ease((v - 4.10) / (T_SW - 4.10)))
    return C(c + np.array([0.35, 0.5, 4.4 - 0.3 * k]), c + np.array([0.35, 0.3, 0.0]), 40)


def glow_out(out, fx):
    """the glows that live on the models: fists, rim light, eyes, the teleport stretch"""
    for F in ('H', 'N'):
        out[F]['hand'] = [float(x) for x in fx['hand'][F]]; out[F]['rim'] = [float(x) for x in fx['rim'][F]]
    if fx['eyesN'] > 0: out['N']['eyes'] = max(out['N']['eyes'], float(fx['eyesN']))
    if 'str' in fx: out['H']['str'] = [float(x) for x in fx.pop('str')]
    del fx['hand'], fx['rim'], fx['eyesN']


# ---------------------------------------------------------------- poses straight from v5 story time (whatever MODE is)
_fk5 = {}


def fk5(F, t):
    k = (F, round(float(t), 4))
    if k not in _fk5:
        if len(_fk5) > 4000: _fk5.clear()
        P, Q = POSE[F]; i = float(np.clip(t / DT, 0, len(TS) - 1.001)); i0 = int(i); w = i - i0
        q = R.qslerp(Q[i0], Q[i0 + 1], np.full((11, 1), w))
        _fk5[k] = R.fk(P[i0] * (1 - w) + P[i0 + 1] * w, q)
    return _fk5[k]


def inv_S(t):
    lo, hi = t - 1.2, t + 0.05
    for _ in range(40):
        m = (lo + hi) / 2
        if S(m) < t: lo = m
        else: hi = m
    return hi


def cam5(t5):
    """the camera the edit is on at v5 story time t5"""
    global MODE, EV, HP
    if SPLIT <= t5 < T_GO:
        MODE = 'new'; evo, hpo = EV, HP; EV, HP = EV5, []
        try: return shot5(t5, t5)
        finally: MODE = 'old'; EV, HP = evo, hpo
    mo = MODE; MODE = 'old'
    try:
        t = _old(t5); return shot(inv_S(t), t)
    finally: MODE = mo


def proj(c, X):
    P = np.array(c['pos'], float); L = np.array(c['look'], float); f = L - P; f /= np.linalg.norm(f)
    r = np.cross(f, UP); r /= np.linalg.norm(r); u = np.cross(r, f)
    X = np.atleast_2d(X); d = X - P; z = d @ f; th = np.tan(np.radians(c['fov']) / 2)
    n = np.stack([(d @ r) / (np.maximum(z, 1e-3) * th * 9 / 16), (d @ u) / (np.maximum(z, 1e-3) * th)], 1)
    return n, z, r, u, f


# ---------------------------------------------------------------- the teleports: (gone, back, opening?) in v5 story time
TPS = []
_pend = None
for _e in sorted([e for e in _E['events'] if e.get('who') == 'H' and e['kind'] in ('vanish', 'appear', 'flash')], key=lambda e: e['t']):
    if _e['kind'] == 'flash': TPS.append((_e['t'], _e['t1'], False))
    elif _e['kind'] == 'vanish': _pend = _e
    elif _pend is not None: TPS.append((_pend['t'], _e['t'], bool(_pend.get('open')))); _pend = None
_dirs = {}


def tp_dirs(i):
    """which way the lines go: out of the frame when he goes, in from the frame edge when he comes back"""
    if i in _dirs: return _dirs[i]
    t0, t1, op = TPS[i]
    a = fk5('H', t0 - 0.006); b = fk5('H', t1 + 0.006)
    if op:                                   # the opening: straight at Notch, both ways
        d = fk5('N', t0)['chestC'] - a['chestC']; d = d / np.linalg.norm(d); _dirs[i] = (d, d); return _dirs[i]
    out = []
    for t, f, sgn in ((t0 - 0.02, a, 1.0), (t1 + 0.01, b, -1.0)):
        c = cam5(t); n, z, r, u, fw = proj(c, f['chestC']); n = n[0]
        nN, zN = proj(c, fk5('N', t)['chestC'])[:2]; nN = nN[0]
        rad = n - (nN if (np.abs(nN) < 1).all() and zN[0] > 0 else 0)
        rad = rad * np.array([1.0, 0.5]) + np.array([0, 0.25])          # sideways rather than down: the classic smear
        if np.linalg.norm(rad) < 0.3 or z[0] < 3.0: rad = np.array([1.0 if n[0] >= 0 else -1.0, 0.35])
        rad = rad / np.linalg.norm(rad)
        w = r * rad[0] + u * rad[1] + fw * 0.15; w = w / np.linalg.norm(w)
        out.append(w * sgn)
    _dirs[i] = tuple(out); return _dirs[i]


def _lin(c): return (np.asarray(c, float) / 255.0) ** 2.2 * 1.7


HCOL = dict(hair=_lin((41, 28, 12)), face=_lin((124, 89, 68)), shirt=_lin((18, 145, 158)), sleeve=_lin((0, 169, 169)),
            skin=_lin((164, 121, 99)), pants=_lin((76, 66, 154)), shoe=_lin((88, 82, 136)))
SEGS = [('headC', 'headC', 0.2, 6, 'head'), ('chestC', 'pelvis', 0.16, 10, 'shirt'), ('uarmR', 'farmR', 0.08, 3, 'sleeve'),
        ('uarmL', 'farmL', 0.08, 3, 'sleeve'), ('farmR', 'fistR', 0.08, 3, 'skin'), ('farmL', 'fistL', 0.08, 3, 'skin'),
        ('thighR', 'shinR', 0.09, 4, 'pants'), ('thighL', 'shinL', 0.09, 4, 'pants'), ('shinR', 'soleR', 0.09, 4, 'shoe'),
        ('shinL', 'soleL', 0.09, 4, 'shoe')]
_lines = {}


def tp_lines(i, which):
    """the lines he turns into: points over his body, each with his colour there; a few white-hot ones on top"""
    key = (i, which)
    if key in _lines: return _lines[key]
    t0, t1, op = TPS[i]
    f = fk5('H', t0 - 0.006 if which == 0 else t1 + 0.006)
    rng = np.random.default_rng(1000 + i * 7 + which)
    d = tp_dirs(i)[which]
    pp = np.cross(d, UP); pp = pp / (np.linalg.norm(pp) + 1e-9); qq = np.cross(d, pp)
    W = np.array([s[3] for s in SEGS], float); W /= W.sum()
    out = []
    for j in range(64):
        add = j >= 50
        a, b, rad, _, nm = SEGS[rng.choice(len(SEGS), p=W)]
        p = f[a] + (f[b] - f[a]) * rng.uniform() + rng.uniform(-rad, rad, 3)
        if add: col = np.array([2.6, 2.2, 2.1]) if rng.uniform() < 0.5 else np.array([2.4, 0.35, 0.18])
        elif nm == 'head': col = HCOL['hair'] if rng.uniform() < 0.5 else HCOL['face']
        else: col = HCOL[nm]
        jit = (pp * rng.normal(0, 0.035) + qq * rng.normal(0, 0.035))
        out.append(dict(p=p, col=col, w=rng.uniform(0.010, 0.032) * (0.6 if add else 1.0), dl=rng.uniform(0, 0.04),
                        vh=rng.uniform(55, 105) * (1.2 if op else 1.0), lam=rng.uniform(0.025, 0.07), jit=jit, add=1 if add else 0))
    _lines[key] = out; return out


def warp_lines(t5):
    out = []
    for i, (t0, t1, op) in enumerate(TPS):
        if t0 - 0.04 <= t5 < t0 + 0.2:          # gone: he stretches into lines and they shoot off out of the frame
            d = tp_dirs(i)[0]
            for L in tp_lines(i, 0):
                age = t5 - (t0 - 0.035 + L['dl'])
                if age < 0: continue
                a = 1.0 if age < 0.07 else 1 - (age - 0.07) / 0.06
                if a <= 0: continue
                dd = d + L['jit']; dd = dd / np.linalg.norm(dd)
                h = L['p'] + dd * L['vh'] * age; tl = L['p'] + dd * L['vh'] * max(0.0, age - L['lam'])
                out.append([*tl, *h, L['w'] * (1.0 if age < 0.05 else 0.7), *L['col'], a, L['add']])
        if t1 - 0.17 <= t5 < t1 + 0.1:          # back: lines come in from off the frame and pull together into him
            d = tp_dirs(i)[1]
            for L in tp_lines(i, 1):
                Ti = t1 - 0.010 - L['dl']
                if t5 < Ti - 0.12 or t5 > Ti + L['lam']: continue
                dd = d + L['jit']; dd = dd / np.linalg.norm(dd)
                h = L['p'] - dd * L['vh'] * max(0.0, Ti - t5); tl = L['p'] - dd * L['vh'] * max(0.0, Ti + L['lam'] - t5)
                a = min(1.0, (t5 - (Ti - 0.12)) / 0.03)
                out.append([*tl, *h, L['w'], *L['col'], a, L['add']])
    return out[:240]


def stretch5(t5):
    """just before he goes he is pulled out along the way he is going; when he lands he snaps back from it"""
    for i, (t0, t1, op) in enumerate(TPS):
        if t0 - 0.035 <= t5 < t0:
            u = (t5 - (t0 - 0.035)) / 0.035; d = tp_dirs(i)[0]; c = fk5('H', t5)['chestC']
            return [*c, *d, 1 + 1.2 * u * u, 1 - 0.35 * u], u
        if t1 <= t5 < t1 + 0.06:
            u = 1 - (t5 - t1) / 0.06; d = tp_dirs(i)[1]; c = fk5('H', t5)['chestC']
            return [*c, *d, 1 + 1.2 * u * u, 1 - 0.35 * u], u
    return None, 0.0


# ---------------------------------------------------------------- the fight teleports: held shots on Notch he leaves and comes back into
BW = {'w1': (4.347, 5.206), 'w2a': (6.15, 6.83), 'w2b': (6.83, 7.29), 'w3a1': (9.29, 9.62), 'w3a2': (9.62, 10.145), 'w3b1': (10.145, 10.48), 'w3b2': (10.48, 10.94),
      'w5a': (26.36, 27.33), 'w5b': (27.33, 28.24)}
BCF = os.path.join(HERE, 'bcam7.json')
_bc = json.load(open(BCF)) if os.path.exists(BCF) else {}


def _bsearch(v0, v1, front=True):
    pts, chs = [], []
    for v in np.arange(v0, v1, 0.02):
        s = S(v)
        if vis('H', s):
            f = fk('H', s); pts += [f[k] for k in ('headC', 'fistR', 'fistL', 'soleR', 'soleL', 'chestC')]; chs.append(f['chestC'])
    fN = fk('N', S((v0 + v1) / 2)); pts += [fN[k] for k in ('headC', 'soleR', 'soleL', 'fistR', 'fistL')]
    pts = np.array(pts); chs = np.array(chs); nC = fN['chestC']
    look = nC * 0.55 + chs.mean(0) * 0.45; look[1] = nC[1] - 0.05
    best = None
    for e in (0.10, 0.22, 0.34):
        for cdeg in (range(100, 265, 5) if front else range(0, 360, 5)):
            c = np.radians(cdeg); dv = np.array([np.cos(e) * np.cos(c), np.sin(e), np.cos(e) * np.sin(c)])
            for D in np.arange(3.8, 8.61, 0.2):
                cam = C(look + dv * D, look, 46)
                n, z = proj(cam, pts)[:2]
                if (z < 0.8).any() or (np.abs(n[:, 0]) > 0.80).any() or (np.abs(n[:, 1]) > 0.86).any(): continue
                nh, zh = proj(cam, chs)[:2]; nn, zn = proj(cam, nC)[:2]
                wN = 0.32 / (zn[0] * np.tan(np.radians(23)) * 9 / 16)
                occ = float(np.mean((zh > zn[0]) & (np.abs(nh[:, 0] - nn[0, 0]) < wN) & (np.abs(nh[:, 1] - nn[0, 1]) < 0.35)))
                sc = D + 4.0 * occ + 0.15 * e + 1.4 * (1 + np.cos(c)) / 2      # rather his front than his back
                if best is None or sc < best[0]: best = (sc, cam, D, occ, cdeg, e)
                break
    return best if best is not None or not front else _bsearch(v0, v1, False)


def bcam(name, v):
    if name not in _bc:
        mo = MODE
        b = _bsearch(*BW[name]); _bc[name] = dict(cam=b[1], D=float(b[2]), occ=b[3], az=b[4], el=b[5])
        try: json.dump(_bc, open(BCF, 'w'), indent=1)
        except OSError: pass
    c = _bc[name]['cam']; v0, v1 = BW[name]
    u = float(ease((v - v0) / (v1 - v0)))
    P = np.array(c['pos']); L = np.array(c['look'])
    cc = C(L + (P - L) * (1 - 0.06 * u), L, c['fov'])
    return hand(cc, v, 0.012, 3)


# ---------------------------------------------------------------- the opening teleport: a little speed-line push either side of it
def open_post(e, v):
    for a, b, k in ((4.31, 4.46, 0.75), (4.53, 4.69, 0.6)):
        if a <= v < b:
            w = np.sin(np.pi * (v - a) / (b - a))
            e['lines'] = max(e.get('lines', 0), k * w); e['lineSeed'] = float(int(v * 30))
            e['lineC'] = [0.5, 0.56] if a < 4.4 else OPEN_LC
    d = v - T_ARR0
    if 0 <= d < 0.3: p = np.exp(-d / 0.07); e['zoom'] = e.get('zoom', 1.0) + 0.03 * p; e['ca'] = e.get('ca', 0.0015) + 0.005 * p
    return e


def _open_lc():
    global MODE
    mo = MODE; MODE = 'new'
    try:
        c = two(T_CAS, dist=3.2, h=0.5, lookh=1.25, sign=-1, fov=42, orbit=0.1); n = proj(c, pt('H', 'chestC', T_CAS))[0][0]
        return [float(0.5 + n[0] / 2), float(0.5 + n[1] / 2)]
    finally: MODE = mo


OPEN_LC = _open_lc()


NFRAMES = int(round(DUR5 * FPS))

if __name__ == '__main__':
    import sys
    for t in [float(x) for x in sys.argv[1:]] or [0.5, 3.8, 4.5, 4.8, 5.0, 5.5, 6.0, 6.2, 7.0, 11.0, 14.7, 23.0, 30.0, 33.5, 34.2]:
        f = frame(int(t * FPS)); print('v=%.2f t=%.3f cam %s fx %s post %s' % (t, f['t'], np.round(f['cam']['pos'], 2), {k: len(v) for k, v in f['fx'].items() if isinstance(v, list)}, {k: round(v, 2) for k, v in f['post'].items() if k in ('flash', 'impact', 'lines', 'duo', 'black')}))
