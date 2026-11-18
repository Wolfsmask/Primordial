"""The End Update: the timeline.  video time v -> renderView params (cameras on splines, the dragon, crystal beams, lights,
post) + the 2D overlay state (the phone, the lyrics).  frame(k) -> dict."""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = 60
SC = json.load(open(os.path.join(HERE, 'web', 'scene.json')))
T_END = 36.6
NFRAMES = int(round(T_END * FPS))
DEG = np.pi / 180
UP = np.array([0, 1.0, 0])
def dirYE(yaw, el): return np.array([-np.sin(yaw * DEG) * np.cos(el * DEG), np.sin(el * DEG), -np.cos(yaw * DEG) * np.cos(el * DEG)])
ECL = dirYE(SC['eclipse']['yaw'], SC['eclipse']['el'])
CRY = np.array(SC['crystals'])


def song_to_v(t): return t - 21.40 + 9.55 if t >= 21.40 else t - 12.0
BEAT0 = 21.455; BP = 0.6502
def beat(k): return song_to_v(BEAT0 + k * BP)            # beat k after the drop (video time)
T_PHONE_TAP = song_to_v(16.254); T_DIVE = song_to_v(16.904); T_TAPE = 8.90; T_SIL = 9.30; T_DROP = beat(0)
SHOTS = [T_DROP + 4 * BP * i for i in range(8)]            # one bar each from the drop
T_BREAK = song_to_v(40.26); T_GIRL = song_to_v(41.37); T_START = song_to_v(45.04); T_CUT = song_to_v(45.95)


def ease(x): x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)
def eout(x, p=2.2): x = np.clip(x, 0, 1); return 1 - (1 - x) ** p
def ein(x, p=2.0): x = np.clip(x, 0, 1); return x ** p
def lerp(a, b, u): return np.asarray(a, float) * (1 - u) + np.asarray(b, float) * u
def unit(v): v = np.asarray(v, float); return v / (np.linalg.norm(v) + 1e-9)


def spline(P, u):
    """Catmull-Rom through the points P at u in [0, 1]"""
    P = np.asarray(P, float); n = len(P) - 1; u = float(np.clip(u, 0, 1)) * n; i = min(int(u), n - 1); t = u - i
    p0, p1, p2, p3 = P[max(i - 1, 0)], P[i], P[i + 1], P[min(i + 2, n)]
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3)


def shake(v, amp, f=9.0):
    return np.array([np.sin(v * f * 2.1 + 1.3) + 0.5 * np.sin(v * f * 4.7), 0.6 * np.sin(v * f * 1.7 + 0.4) + 0.4 * np.sin(v * f * 5.3), 0.5 * np.sin(v * f * 3.1)]) * amp


# ---------------------------------------------------------------- the dragon's flight (control points, times)
def _ring(a, r=46, y=34): return (r * np.cos(np.radians(a)), y, r * np.sin(np.radians(a)))
DRAGON = [(SHOTS[5] - 1.0, (62, -58, -52)), (SHOTS[5] + 0.3, (58, -30, -48)), (SHOTS[5] + 1.3, (52, 0, -44)), (SHOTS[5] + 2.2, (46, 22, -40)),
          (SHOTS[6] + 0.0, _ring(-38, 50, 32)), (SHOTS[6] + 0.9, _ring(-58, 47, 34)), (SHOTS[6] + 1.8, _ring(-80, 45, 35)), (SHOTS[7] + 0.0, _ring(-104, 44, 34)),
          (SHOTS[7] + 0.45, _ring(-118, 40, 31)), (SHOTS[7] + 0.9, (-12, 28, 52)), (SHOTS[7] + 1.4, (2, 24, 62))]
DT_ = np.array([t for t, _ in DRAGON]); DP_ = np.array([p for _, p in DRAGON], float)


def dragon_pos(t):
    if t <= DT_[0]: return DP_[0] + (DP_[1] - DP_[0]) * (t - DT_[0]) / (DT_[1] - DT_[0])
    if t >= DT_[-1]: return DP_[-1] + (DP_[-1] - DP_[-2]) * (t - DT_[-1]) / (DT_[-1] - DT_[-2])
    i = int(np.searchsorted(DT_, t) - 1); u = (i + (t - DT_[i]) / (DT_[i + 1] - DT_[i])) / (len(DT_) - 1)
    return spline(DP_, u)


def fw_of(t): return unit(dragon_pos(t + 0.1) - dragon_pos(t - 0.1))
def sd_of(t): return unit(np.cross(UP, fw_of(t)) + np.array([1e-4, 0, 0]))


def mid_of(t):
    d = dragon(t); return np.array(d['spine'][7]) if d else dragon_pos(t)


def smooth_track(v, f, w=0.35, n=7):
    return np.mean([f(v + w * (i / (n - 1) - 0.5)) for i in range(n)], 0)


def dragon(v):
    if not (SHOTS[5] - 0.6 <= v < SHOTS[7] + 1.2): return None
    ts_ = v - np.arange(0, 400) * 0.012; P = np.array([dragon_pos(t) for t in ts_])
    L = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    spacing = np.array([0, 3.0] + [1.8] * 5 + [2.4] * 3 + [1.8] * 8)[:18]; want = np.cumsum(spacing)
    sp = [[float(np.interp(d, L, P[:, j])) for j in range(3)] for d in want]
    a = np.array(sp); vel = (dragon_pos(v + 0.02) - dragon_pos(v - 0.02)) / 0.04; acc = (dragon_pos(v + 0.05) - 2 * dragon_pos(v) + dragon_pos(v - 0.05)) / 0.0025
    lat = acc - np.dot(acc, unit(vel)) * unit(vel); up = unit(UP * 9.8 + lat * 0.6)
    flap = 0.55 * np.sin(v * 2 * np.pi * 1.5) + 0.1; fold = 0.0
    if vel[1] < -20: fold = 0.5; flap = 0.25 + 0.1 * np.sin(v * 9)
    roar = max(0.0, 1 - abs(v - (SHOTS[5] + 1.9)) / 0.6); jaw = 0.15 + 0.7 * roar + 0.6 * max(0.0, 1 - abs(v - (SHOTS[7] + 1.0)) / 0.35)
    return dict(spine=sp, up=up.tolist(), flap=float(flap), fold=fold, jaw=float(jaw), breath=float(roar * 1.2), t=float(v))


# ---------------------------------------------------------------- shots
def cam(v):
    """-> dict(pos, look, fov, roll, extra...)"""
    if v < T_DIVE: return None                                      # the phone (2D)
    if v < T_DROP:                                                  # inside the portal: falling through the stars
        return dict(pos=[0, 400, 0], look=[0, 300, 0.01], fov=64, roll=float(10 * np.sin(v * 0.7)))
    s = SHOTS
    if v < s[1]:            # R1: up out of the void mist, the pillars against the eclipse
        u = eout((v - s[0]) / (s[1] - s[0]), 2.6)
        p = spline([(-40, -52, -96), (-35, -14, -86), (-31, 16, -76), (-29, 25, -70)], u)
        l = lerp([0, -8, 0], [0, 15, 0], ease(u * 1.2)) + ECL * 30 * u
        return dict(pos=p, look=l, fov=58 - 4 * u, roll=-6 * (1 - u))
    if v < s[2]:            # R2: low and fast between the pillars, on to the portal
        u = (v - s[1]) / (s[2] - s[1])
        p = spline([(-36, 7, -48), (-24, 6.2, -32), (-12, 6.5, -16), (-2, 8, -4)], u)
        l = spline([(-24, 6.5, -32), (-10, 7, -14), (2, 8, 4), (14, 12, 22)], u)
        return dict(pos=p, look=l, fov=64, roll=float(-8 * np.sin(u * 3.0)))
    if v < s[3]:            # R3: round a crystal as they all fire into the sky
        c = CRY[4]; u = (v - s[2]) / (s[3] - s[2]); a = 0.5 + 1.6 * ease(u)
        p = c + np.array([np.cos(a) * 9.5, 1.5 - 4.0 * u, np.sin(a) * 9.5])
        return dict(pos=p, look=c + [0, 3.0 * u, 0], fov=56)
    if v < s[4]:            # R4: over the ender grove, the void pouring off its rim
        u = (v - s[3]) / (s[4] - s[3])
        p = spline([(66, 12, -26), (82, 6, -36), (96, 3, -36), (110, -4, -42)], u)
        l = spline([(98, -2, -58), (98, -4, -58), (100, -10, -60), (112, -40, -66)], u)
        return dict(pos=p, look=l, fov=58)
    if v < s[5]:            # R5: up the end city's tower
        u = ease((v - s[4]) / (s[5] - s[4]))
        p = lerp((-78, 6, -18), (-84, 58, -24), u); l = lerp((-116, 20, -50), (-116, 62, -50), u)
        return dict(pos=p, look=l, fov=58)
    if v < s[6]:            # D1: it comes up out of the void, past us
        d = mid_of(v)
        p = smooth_track(v, lambda t: mid_of(t) + sd_of(t) * 40 + np.array([0, -8.0, 0])) + shake(v, 0.15 * max(0, 1 - abs(v - s[5] - 1.9)), 6)
        return dict(pos=p, look=d + [0, 5.0, 0], fov=58)
    if v < s[7]:            # D2: alongside it, round the pillars, beams pouring into it
        d = mid_of(v)
        p = smooth_track(v, lambda t: mid_of(t) + sd_of(t) * 36 + np.array([0, 9.0, 0]) + fw_of(t) * 4)
        return dict(pos=p, look=d + fw_of(v) * 6, fov=52)
    if v < T_BREAK:         # D3: it turns and comes straight at us
        d = dragon_pos(v); p = np.array([34, 24, 76.0])
        return dict(pos=p, look=d, fov=50 - 10 * ease((v - s[7]) / (T_BREAK - s[7])))
    if v < T_GIRL + 0.23:   # the title, over the eclipse
        u = (v - T_BREAK) / (T_GIRL + 0.23 - T_BREAK); p = np.array([-6, 30, -20.0]) + ECL * 6 * u
        return dict(pos=p, look=p + ECL * 100 + np.array([0, -38.0, 0]), fov=50)
    if v < T_CUT:           # down onto the exit portal - and into it on "start"
        pc = np.array(SC['portal']['c']); u = (v - T_GIRL - 0.23) / (T_START - T_GIRL - 0.23)
        if v < T_START:
            p = spline([(28, 64, 40), (16, 40, 24), (8, 22, 9), (4.0, 12, 3.6)], ease(u)); l = pc + [0.8, -1, 0.6]
        else:
            k = ein((v - T_START) / (T_CUT - T_START), 1.6); q = pc + np.array([2.2, 0, 1.2]); p = lerp((4.0, 12, 3.6), q + [0, -3.5, 0], k); l = q + [0, -10, 0]
        return dict(pos=p, look=l, fov=60, roll=float(12 * ein(u, 3)))
    return None                                                     # black (and the notification)


def lights_for(c, v):
    look = np.array(c['look'], float); pos = np.array(c['pos'], float); pts = []
    for k in CRY: pts.append((np.linalg.norm(k - look) + 0.5 * np.linalg.norm(k - pos), [*k, 1.0, 1.0, 0.35, 1.0]))
    for g in SC['glows']: pts.append((np.linalg.norm(np.array(g[:3]) - look) + 0.5 * np.linalg.norm(np.array(g[:3]) - pos), [*g[:3], 0.8, 0.9, 0.35, 1.1]))
    for l in SC['lamps']: pts.append((np.linalg.norm(np.array(l) - look) * 1.3, [*l, 0.5, 1.0, 0.9, 1.1]))
    pts.sort(key=lambda x: x[0]); return [p for _, p in pts[:4]]


def beams_for(v):
    out = []
    s = SHOTS
    if s[2] - 0.2 <= v < s[4]:                        # the crystals fire into the sky, all to one point above the portal
        k = ease((v - s[2] - 0.3) / 0.25) * (1 - ease((v - s[4] + 0.6) / 0.6))
        if k > 0:
            top = np.array([0, 140, 0.0])
            for i, c in enumerate(CRY):
                fl = 0.75 + 0.25 * np.sin(v * 23 + i)
                out.append([*c, *top, 0.35 + 0.15 * (i % 2), k * fl * (1.6 if (int((v - s[2]) / BP) % 2 == 0) else 1.1)])
    d = dragon(v)
    if d and s[6] <= v < s[7] + 0.4:                  # and into the dragon as it passes
        dp = np.array(d['spine'][7]); near = sorted(CRY.tolist(), key=lambda c: np.linalg.norm(np.array(c) - dp))[:3]
        for c in near: out.append([*c, *dp, 0.3, 0.9 + 0.3 * np.sin(v * 31), 1.0, 0.45, 1.0])
    return out


def post_at(v):
    e = dict(exposure=1.05, vig=0.38, ca=0.0015, grain=0.03, zoom=1.0, shake=[0, 0], flash=0.0, sat=1.0, contrast=1.0, tint=[1, 1, 1], bloom=0.09, bars=0.0)
    if v < T_DROP:
        if v < T_DIVE + 0.25: e['flash'] = max(0.0, 1 - (v - T_DIVE) / 0.25)
        if v >= T_TAPE:                                   # the tape stop: it slows, it greys, it goes dark
            k = (v - T_TAPE) / (T_SIL - T_TAPE); e['sat'] = 1 - 0.8 * k; e['exposure'] = 1.05 * (1 - k); e['zoom'] = 1 + 0.06 * k
        if v >= T_SIL: e['black'] = 1.0
        return e
    d = v - T_DROP
    if 0 <= d < 0.5: e['flash'] = max(e['flash'], np.exp(-d / 0.08)); e['zoom'] += 0.08 * np.exp(-d / 0.15); e['ca'] += 0.01 * np.exp(-d / 0.15)
    for k in range(0, 46):
        t = beat(k); dd = v - t
        if 0 <= dd < 0.3 and t < T_BREAK: p = np.exp(-dd / 0.07); e['zoom'] += 0.012 * p; e['ca'] += 0.003 * p
    for t in SHOTS[1:] + [T_BREAK, T_GIRL + 0.23]:   # a hard cut gets a little flash and push
        dd = v - t
        if 0 <= dd < 0.18: e['flash'] = max(e['flash'], 0.25 * (1 - dd / 0.18)); e['zoom'] += 0.03 * (1 - dd / 0.18)
    if SHOTS[5] <= v < T_BREAK:                       # the dragon: wider, darker, heavier
        e['bars'] = 0.06; e['contrast'] = 1.25; e['vig'] = 0.5
        rr = max(0.0, 1 - abs(v - (SHOTS[5] + 1.9)) / 0.5)
        sh = shake(v, 0.006 * rr, 23); e['shake'] = [sh[0], sh[1]]
        if v > SHOTS[7] + 0.4: e['lines'] = 0.6 * ease((v - SHOTS[7] - 0.4) / 0.2); e['lineSeed'] = float(int(v * 30))
    if T_BREAK <= v < T_GIRL + 0.23:                  # under water: soft, slow, the title
        e['bloom'] = 0.12; e['sat'] = 0.9; e['vig'] = 0.6; e['exposure'] = 0.85; dd = v - T_BREAK
        if dd < 0.2: e['flash'] = max(e['flash'], 0.8 * (1 - dd / 0.2))
    if v >= T_START:
        k = ein((v - T_START) / (T_CUT - T_START), 2); e['zoom'] += 0.15 * k; e['lines'] = 0.7 * k; e['lineSeed'] = float(int(v * 30)); e['ca'] += 0.01 * k
        if v > T_CUT - 0.12: e['flash'] = (v - (T_CUT - 0.12)) / 0.12
    if v >= T_CUT: e['black'] = 1.0
    return e


def texts_at(v):
    if not (T_BREAK <= v < T_GIRL + 0.23): return []
    u = ease((v - T_BREAK - 0.05) / 0.35); o = 1 - ease((v - (T_GIRL + 0.05)) / 0.18); push = 1.0 + 0.05 * (v - T_BREAK)
    return [dict(id='THE', pos=[0, 1.25, -10], rot=[0, 0, 0], scale=push, opacity=u * o, glow=0.4),
            dict(id='END', pos=[0, 0.0, -10], rot=[0, 0, 0], scale=(1.25 - 0.25 * u) * push, opacity=u * o, glow=0.6 + 0.6 * (1 - u)),
            dict(id='UPDATE', pos=[0, -1.32, -10], rot=[0, 0, 0], scale=push, opacity=ease((v - T_BREAK - 0.25) / 0.35) * o, glow=0.4)]


# ---------------------------------------------------------------- lyrics (word timings from the vocal, in song time)
LYRICS = [
    [("It's", 12.01), ('your', 12.83), ('undress', 13.15), ('rehearsal', 14.10)],
    [("You're", 16.02), ('a', 17.11), ('main', 17.16), ('attraction,', 17.48)],
    [('dance', 18.93), ('in', 19.43), ('my', 19.77), ('back', 20.43), ('seat', 20.70)],
    [('Ooh', 21.52), ('la', 21.81), ('la', 22.84), ('la', 23.15), ('la,', 23.79), ('da', 24.18), ('da,', 24.45), ('la', 25.13), ('la', 25.45), ('la', 25.71), ('la', 25.98)],
    [('Ooh', 26.61), ('la', 27.14), ('la', 28.04), ('la', 28.25), ('la,', 28.49), ('da', 29.01), ('da,', 29.36), ('la', 29.80), ('la', 30.05), ('la', 30.25), ('la', 30.47)],
    [('Ooh', 32.42), ('la', 32.84), ('la', 33.29), ('la', 33.61), ('la,', 33.77), ('da', 34.50), ('da,', 34.90), ('la', 35.48), ('la', 35.85), ('la', 36.01), ('la', 36.25)],
    [('Ooh,', 36.56), ('la,', 38.12), ('la,', 38.39), ('la,', 38.73), ('la,', 39.78), ('la', 40.05)],
    [('Girl,', 41.37), ('tell', 41.82), ('me', 41.95), ('now,', 42.66)],
    [('you', 43.59), ('see', 43.74), ("it's", 44.14), ('about', 44.62), ('to', 44.64), ('start', 45.04)]]
LYR = []
for i, line in enumerate(LYRICS):
    t0 = song_to_v(line[0][1]) - 0.08
    t1 = song_to_v(LYRICS[i + 1][0][1]) - 0.12 if i + 1 < len(LYRICS) else T_CUT
    t1 = min(t1, song_to_v(line[-1][1]) + 1.6)
    if line[0][1] < 21.40 and t1 > T_TAPE: t1 = T_TAPE
    LYR.append(dict(t0=t0, t1=t1, words=[(w, song_to_v(t)) for w, t in line]))


def lyric_at(v):
    for L in LYR:
        if L['t0'] <= v < L['t1']:
            return dict(words=L['words'], v=v, t0=L['t0'], t1=L['t1'])
    return None


def frame(k):
    v = k / FPS
    out = dict(v=v, lyric=lyric_at(v), phone=None, notif=None, render=None)
    if v < T_DIVE:
        out['phone'] = dict(v=v, tap=T_PHONE_TAP, dive=T_DIVE, n2=song_to_v(14.977)); return out
    if v >= T_CUT:
        out['notif'] = dict(v=v, t0=T_CUT + 0.55, t1=T_END - 0.75); return out
    c = cam(v)
    r = dict(pos=[float(x) for x in c['pos']], look=[float(x) for x in c['look']], fov=float(c['fov']), roll=float(c.get('roll', 0.0)), time=float(v))
    if v < T_DROP:
        tt = min(v, T_TAPE) + max(0.0, 0.5 * (v - T_TAPE) - 0.6 * (v - T_TAPE) ** 2) if v > T_TAPE else v
        r.update(dive=1.0, diveFwd=[0, -1.0, 0.0], diveSpeed=1.0 + 2.5 * ein((v - T_DIVE) / (T_TAPE - T_DIVE), 2), time=float(tt), motes=0.0)
    if T_START <= v < T_CUT:
        r.update(dive=float(ein((v - T_START - 0.35) / 0.4, 1.5)), diveFwd=[0, -1.0, 0.0], diveSpeed=3.0)
    r['shadowC'] = r['look']
    r['lights'] = lights_for(c, v); r['beams'] = beams_for(v); r['dragon'] = dragon(v); r['texts'] = texts_at(v)
    r['post'] = post_at(v); r['god'] = 0.0 if v < T_DROP else (0.35 if T_BREAK <= v < T_GIRL + 0.23 else 0.6); r['fog'] = 0.0 if (v < T_DROP or v >= T_START + 0.5) else 1.0
    out['render'] = r
    return out


if __name__ == '__main__':
    print('drop %.3f shots %s break %.3f girl %.3f start %.3f cut %.3f end %.1f frames %d' % (T_DROP, np.round(SHOTS, 2), T_BREAK, T_GIRL, T_START, T_CUT, T_END, NFRAMES))
    for v in (1.0, 6.0, 9.0, 10.0, 13.0, 16.0, 19.0, 21.0, 23.5, 26.0, 28.2, 29.0, 31.0, 33.5, 35.0):
        f = frame(int(v * FPS)); r = f['render']
        print('%.1f' % v, 'phone' if f['phone'] else ('notif' if f['notif'] else 'cam %s look %s' % (np.round(r['pos'], 1), np.round(r['look'], 1))), 'lyric' if f['lyric'] else '')
