"""Bleedout spot v2, the body: a slow first-person walk out of the rock cut to the cherry tree, a hand on the bark,
a turn, a slide down the trunk and a plop on the beat, then slumped breathing, the look up on "Windmill" and a
look around.  Everything is in video seconds (song time - SONG0).  build() -> list of per-frame dicts."""
import math, os, pickle, sys
import numpy as np
from scipy.interpolate import PchipInterpolator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rig'))
import rig, poses4 as P4
HERE = os.path.dirname(os.path.abspath(__file__))
ST = P4.STATIC
PX = ST.scale / 16                       # metres per model px
FPS = 30
SONG0 = 59.85
BEAT = 0.4331


def beat(k):
    return (50.78 - SONG0) + BEAT * k


T_PLOP = beat(28)                        # 3.06: the last big hit before the bridge
T_WIND = 65.28 - SONG0                   # 5.43: "Windmill"
DUR = 75.95 - SONG0                      # 16.1: out on "...take it all in on your stride"
GROUND = 251.0
LEG = 12 * PX


def ss(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def ease(a, b, t0, t1, t):
    return a + (b - a) * ss((t - t0) / (t1 - t0))


# ---------------------------------------------------------------- the path (pelvis, top view x/z)
WP = np.array([(12.15, -0.02), (11.55, 0.24), (11.0, 0.50), (10.70, 0.74), (10.52, 1.02)])


def _catmull(P, n=3000):
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, n // (len(P) - 3), endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    out.append(P[-2])
    return np.array(out)


PATH = _catmull(WP)
SL = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(PATH, axis=0), axis=1))])
PLEN = SL[-1]


def path_at(s):
    s = min(max(s, 0.0), PLEN - 1e-6)
    i = int(np.searchsorted(SL, s) - 1); i = min(max(i, 0), len(PATH) - 2)
    u = (s - SL[i]) / max(SL[i + 1] - SL[i], 1e-9)
    p = PATH[i] + (PATH[i + 1] - PATH[i]) * u
    j0, j1 = max(i - 6, 0), min(i + 7, len(PATH) - 1)
    h = PATH[j1] - PATH[j0]; h = h / np.linalg.norm(h)
    return p, h


def yaw_of(h):                           # forward (x, z) = (-sin y, -cos y)
    return math.degrees(math.atan2(-h[0], -h[1]))


def left_of(yaw):
    y = math.radians(yaw)
    return np.array([-math.cos(y), math.sin(y)])


def fwd_of(yaw):
    y = math.radians(yaw)
    return np.array([-math.sin(y), -math.cos(y)])


# ---------------------------------------------------------------- footfalls
# (time, side, pelvis arc length at contact, kind)
STEPS = [(beat(21), 'R', 0.20, 'walk'), (beat(22), 'L', 0.62, 'walk'), (beat(23), 'R', 1.04, 'walk'), (beat(24), 'L', 1.42, 'end')]
_K = (PLEN - 0.22) / 1.42                # fit the step lengths to the path
STEPS = [(a, b, c * _K, d) for a, b, c, d in STEPS]
T_WALK_END = beat(24) + 0.42             # pelvis reaches the end of the path
T_TURN0, T_TURN1 = beat(24) + 0.30, beat(25.75) + 0.08     # turning his back to the trunk
STAND = np.array([10.80, 1.45])          # pelvis with his back on the bark, facing west
FOOT_R_ST, FOOT_L_ST = np.array([10.52, 1.32]), np.array([10.55, 1.60])
T_SLIDE0, T_GIVE = beat(25.75) + 0.12, T_PLOP - 0.34
SEAT = np.array([10.66, 1.45]); SEAT_YAW = 93.0
FOOT_R_SEAT, FOOT_L_SEAT = np.array([9.975, 1.36]), np.array([10.06, 1.68])


def s_of_t():
    ts = [0.0] + [a[0] for a in STEPS[:-1]] + [STEPS[-1][0], T_WALK_END]
    ss_ = [STEPS[0][2] - 0.42 * _K * STEPS[0][0] / 0.87] + [a[2] for a in STEPS] + [PLEN]
    base = PchipInterpolator(ts, ss_)
    cts = [a[0] for a in STEPS]

    def f(t):
        t = min(t, T_WALK_END)
        s = float(base(t))
        i = np.searchsorted(cts, t) - 1
        if 0 <= i < len(cts) - 1:                    # slowest just after each contact: he loads a weak leg
            u = (t - cts[i]) / (cts[i + 1] - cts[i])
            ds = (ss_[i + 2] - ss_[i + 1])
            s -= 0.22 * ds * math.sin(2 * math.pi * u) / (2 * math.pi)
        return s
    return f


S_T = s_of_t()


def pelvis_xz_yaw(t):
    if t <= T_TURN0:
        p, h = path_at(S_T(t))
        return p, yaw_of(h)
    p_end, h_end = path_at(S_T(T_TURN0))
    y0 = yaw_of(h_end)
    if t < T_SLIDE0 + 0.1:
        u = ss((t - T_TURN0) / (T_TURN1 + 0.1 - T_TURN0))
        p = p_end + (STAND - p_end) * ss((t - (T_TURN0 + 0.15)) / (T_TURN1 + 0.25 - T_TURN0 - 0.15))
        return p, y0 + (90.0 - y0) * u
    # sliding down the trunk: the pelvis moves a little forward as it drops, then the plop
    u = ss((t - T_SLIDE0) / (T_PLOP - T_SLIDE0))
    p = STAND + (SEAT - STAND) * u
    return p, 90.0 + (SEAT_YAW - 90.0) * ss((t - T_PLOP) / 0.8)


def foot_plants():
    """per side: list of (t_land, xz) and the lift times; walk plants from the pelvis path, then the stance and seat"""
    plants = {'R': [], 'L': []}
    prev_s = None
    for i, (t, side, s, kind) in enumerate(STEPS):
        if kind == 'end':
            p, h = path_at(min(s + 0.18, PLEN - 0.02)); y = yaw_of(h)
            xz = p + left_of(y) * (0.11 if side == 'L' else -0.11)
        else:
            lead = 0.0 if kind == 'close' else 0.5 * (s - (prev_s if prev_s is not None else s - 0.42))
            if kind == 'stumble':
                lead *= 0.7
            p, h = path_at(s + lead); y = yaw_of(h)
            lat = 0.10 + (0.05 if kind == 'stumble' else 0.0)
            xz = p + left_of(y) * (lat if side == 'L' else -lat)
            prev_s = s
        plants[side].append([t, xz, kind])
    plants['R'].append([beat(25), FOOT_R_ST.copy(), 'turn'])
    plants['L'].append([beat(25.75), FOOT_L_ST.copy(), 'turn'])
    return plants


PLANTS = foot_plants()
CONTACTS = sorted([(p[0], s) for s in 'RL' for p in PLANTS[s]])
SWINGMAX = 0.68
DS = 0.17


def foot_xz_lift(side, t):
    pl = PLANTS[side]
    if t < pl[0][0]:                                   # before the first plant: swinging in from behind
        nxt = pl[0]
        prv_xz = nxt[1] - fwd_of(yaw_of(path_at(0.3)[1])) * 0.42
        t_lift = nxt[0] - SWINGMAX
        if t < t_lift:
            return prv_xz, 0.0
        u = (t - t_lift) / (nxt[0] - t_lift)
        return prv_xz + (nxt[1] - prv_xz) * ss(u), u
    for k in range(len(pl)):
        t_land = pl[k][0]
        if k + 1 == len(pl):
            return pl[k][1], 0.0
        nxt = pl[k + 1]
        other = [c for c in CONTACTS if c[1] != side and t_land < c[0] < nxt[0]]
        t_lift = max((other[0][0] + DS) if other else t_land + DS, nxt[0] - SWINGMAX)
        if t < t_lift:
            return pl[k][1], 0.0
        if t < nxt[0]:
            u = (t - t_lift) / (nxt[0] - t_lift)
            return pl[k][1] + (nxt[1] - pl[k][1]) * (0.15 * u + 0.85 * ss(u)), u
    return pl[-1][1], 0.0


def feet(t):
    """world targets for the soles"""
    out = {}
    for side in 'RL':
        if t < T_GIVE:
            xz, u = foot_xz_lift(side, t)
            lift = 0.042 * math.sin(math.pi * u) ** 1.3 if 0 < u < 1 else 0.0
            out[side] = np.array([xz[0], GROUND + lift + 0.002, xz[1]])
        else:                                          # the legs give: feet skid forward, then the legs slide out
            st = FOOT_R_ST if side == 'R' else FOOT_L_ST
            seat = FOOT_R_SEAT if side == 'R' else FOOT_L_SEAT
            mid = st + (seat - st) * 0.45
            if t < T_PLOP:
                xz = st + (mid - st) * ss((t - T_GIVE) / (T_PLOP - T_GIVE)) ** 1.5
                y = GROUND + 0.002
            else:
                k = (t - T_PLOP) / (0.55 if side == 'R' else 0.75)
                xz = mid + (seat - mid) * (1 - (1 - min(k, 1)) ** 2.2)
                y = GROUND + 0.002 + 0.10 * ss(k)
            out[side] = np.array([xz[0], y, xz[1]])
    return out


# ---------------------------------------------------------------- body heights, lean, sway
def walk_phase(t):
    cts = [c[0] for c in CONTACTS]
    i = np.searchsorted(cts, t) - 1
    if i < 0:                                          # before the first footfall: as if one landed a step earlier
        return (t - (cts[0] - 0.866)) / 0.866, 'L' if CONTACTS[0][1] == 'R' else 'R', 0.866
    if i >= len(cts) - 1:
        return 0.0, CONTACTS[-1][1], 0.87
    u = (t - cts[i]) / (cts[i + 1] - cts[i])
    return u, CONTACTS[i][1], cts[i + 1] - cts[i]


def pelvis_height(t, ft, xz):
    if t < T_SLIDE0:
        u, stance, _ = walk_phase(t)
        h = 0.655 + 0.013 * math.sin(math.pi * u) - 0.012 * math.exp(-((u * 0.87) / 0.12) ** 2)
        if beat(14.5) - 0.25 < t < beat(14.5) + 0.35:                       # the stumble: the knee buckles
            h -= 0.06 * math.exp(-((t - (beat(14.5) + 0.02)) / 0.16) ** 2)
        if t > T_TURN1 - 0.3:
            h = ease(h, 0.640, T_TURN1 - 0.3, T_SLIDE0, t)
        # never ask a leg for more than it has
        for side in 'RL':
            d = np.hypot(*(ft[side][[0, 2]] - xz))
            h = min(h, math.sqrt(max(LEG ** 2 - d ** 2, 0.05)) + (ft[side][1] - GROUND) - 0.004)
        return h
    if t < T_GIVE:
        return ease(0.640, 0.40, T_SLIDE0, T_GIVE, t)
    if t < T_PLOP:
        u = (t - T_GIVE) / (T_PLOP - T_GIVE)
        return 0.40 - (0.40 - 0.106) * u ** 1.8
    k = t - T_PLOP
    return 0.106 + 0.014 * math.sin(math.pi * min(k / 0.16, 1)) * (k < 0.16)


_BK = np.arange(0, 40, 0.01)
_BPER = 1.6 + 1.6 * np.array([ss(k / 6.0) for k in _BK])
_BPH = np.concatenate([[0], np.cumsum(2 * np.pi / _BPER[:-1] * 0.01)])


def breath(t):
    """chest rise: heavy and quick after the plop, slower by the time he looks up"""
    if t < T_PLOP:
        return 0.5 * math.sin(2 * math.pi * t / 1.75)
    ph = float(np.interp(t - T_PLOP, _BK, _BPH))
    return math.sin(ph) * (1.0 - 0.45 * ss((t - T_WIND) / 4.0))


def torso(t):
    """hips rx, waist x: hunched forward walking; straighter against the bark; leaning back on it seated"""
    if t < T_TURN0:
        hx, wx = 0.08, 0.25
        if beat(8) < t < beat(11):                                    # he straightens a little to look for the tree
            k = ss((t - beat(8)) / 0.5) * (1 - ss((t - (beat(11) - 0.45)) / 0.45))
            wx -= 0.10 * k
        if beat(14.5) - 0.2 < t < beat(14.5) + 0.7:                   # stumble lurch
            wx += 0.16 * math.exp(-((t - (beat(14.5) + 0.05)) / 0.2) ** 2)
        u, _, _ = walk_phase(t)
        wx += 0.025 * math.sin(2 * math.pi * u) + 0.02 * breath(t)
        return hx, wx
    if t < T_PLOP:
        hx = ease(0.08, -0.06, T_TURN0, T_SLIDE0, t)
        wx = ease(0.25, 0.02, T_TURN0, T_SLIDE0, t)
        if t > T_SLIDE0:
            hx = ease(-0.06, -0.12, T_SLIDE0, T_PLOP, t)
            wx = ease(0.02, -0.10, T_SLIDE0, T_PLOP, t)
        return hx, wx
    k = t - T_PLOP
    jolt = 0.24 * math.exp(-((k - 0.10) / 0.09) ** 2) - 0.05 * math.exp(-((k - 0.36) / 0.12) ** 2)   # the impact throws the chest forward
    wx = -0.10 + (-0.17 + 0.10) * ss(k / 0.9) + jolt + 0.03 * breath(t)
    return -0.12, wx


# ---------------------------------------------------------------- where he looks (camera, world space)
# (t, yaw offset from the body (+ = left), pitch, roll)
# he keeps his eyes down (nothing of the view) until "Windmill"
LOOK = [(0.0, 10, -60, 1), (0.30, 26, -59, 2), (0.55, 36, -58, 2), (0.85, 26, -60, 1), (1.30, 8, -62, 0), (1.85, 8, -64, 0), (2.25, -6, -66, -2),
        (2.70, 0, -65, 0), (T_PLOP, 0, -63, 0), (T_PLOP + 0.12, 0, -72, 2), (T_PLOP + 0.5, -2, -63, 1), (4.4, 3, -62, 0),
        (5.30, 2, -62, 0), (7.50, -14, -4, 0), (8.80, -16, -2, 0), (10.60, 47, 3, 0), (11.30, 50, 2, 0), (13.40, -55, -6, 0),
        (14.00, -57, -5, 0), (15.20, -15, 2, 1), (DUR, -13, -3, 3)]
_LK = [PchipInterpolator([k[0] for k in LOOK], [k[j] for k in LOOK]) for j in (1, 2, 3)]


def look(t, body_yaw):
    yo, pitch, roll = (float(f(t)) for f in _LK)
    if t < T_SLIDE0:                                                  # the head bobs with each step
        u, stance, per = walk_phase(t)
        pitch += 1.1 * math.sin(2 * math.pi * u + 0.6)
        roll += 1.4 * (1 if stance == 'R' else -1) * math.sin(math.pi * u)
    pitch += 0.9 * breath(t) * (1.0 if t > T_PLOP else 0.4)
    yaw = body_yaw + yo
    if T_TURN0 < t < T_TURN1 + 0.4:                                  # the eyes lead the body round
        pass
    fov = 86.0 + 9.0 * ss((t - (T_WIND - 0.1)) / 2.2) - 2.0 * ss((t - 12.0) / 3.0)
    return yaw, pitch, roll, fov


def cam_basis(yaw, pitch, roll):
    y, p, r = math.radians(yaw), math.radians(pitch), math.radians(roll)
    f = np.array([-math.sin(y) * math.cos(p), math.sin(p), -math.cos(y) * math.cos(p)])
    right = np.cross(f, [0, 1, 0]); right /= np.linalg.norm(right)
    up0 = np.cross(right, f)
    up = up0 * math.cos(r) - right * math.sin(r)
    return f, up


# ---------------------------------------------------------------- arms
LH_CHEST = (1.7, 8.2, -2.7)                  # left fingertips on the shirt, below the middle arrow (top px)
LH_ELBOW = (6.4, 9.6, -1.4)
BARK_A, BARK_C, BARK_B = np.array([11.16, 251.78, 0.975]), np.array([10.985, 251.80, 1.05]), np.array([10.975, 251.82, 1.30])
T_REACH0, T_REACH1, T_BARK1, T_BACK = 0.20, 0.55, 1.85, 2.45
RH_GROUND = np.array([10.60, GROUND + 0.035, 1.02])
LH_LAP = np.array([10.28, GROUND + 0.30, 1.60])


def bark_point(t):
    """the hand steadying him on the trunk: planted, a hand-step to the corner, a slide round onto the west face"""
    if t < 1.05:
        return BARK_A + np.array([-0.02, -0.01, 0]) * ss((t - T_REACH1) / 0.2)
    if t < 1.25:
        w = ss((t - 1.05) / 0.20)
        a = BARK_A + np.array([-0.02, -0.01, 0])
        return a + (BARK_C - a) * w + np.array([0, 0.04, -0.03]) * math.sin(math.pi * w)
    if t < 1.50:
        return BARK_C + (BARK_B - BARK_C) * ss((t - 1.25) / 0.25)
    return BARK_B + np.array([0, -0.02, 0.02]) * ss((t - 1.50) / 0.4)


def solve_leg_ref(p, nm, target, ref_x, ref_knee, w=0.5):
    """leg x / yaw / roll + knee onto the target, staying near a natural reference; the yaw lets the feet splay"""
    from scipy.optimize import least_squares
    def res(v):
        q = dict(p); q[nm + '_leg'] = [v[0], v[3], v[1]]; q[nm + '_knee'] = v[2]
        f = rig.pt(ST.frames(q)[nm + '_foot'])
        return np.concatenate([(f - target) * 20, [2.5 * v[1], 0.4 * v[3], w * (v[0] - ref_x), w * (v[2] - ref_knee)]])
    x0 = [ref_x, 0.0, max(0.05, ref_knee), p[nm + '_leg'][1] if abs(p[nm + '_leg'][1]) < 0.9 else 0.0]
    lo, hi = [-2.6, -1.2, 0.0, -0.9], [1.6, 1.2, 2.6, 0.9]
    x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
    r = least_squares(res, x0, bounds=(lo, hi), xtol=1e-8, max_nfev=300)
    p[nm + '_leg'] = [float(r.x[0]), float(r.x[3]), float(r.x[1])]; p[nm + '_knee'] = float(r.x[2])


def build(cache=os.path.join(HERE, 'motion3.pkl')):
    n = int(round(DUR * FPS))
    out = []
    prev = None
    worst = []
    raw = []
    for k in range(n):
        t = k / FPS
        xz, yaw = pelvis_xz_yaw(t)
        ft = feet(t)
        u, stance, per = walk_phase(t)
        sway = 0.0
        if t < T_SLIDE0:
            sway = 0.028 * (1 if stance == 'L' else -1) * math.sin(math.pi * u)
            if beat(14.5) - 0.2 < t < beat(14.5) + 0.6:
                sway += 0.05 * math.exp(-((t - (beat(14.5) + 0.12)) / 0.2) ** 2)
        root_xz = xz + left_of(yaw) * sway
        raw.append([root_xz[0], root_xz[1], yaw, pelvis_height(t, ft, root_xz), *torso(t)])
    raw = np.array(raw)
    sm = raw.copy()
    for k in range(n):
        t = k / FPS
        sig = 2.6 * ss((abs(t - (T_PLOP + 0.14)) - 0.45) / 0.5)          # none round the impact, ramping in
        if sig < 0.3:
            continue
        j = np.arange(max(0, k - int(3 * sig) - 1), min(n, k + int(3 * sig) + 2))
        tj = j / FPS
        w = np.exp(-0.5 * ((j - k) / sig) ** 2)
        w = w / w.sum()
        sm[k] = (raw[j] * w[:, None]).sum(0)
        sm[k, 3] = raw[k, 3] + 0.6 * (sm[k, 3] - raw[k, 3])          # height: lighter, the feet need it
    for k in range(n):
        t = k / FPS
        ft = feet(t)
        u, stance, per = walk_phase(t)
        root_xz = sm[k, :2]; yaw = float(sm[k, 2]); h = float(sm[k, 3]); hx, wx = float(sm[k, 4]), float(sm[k, 5])
        pr = 0.07 * (1 if stance == 'R' else -1) * math.cos(math.pi * u) if t < T_SLIDE0 else 0.0
        p = dict(root=[float(root_xz[0]), GROUND, float(root_xz[1]), float(yaw), 0.0, 0.0], scale=ST.scale,
                 hips=[0.0, 12.016 - h / PX, 0.0, hx, pr, 0.0], waist=[wx, 0.0, 0.0], head=[0.0, 0.0, 0.0],
                 right_arm=[0.0, 0.0, 0.07], left_arm=list(prev['left_arm']) if prev else [-1.6, 0.5, -0.5],
                 right_elbow=-0.22, left_elbow=prev['left_elbow'] if prev else -1.8,
                 right_leg=list(prev['right_leg']) if prev else [-0.2, 0, 0], left_leg=list(prev['left_leg']) if prev else [0.2, 0, 0],
                 right_knee=prev['right_knee'] if prev else 0.3, left_knee=prev['left_knee'] if prev else 0.3, wrist=[0, 0, 0])
        # legs onto their soles
        for side, nm in (('R', 'right'), ('L', 'left')):
            if t < T_GIVE - 0.4:
                rig.solve_leg(ST, p, nm, ft[side], knee_hint=0.4)
            else:                            # crouched on the slide, then sat with the legs out (left knee up a little)
                k = ss((t - T_PLOP) / 0.8)
                rx_ = -1.15 + ((-1.42 if nm == 'right' else -1.62) + 1.15) * k
                rk = 1.6 + ((0.25 if nm == 'right' else 0.95) - 1.6) * k
                solve_leg_ref(p, nm, ft[side], rx_, rk)
            err = np.linalg.norm(rig.pt(ST.frames(p)[nm + '_foot']) - ft[side])
            if err > 0.02:
                worst.append((round(t, 2), nm, round(float(err), 3)))
        # right arm: hangs and swings, braces on the ground for the plop, rests there
        lean = hx + wx
        swing = 0.10 * (1 if stance == 'L' else -1) * math.sin(math.pi * u) if t < T_SLIDE0 else 0.0
        p['right_arm'] = [-lean + swing - 0.04, 0.0, 0.08]
        p['right_elbow'] = -0.22 - 0.06 * math.sin(math.pi * u)
        if t > T_SLIDE0 - 0.2:
            fk = dict(p); hand_fk = P4.hand(ST, fk, 'right')
            w = ss((t - (T_SLIDE0 - 0.2)) / (T_PLOP - T_SLIDE0 + 0.15))
            tgt = hand_fk + (RH_GROUND - hand_fk) * w
            P4.reach(ST, p, 'right', tgt, prior=p['right_arm'], w_prior=0.3)
        # left hand: pressed to the wound, out to the bark, back to the wound, then onto his lap
        F = ST.frames(p)
        chest = rig.pt(F['top'], LH_CHEST); chest_el = rig.pt(F['top'], LH_ELBOW)
        sh = rig.pt(F['left_arm'])
        if t < T_REACH0:
            tgt, hint = chest, chest_el
        elif t < T_REACH1:
            w = ss((t - T_REACH0) / (T_REACH1 - T_REACH0))
            b = BARK_A
            tgt = chest + (b - chest) * w + np.array([0, 0.06, 0]) * math.sin(math.pi * w)
            hint = chest_el + ((sh + b) / 2 - np.array([0, 0.12, 0]) - chest_el) * w
        elif t < T_BARK1:
            tgt = bark_point(t); hint = (sh + tgt) / 2 - np.array([0, 0.14, 0])
        elif t < T_BACK:
            w = ss((t - T_BARK1) / (T_BACK - T_BARK1))
            tgt = BARK_B + (chest - BARK_B) * w; hint = (sh + BARK_B) / 2 - np.array([0, 0.14, 0]) + (chest_el - (sh + BARK_B) / 2 + np.array([0, 0.14, 0])) * w
        elif t < T_PLOP + 0.05:
            tgt, hint = chest, chest_el
        else:
            w = ss((t - (T_PLOP + 0.05)) / 0.7)
            tgt = chest + (LH_LAP - chest) * w; hint = chest_el + (rig.pt(F['top'], (7.0, 11.0, 1.0)) - chest_el) * w
        e = P4.reach(ST, p, 'left', tgt, prior=p['left_arm'], w_prior=0.2, elbow_hint=hint, w_elbow=3.0)
        if e > 0.05 and T_REACH1 < t < T_BARK1:
            worst.append((round(t, 2), 'lhand', round(e, 3)))
        # where the eyes are
        F = ST.frames(p)
        cy, cp, cr, fov = look(t, yaw)
        f, up = cam_basis(cy, cp, cr)
        neck = rig.pt(F['top'])
        eye = neck + up * 0.195 + f * 0.13
        out.append(dict(t=t, pose=p, eye=eye, f=f, up=up, fov=fov))
        prev = p
    if worst:
        print('IK misses (first 12):', worst[:12], 'total', len(worst), 'times', sorted(set(round(w[0]) for w in worst)))
    pickle.dump(out, open(cache, 'wb'))
    return out


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('path length %.2f m' % PLEN)
    for s in 'RL':
        print(s, [(round(a[0], 2), np.round(a[1], 2).tolist(), a[2]) for a in PLANTS[s]])
    fr = build()
    print('built %d frames in %.0fs' % (len(fr), time.time() - t0))
