"""v7: Herobrine fights like a god (no guard, one-handed palms, swats, smites; floats; rises to power up).
Herobrine VS Notch v5 choreography: v4 + a casual opening straight that stops dead (4 beats inserted at 4.347 s).
v4 (story time, 0..33 s; music song4.wav, the video ends at 32.6 s).
Herobrine teleports (black smoke) and flash-steps (red streaks) all round Notch and throws only short, fast strikes: jabs,
straights, knees, snap kicks, hammer fists.  An invisible barrier stops every one a few centimetres short of Notch, who floats,
calm, and does not turn round.  Drop 2: the barrier shows, cracks under each strike and shatters; Notch catches the last punch
in one hand and presses him down onto a knee.  No spins, no flips: knockbacks are slides on the feet.
python3 story4.py -> story4.npz + story4_events.json + a clearance report"""
import json, os
import numpy as np
import keyanim as K, poses7 as PS, poses as P3, rig as R

HERE = os.path.dirname(os.path.abspath(__file__))
T_END = 33.0                  # v4 story end (keys inside build are written in v4 time, then shifted)
SPLIT = 4.347                 # everything from here on (v4 time) moves later by SH
SH = 4 * 0.4295               # the inserted bar
TB = T_END + SH + 0.1         # v5 bake length
EPS = 1.0 / 240


def V(m):
    return 1.77 + 0.4295 * m + (0.0185 if m >= 64 else 0.0)


def fwd(yaw): return np.array([np.sin(np.radians(yaw)), 0, np.cos(np.radians(yaw))])


def lft(yaw): return np.array([np.cos(np.radians(yaw)), 0, -np.sin(np.radians(yaw))])


def feet(c, yaw, lead='L', fw=0.18, sd=0.14):
    c = np.array([c[0], 0, c[1]]); f = fwd(yaw); l = lft(yaw)
    pl = c + l * sd + f * (fw if lead == 'L' else -fw); pr = c - l * sd + f * (fw if lead == 'R' else -fw)
    return pr[[0, 2]], pl[[0, 2]]


def rel(pose, yaw, part, pitch=None):
    C = {c: np.array([float(pose.get(c, 0.0))]) for c in K.CH}; C['yaw'][:] = yaw
    if pitch is not None: C['pitch'][:] = pitch
    for c in K.CH:
        b = K.base(c)
        if b in K.LIM: C[c] = np.clip(C[c], *K.LIM[b])
    W = K.world_quats(K.local_quats(C)); Q = np.stack([W[b] for b in R.BONES], 1)
    return R.fk(np.zeros((1, 3)), Q)[part][0]


def at(pose, yaw, part, target):
    return np.asarray(target, float) - rel(pose, yaw, part)


def eo(u): u = np.clip(u, 0, 1); return 1 - (1 - u) ** 2.2


# ---------------------------------------------------------------- the stage
NX, NY = 1.0, 0.95            # Notch hovering, drop 1 (pelvis); he faces -X (yaw -90)
YN1 = -90.0
NX2 = -1.0                    # he drifts after Herobrine at 14.6
NC2 = np.array([NX2 - 0.4, NY + 0.05, 0.0])     # breakdown + drop 2
YN2 = -16.0
HX0 = -3.2
WALLX = -7.6
STANCE_Y = 0.71
SHELL = dict(dy=0.32, rx=0.56, ry=1.25)          # the barrier once it shows (drop 2): ellipsoid round Notch's pelvis + dy
GAP = 0.08                                       # the invisible barrier: strikes stop this far off his body
PT = {'kneeR': 'shinR'}                          # contact points that are joints
CAPS = {'fistR': 'farmR', 'fistL': 'farmL', 'soleR': 'shinR', 'soleL': 'shinL', 'kneeR': 'thighR'}
YL = 180.0                                       # Herobrine after the backhand: facing -Z, at Notch
H_SPR = {'pitch': (9, 0.8), 'roll': (9, 0.8), 'sb': (8, 0.75), 'st': (8.5, 0.75), 'ss': (7, 0.75), 'nb': (6, 0.75), 'nt': (6, 0.75), 'ns': (6, 0.75),
         'af': (16, 0.85), 'aa': (16, 0.85), 'at': (14, 0.85), 'ae': (18, 0.82), 'lf': (10, 0.8), 'la': (10, 0.8), 'lt': (10, 0.82), 'lk': (11, 0.78)}

KIND = {
    'jab':      dict(load=PS.GUARD, hit=PS.JAB, rec=PS.GUARD, part='fistL', ground='both', tgt='chestC'),
    'straight': dict(load=PS.LOAD_R, hit=PS.STRAIGHT, rec=PS.GUARD, part='fistR', ground='both', tgt='chestC'),
    'air':      dict(load=PS.AIR_LOAD, hit=PS.AIR_STRAIGHT, rec=PS.AIR_LOAD, part='fistR', ground=None, tgt='headC'),
    'knee':     dict(load=PS.KNEE_LOAD, hit=PS.KNEE, rec=PS.KNEE_LOAD, part='kneeR', ground=None, tgt='chestC'),
    'snap':     dict(load=PS.SNAP_CHAMBER, hit=PS.SNAP, rec=PS.SNAP_CHAMBER, part='soleR', ground='L', tgt='belly', y=0.74),
    'hammer':   dict(load=PS.HAMMER_UP, hit=PS.HAMMER_DOWN, rec=PS.HAMMER_UP, part='fistR', ground=None, tgt='top'),
}
STANDOFF = {'chestC': 0.36, 'headC': 0.46, 'belly': 0.36, 'top': 0.40}


def n_pt(name, c, ny, yawN, pose):
    p = np.array([c[0], ny, c[1]]) + rel(pose, yawN, 'headC' if name == 'top' else name)
    return p + ([0, 0.25, 0] if name == 'top' else 0)


GY = 0.745        # v7: standing tall (pelvis height)
GFW = 0.08        # v7: feet nearly square, not a fighter's stagger
KIND.update({
    'palm':    dict(load=PS.PALM_LOAD, hit=PS.PALM_HIT, rec=PS.GOD, part='fistR', ground='both', tgt='chestC', y=GY, solve=(50, 150), bk=(0.05, 0.04), fb=0.04, fw=GFW),
    'palm_hi': dict(load=PS.PALM_LOAD, hit=PS.PALM_HIT_HI, rec=PS.PALM_LOAD, part='fistR', ground='both', tgt='headC', y=GY, solve=(70, 178), bk=(0.05, 0.04), fb=0.04, fw=GFW),
    'swat':    dict(load=PS.SWAT_LOAD, hit=PS.SWAT, rec=PS.SWAT_END, part='fistR', ground='both', tgt='chestC', y=GY, solve=(50, 150), bk=(0.04, 0.04), fb=0.04, fw=GFW),
    'fpalm':   dict(load=PS.FPALM_LOAD, hit=PS.FPALM_HIT, rec=PS.GOD_FLOAT, part='fistR', ground=None, tgt='headC', bk=(0.06, 0.06)),
    'smite':   dict(load=PS.SMITE_UP, hit=PS.SMITE, rec=PS.GOD_FLOAT, part='fistR', ground=None, tgt='top', bk=(0.04, 0.08)),
    # v7b: Thragg - full-body power strikes, flying in
    'power':   dict(load=PS.POWER_LOAD, hit=PS.POWER, rec=PS.WARRIOR, part='fistR', ground='both', tgt='chestC', y=0.715, solve=(50, 150), bk=(0.10, 0.06), fb=0.17, fw=0.17, sd=0.17),
    'power_l': dict(load=PS.POWER_LOAD_L, hit=PS.POWER_L, rec=PS.WARRIOR, part='fistL', ground='both', tgt='chestC', y=0.715, solve=(50, 150), ch='afL', bk=(0.10, 0.06), fb=0.17, fw=0.17, sd=0.17, lead='R'),
    'hay':     dict(load=PS.HAY_LOAD, hit=PS.HAY, rec=PS.HAY_STOP, part='fistR', ground='both', tgt='headC', y=0.715, solve=(60, 160), bk=(0.08, 0.12), fb=0.14, fw=0.15, sd=0.17),
    'fly':     dict(load=PS.FLY_LOAD, hit=PS.FLY_HIT, rec=PS.FLY_LOAD, part='fistR', ground=None, tgt='headC', bk=(0.30, 0.10)),
    'hammer2': dict(load=PS.THAMMER_UP, hit=PS.THAMMER_DOWN, rec=PS.WAR_FLOAT, part='fistR', ground=None, tgt='top', bk=(0.04, 0.08)),
})


def plan3(kind, az, c, ny, yawN, tgt=None, shell=False, npose=PS.FLOAT):
    """where Herobrine's pelvis goes for this strike (and the hit pose, the arm raised to the target's height for the palms)"""
    k = KIND[kind]; tg = tgt or k['tgt']
    d = fwd(yawN + az); yawH = yawN + az + 180
    T = n_pt(tg, c, ny, yawN, npose)
    u = np.array([d[0] * 0.55, 0.83, d[2] * 0.55]) if tg == 'top' else d
    D = T + u * (STANDOFF[tg] + (0.22 if shell else 0.0))
    part = PT.get(k['part'], k['part'])
    if k['ground'] and k.get('solve'):
        hit, ph = solve_arm(k['hit'], yawH, part, D, k.get('y', STANCE_Y), ch=k.get('ch', 'afR'), lo=k['solve'][0], hi=k['solve'][1])
    else:
        hit = k['hit']; ph = at(hit, yawH, part, D)
        if k['ground']: ph[1] = k.get('y', STANCE_Y)
    return ph, yawH, hit


def plan(kind, az, c, ny, yawN, tgt=None, shell=False, npose=PS.FLOAT):
    ph, yawH, _ = plan3(kind, az, c, ny, yawN, tgt, shell, npose)
    return ph, yawH


def solve_arm(pose, yaw, part, target, py, ch='afR', lo=60, hi=160):
    """pick the arm raise that puts `part` at the target's height with the pelvis at py -> (pose, pelvis)"""
    best = None
    for a in np.arange(lo, hi + 0.1, 1.0):
        p = dict(pose); p[ch] = a; ph = at(p, yaw, part, target)
        e = abs(ph[1] - py)
        if best is None or e < best[0]: best = (e, p, ph)
    p, ph = best[1], best[2].copy(); ph[1] = py
    return p, ph


def _sh(t): return t + SH if t >= SPLIT - 1e-6 else t


def shift(H, N, EV, HITS, LOCK):
    """move everything keyed in v4 time at or after SPLIT later by SH (the inserted bar)"""
    for a in (H, N):
        for c in a.k: a.k[c] = [(_sh(t), v) for t, v in a.k[c]]
        a.cuts = [_sh(t) for t in a.cuts]
        for sd in a.plants:
            for p in a.plants[sd]: p['t0'] = _sh(p['t0']); p['t1'] = _sh(p['t1'])
        a.looks = [(_sh(t0), _sh(t1), tg, w, r) for t0, t1, tg, w, r in a.looks]
        a.hidden = [(_sh(t0), _sh(t1)) for t0, t1 in a.hidden]
        a.stiffw = [(_sh(t0), _sh(t1), k) for t0, t1, k in a.stiffw]
        a.offs = [(_sh(t0), _sh(t1), _sh(t2), _sh(t3), v) for t0, t1, t2, t3, v in a.offs]
        if hasattr(a, '_holds'): a._holds = [(_sh(t0), _sh(t1), c) for t0, t1, c in a._holds]
    for e in EV:
        e['t'] = _sh(e['t'])
        if 't1' in e: e['t1'] = _sh(e['t1'])
    for h in HITS: h['t'] = _sh(h['t'])
    LOCK['t0'] = _sh(LOCK['t0']); LOCK['t1'] = _sh(LOCK['t1'])


def V5(m): return 1.77 + 0.4295 * m + (0.0185 if m >= 68 else 0.0)


T_CAS = V5(7)          # 4.777: the casual straight lands (and stops)
T_GO = V5(10)          # 6.065: gone (the v4 teleports start here)
FIST_LOOK = (5.27, 5.86)
T_V = V5(6)            # v6: he goes (on the beat)
T_A = 4.64             # v6: and is there


def casual(H, N, EV, HITS, adj):
    """v5 time: Notch settles; Herobrine walks up on the beat and throws a lazy straight - it stops dead; he looks at his fist"""
    # Notch settles in the air in front of him (eyes calm)
    N.key(4.20, PS.FLOAT, pos=(NX, 1.12, 0.0), yaw=YN1, eyes=1.6)
    N.key(4.62, PS.FLOAT, pos=(NX, NY + 0.01, 0.0), yaw=YN1)
    N.key(4.70, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1)
    N.key(4.86, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1, eyes=2.0)
    N.key(T_GO - 0.1, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1)
    EV.append(dict(kind='hover', t=4.62, x=NX, z=0.0))
    # where the fist stops: in front of his chest, a touch high
    yawH = 90.0; f = fwd(yawH)
    T = n_pt('headC', (NX, 0.0), NY, YN1, PS.FLOAT) + np.array([-0.40, -0.21, 0.0])
    punch, ph = solve_arm(PS.CASUAL_PUNCH, yawH, 'fistR', T, 0.73, ch='afR', lo=70, hi=150)
    print('casual straight: afR %.0f, pelvis %s' % (punch['afR'], np.round(ph, 2)))
    ph = ph + np.asarray(adj.get('c0', np.zeros(3)), float)
    cen = (ph - f * 0.14)[[0, 2]]
    fRe, fLe = feet(cen, yawH, lead='L', fw=0.2, sd=0.15)        # the stance he ends in
    xs = cen[0] - 5.6
    INTRO['xs'] = xs
    # v6: standing well back, watching him come down; on the beat he goes
    sR, sL = feet((xs, 0.0), yawH, fw=0.0, sd=0.13)
    H.key(0.0, PS.STAND, pos=(xs, 0.74, 0.0), yaw=yawH, eyes=1.2)
    H.key(1.6, PS.STAND, pos=(xs, 0.738, 0.0), yaw=yawH, eyes=2.4)
    H.key(2.62, PS.STAND, pos=(xs, 0.736, 0.0), yaw=yawH, eyes=2.6)
    H.key(3.70, PS.STAND, pos=(xs, 0.735, 0.0), yaw=yawH)
    lean = P3.P(**{**PS.STAND_TENSE, 'pitch': 9, 'lkR': 22, 'lkL': 22, 'lfR': 12, 'lfL': 12})
    H.key(4.16, lean, pos=(xs + 0.04, 0.715, 0.0), yaw=yawH, eyes=3.2)
    H.key(T_V - 0.002, lean, pos=(xs + 0.07, 0.712, 0.0), yaw=yawH, eyes=3.4)
    H.plant('R', 0.0, T_V, sR); H.plant('L', 0.0, T_V, sL)
    # the teleport: gone on the beat, back right in front of him, already moving into the straight
    H.hide(T_V, T_A); H.cut(T_A)
    EV.append(dict(kind='vanish', t=float(T_V), who='H', open=1)); EV.append(dict(kind='appear', t=float(T_A), who='H', open=1))
    H.plant('R', T_A, T_GO + 0.01, fRe)
    H.plant('L', T_A, 5.30, fLe); H.plant('L', 5.58, T_GO + 0.01, fLe - f[[0, 2]] * 0.42)
    H.key(T_A, P3.P(**{**PS.CASUAL_LOAD, 'pitch': PS.CASUAL_LOAD['pitch'] + 4}), pos=(ph[0] - 0.30, 0.724, 0.0), yaw=yawH, eyes=3.0)
    H.key(T_A + 0.06, PS.CASUAL_LOAD, pos=(ph[0] - 0.20, 0.726, 0.0), yaw=yawH)
    # the straight: out, and it stops dead
    H.key(T_CAS, punch, pos=tuple(ph), yaw=yawH, eyes=2.6)
    H.stiff(4.60, T_CAS + 0.03, 2.4)
    HITS.append(dict(id='c0', t=float(T_CAS), who='H', part='fistR', kind='barrier', big=1.2, hold=0.45, ground=True, crack=0, shatter=False))
    EV.append(dict(kind='casual', t=float(T_CAS), who='H'))
    # he leans on it - nothing; the fist trembles against the air
    press = P3.P(**{**punch, 'pitch': punch['pitch'] + 2, 'st': punch['st'] + 3, 'afL': -10, 'aeL': 30})
    for i, t in enumerate(np.arange(T_CAS + 0.10, 5.22, 0.055)):
        j = 1 if i % 2 else -1
        H.key(float(t), P3.P(**{**press, 'afR': punch['afR'] + 0.8 * j, 'aeR': 2 + 1.5 * (1 + j)}), pos=tuple(ph - f * (0.006 + 0.004 * j)), yaw=yawH)
    # draws it back, turns it over, looks at it
    H.key(5.30, P3.P(**{**press, 'aeR': 20}), pos=tuple(ph - f * 0.04), yaw=yawH)
    H.key(5.50, PS.LOOK_FIST, pos=tuple(ph - f * 0.30 + [0, 0.01, 0]), yaw=yawH)
    H.key(5.80, P3.P(**{**PS.LOOK_FIST, 'atR': 42, 'aeR': 96, 'afR': 84}), pos=tuple(ph - f * 0.33 + [0, 0.005, 0]), yaw=yawH, eyes=2.8)
    # looks back up at him; the eyes flare
    H.key(5.96, PS.SET, pos=tuple(ph - f * 0.34 + [0, -0.02, 0]), yaw=yawH, eyes=4.6)
    H.key(T_GO - 0.004, PS.SET, pos=tuple(ph - f * 0.34 + [0, -0.025, 0]), yaw=yawH, eyes=4.6)


INTRO = {}


def build(adj):
    H = K.Actor('H', seed=1); N = K.Actor('N', seed=2)
    H.spr = dict(H_SPR); H.noise_k = 0.6
    EV = []; HITS = []; LOCK = {}; LAST = {}

    def ev(kind, t, **kw):
        d = dict(kind=kind, t=float(t)); d.update(kw); EV.append(d); return d

    def A(sid): return np.asarray(adj.get(sid, np.zeros(3)), float)

    def tp(t_out, t_in, style='smoke'):
        """gone at t_out, back at t_in: a teleport (black smoke) or a flash step (red streak)"""
        H.hide(t_out, t_in); H.cut(t_in)
        if style == 'smoke': ev('vanish', t_out, who='H'); ev('appear', t_in, who='H')
        else: ev('flash', t_out, t1=t_in, who='H')

    def strike(sid, t, kind, az, c, ny, yawN, tin, tout, big=0.8, hold=0.04, shell=False, crack=0, shatter=False, tgt=None,
               keep_in=False, keep_out=False, plant=True, npose=PS.FLOAT, rec=None, load=None):
        k = KIND[kind]
        ph, yawH, hit = plan3(kind, az, c, ny, yawN, tgt, shell, npose)
        ph = ph + A(sid); f = fwd(yawH)
        bl, br = k.get('bk', (0.13, 0.11))
        pl = ph - f * bl + ([0, 0.04, 0] if not k['ground'] else 0)
        pr = ph - f * br
        if not keep_in: H.key(tin, load or k['load'], pos=tuple(pl), yaw=yawH)
        H.key(t, hit, pos=tuple(ph), yaw=yawH)
        if hold > 0: H.key(t + hold, hit, pos=tuple(ph - f * 0.01), yaw=yawH)
        if kind == 'hay' and rec is None:      # stopped dead: the arm stays where the barrier stopped it, the body pulls back
            rec = P3.P(**{**hit, 'st': hit['st'] - 4, 'aeR': hit['aeR'] - 6})
        if not keep_out: H.key(tout, rec or k['rec'], pos=tuple(pr), yaw=yawH)
        if k['ground'] and plant:
            fR, fL = feet((ph - f * k.get('fb', 0.14))[[0, 2]], yawH, lead=k.get('lead', 'L'), fw=k.get('fw', 0.2), sd=k.get('sd', 0.15))
            if k['ground'] == 'both': H.plant('R', tin, tout, fR)
            H.plant('L', tin, tout, fL)
        H.stiff(tin - 0.02, (tout + 0.02) if kind in ('hay', 'swat') else (t + 0.03), 2.4)
        HITS.append(dict(id=sid, t=float(t), who='H', part=k['part'], kind='shell' if shell else 'barrier', big=big, hold=hold,
                         ground=bool(k['ground']), crack=crack, shatter=shatter, win=3 if kind in ('swat', 'hay') else (6 if kind in ('palm', 'palm_hi', 'power', 'power_l', 'fly', 'fpalm', 'hammer2', 'smite') else 13)))
        LAST['hit'] = hit
        return ph, yawH
    # =================================================================================================================
    # NOTCH: comes down out of the sky, floats; never touches the ground, never turns round to look
    N.key(0.0, PS.FLOAT_DOWN, pos=(NX + 0.3, 9.6, 0.4), yaw=-112, eyes=0.0)
    N.key(1.77, PS.FLOAT_DOWN, pos=(NX + 0.22, 6.2, 0.3), yaw=-106)
    N.key(2.63, PS.FLOAT_DOWN, pos=(NX + 0.12, 3.9, 0.16), yaw=-99, eyes=0.3)
    N.key(3.49, PS.FLOAT, pos=(NX + 0.04, 2.1, 0.05), yaw=-93, eyes=1.4)
    N.key(5.206, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1, eyes=2.4)
    ev('descend', 0.0, dur=4.6)
    N.key(7.40, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1)
    # 7.78: an open hand - the push
    N.key(7.60, P3.P(**{**PS.HAND_UP, 'afR': 60, 'aeR': 40}), pos=(NX + 0.02, NY, 0.0), yaw=YN1)
    N.key(7.74, PS.HAND_UP, pos=(NX + 0.03, NY, 0.0), yaw=YN1)
    N.key(7.84, P3.P(**{**PS.HAND_UP, 'afR': 92, 'aeR': 4}), pos=(NX + 0.06, NY + 0.01, 0.0), yaw=YN1)
    N.key(8.40, PS.HAND_UP, pos=(NX + 0.04, NY, 0.0), yaw=YN1)
    N.key(9.00, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1)
    # the long front combo; one finger; the flick
    N.key(12.10, PS.FLOAT, pos=(NX, NY, 0.0), yaw=YN1)
    N.key(12.36, PS.FINGER, pos=(NX, NY, 0.0), yaw=YN1)
    N.key(12.47, P3.P(**{**PS.FINGER, 'afR': 88, 'aeR': 18}), pos=(NX, NY, 0.0), yaw=YN1)
    N.key(12.53, P3.P(**{**PS.FINGER, 'afR': 96, 'aeR': 0, 'st': -26}), pos=(NX + 0.02, NY, 0.0), yaw=YN1)
    N.key(12.95, PS.FINGER, pos=(NX, NY, 0.0), yaw=YN1)
    N.key(13.4, PS.FLOAT, pos=(NX - 0.1, NY, 0.0), yaw=YN1)
    N.key(14.6, PS.FLOAT, pos=(NX2, NY + 0.03, 0.0), yaw=YN1)
    N.key(15.10, PS.FLOAT, pos=(NX2, NY, 0.0), yaw=YN1)
    # 15.51 the palm (he glides in), 15.94 the backhand
    pa = A('palm'); bh = A('bh')
    N.key(15.34, P3.P(**{**PS.PALM, 'afR': 40, 'aeR': 90, 'st': 10}), pos=(NX2 + 0.05, NY - 0.10, 0.0), yaw=YN1)
    N.key(15.51, PS.PALM, pos=tuple(np.r_[NX2 - 0.30, NY - 0.12, 0.0] + pa), yaw=YN1)
    N.key(15.66, PS.PALM, pos=tuple(np.r_[NX2 - 0.22, NY - 0.08, 0.0] + pa * 0.7), yaw=YN1)
    N.key(15.80, PS.BACKHAND_LOAD, pos=(NX2 - 0.40, NY - 0.10, 0.0), yaw=YN1 - 6)
    N.key(15.944, PS.BACKHAND, pos=tuple(np.r_[NX2 - 0.46, NY - 0.10, 0.0] + bh), yaw=YN1 - 14)
    N.key(16.25, P3.P(**{**PS.BACKHAND, 'st': 50, 'aaR': 96}), pos=(NX2 - 0.48, NY - 0.05, 0.0), yaw=YN1 - 18)
    N.key(16.9, PS.FLOAT, pos=(NX2 - 0.45, NY, 0.0), yaw=-60)
    N.key(17.8, PS.FLOAT, pos=(NX2 - 0.4, NY, 0.0), yaw=-20)
    N.key(19.6, PS.FLOAT, pos=(NX2 - 0.4, NY + 0.03, 0.0), yaw=YN2)
    N.key(20.5, PS.ARMS_OPEN, pos=(NX2 - 0.4, NY + 0.06, 0.0), yaw=YN2, eyes=2.0)
    N.key(22.2, PS.FLOAT_CALM, pos=tuple(NC2), yaw=YN2, eyes=2.2)
    N.key(28.9, PS.FLOAT_CALM, pos=tuple(NC2), yaw=YN2)
    # the catch, the press (his root is then locked so the hand stays on Herobrine's fist)
    N.key(29.05, P3.P(**{**PS.CATCH, 'afR': 40, 'aeR': 40}), pos=tuple(NC2), yaw=YN2)
    N.key(29.24, PS.CATCH, pos=tuple(NC2), yaw=YN2, eyes=2.4)
    N.key(29.276, PS.CATCH, pos=tuple(NC2), yaw=YN2)
    N.key(30.95, P3.P(**{**PS.CATCH, 'nb': 8}), pos=tuple(NC2), yaw=YN2, eyes=2.6)
    N.key(31.42, PS.PRESS, pos=tuple(NC2), yaw=YN2, eyes=3.0)
    N.key(31.85, PS.PRESS, pos=tuple(NC2), yaw=YN2, eyes=3.0)
    N.key(32.10, PS.PRESS, pos=tuple(NC2), yaw=YN2, eyes=6.0)
    N.key(T_END, PS.PRESS, pos=tuple(NC2), yaw=YN2, eyes=6.0)

    # =================================================================================================================
    # HEROBRINE (v7): never a boxer.  The first few he throws like a god toying with him - one hand behind his back, open palms,
    # a backhand.  From 6.15 on he fights like Thragg: no guard, chest out, fists low; full-body power straights (the other fist
    # back at the hip), haymakers, flying punches and knees, two-fisted hammers from above; he rises into the air to power up
    H.look(0.6, T_END, 'headC', 1.0, 0.3)
    c1 = (NX, 0.0)
    # ---- before the drop: he blinks in behind Notch, at his side, in front - nothing lands
    tp(4.347, 4.47)
    strike('i1', 4.562, 'fpalm', 180, c1, 1.04, YN1, 4.47, 4.66, big=0.9)
    tp(4.66, 4.70)
    strike('i2', 4.777, 'swat', -80, c1, 1.0, YN1, 4.70, 4.88, big=0.9)
    tp(4.88, 5.06)
    ph3, y3 = strike('i3', 5.206, 'fpalm', 0, c1, NY, YN1, 5.06, 5.50, big=1.6, hold=0.26, keep_out=True)
    f = fwd(y3)
    # floats down in front of him and goes straight back to work, one hand: the face, the chest, then everywhere at once
    jb, _ = plan('palm', 0, c1, NY, YN1); jb = jb + A('b2')
    FS = (jb - f * 0.04)[[0, 2]]
    H.key(5.56, PS.GOD_FLOAT, pos=(FS[0] + f[0] * 0.06, 1.0, FS[1]), yaw=y3)
    H.key(5.72, PS.GOD, pos=(FS[0] + f[0] * 0.04, GY + 0.005, FS[1]), yaw=y3)
    gR, gL = feet(FS, y3, lead='L', fw=GFW, sd=0.15)
    H.plant('R', 5.70, 6.15, gR); H.plant('L', 5.70, 6.15, gL)
    strike('b1', 5.850, 'palm_hi', 0, c1, NY, YN1, 5.77, 5.95, plant=False)
    strike('b2', 6.065, 'palm', 0, c1, NY, YN1, 5.96, 6.15, big=1.0, plant=False)
    tp(6.15, 6.21)
    # ---- the hand comes out from behind his back: now he means it
    strike('b3', 6.280, 'fly', 170, c1, NY, YN1, 6.21, 6.37)
    tp(6.37, 6.43)
    strike('b4', 6.495, 'power', 95, c1, NY, YN1, 6.43, 6.58)
    tp(6.58, 6.64)
    strike('b5', 6.709, 'hammer2', 65, c1, NY, YN1, 6.64, 6.80, big=1.0)
    tp(6.80, 6.85)
    strike('b6', 6.924, 'hay', -10, c1, NY, YN1, 6.85, 7.01)
    tp(7.01, 7.07)
    strike('b7', 7.139, 'fly', -100, c1, NY, YN1, 7.07, 7.22, big=1.0)
    tp(7.22, 7.29)
    jb2, _ = plan('power', 0, c1, NY, YN1); jb2 = jb2 + A('b9'); FS2 = (jb2 - f * 0.10)[[0, 2]]
    gR, gL = feet(FS2, y3, lead='L', fw=0.10, sd=0.18)
    H.plant('R', 7.29, 7.82, gR); H.plant('L', 7.29, 7.82, gL)
    strike('b8', 7.354, 'power_l', 0, c1, NY, YN1, 7.29, 7.41, plant=False, keep_out=True)
    strike('b9', 7.461, 'power', 0, c1, NY, YN1, 7.41, 7.56, big=1.0, plant=False)
    H.key(7.70, PS.WARRIOR, pos=(FS2[0] + f[0] * 0.06, 0.72, FS2[1]), yaw=y3)
    # ---- 7.78 the push: he slides back on his feet the length of the yard - upright, no hand down
    ev('push', 7.783, who='N')
    P0 = np.array([FS2[0] + f[0] * 0.06, 0.72, FS2[1]]); X1 = -5.0
    H.key(7.783, PS.WARRIOR, pos=tuple(P0), yaw=y3)
    sR0, sL0 = feet(FS2, y3, lead='L', fw=0.14, sd=0.17); sR1, sL1 = feet((X1, FS2[1]), y3, lead='L', fw=0.14, sd=0.17)
    H.plant('R', 7.83, 8.75, sR0 + [-0.25, 0], sR1, ease='out'); H.plant('L', 7.83, 8.75, sL0 + [-0.25, 0], sL1, ease='out')
    for tk in (7.86, 8.0, 8.2, 8.45, 8.75):
        u = eo((tk - 7.83) / 0.92); x = (FS2[0] - 0.25) + (X1 - FS2[0]) * u + f[0] * 0.05
        H.key(tk, PS.SLIDE, pos=(x, 0.68 if tk > 7.9 else 0.70, FS2[1]), yaw=y3)
    ev('skid', 7.83, who='H', dur=0.92)
    # straightens, chest out, fists low; the eyes flare
    H.key(9.0, PS.WARRIOR, pos=(X1 + 0.05, 0.715, FS2[1]), yaw=y3, eyes=2.6)
    H.key(9.25, P3.P(**{**PS.WARRIOR, 'nb': -4, 'pitch': 5}), pos=(X1 + 0.07, 0.71, FS2[1]), yaw=y3, eyes=4.2)
    gR, gL = feet((X1 - f[0] * 0.02, FS2[1]), y3, lead='L', fw=0.12, sd=0.18)
    H.plant('R', 8.75, 9.30, sR1, gR); H.plant('L', 8.75, 9.30, sL1, gL)
    # ---- 9.3 - 12.5: flash steps - he's everywhere, faster than the eye
    seq = [('s1', 9.501, 'power', 45, 9.30, 9.44, 9.58), ('s2', 9.716, 'knee', 35, 9.58, 9.65, 9.80),
           ('s3', 9.931, 'hay', -20, 9.80, 9.86, 10.02), ('s4', 10.145, 'fly', 150, 10.02, 10.08, 10.23),
           ('s5', 10.360, 'hammer2', 100, 10.23, 10.29, 10.45), ('s6', 10.575, 'power', 40, 10.45, 10.51, 10.66),
           ('s7', 10.790, 'fly', -120, 10.66, 10.72, 10.88)]
    for sid, t, kind, az, t_out, tin, tout in seq:
        tp(t_out, tin, 'flash'); strike(sid, t, kind, az, c1, NY, YN1, tin, tout)
    tp(10.88, 10.94, 'flash')
    # planted square in front of him: a flurry of full power straights, left, right, left - each one the whole body
    jb3, _ = plan('power', 0, c1, NY, YN1); jb3 = jb3 + A('f6'); FS3 = (jb3 - f * 0.10)[[0, 2]]
    gR, gL = feet(FS3, y3, lead='L', fw=0.08, sd=0.19)
    H.plant('R', 10.94, 12.52, gR); H.plant('L', 10.94, 12.52, gL)
    combo = [('f1', 11.004, 'power_l'), ('f2', 11.219, 'power'), ('f3', 11.434, 'power_l'), ('f4', 11.649, 'power'), ('f5', 11.863, 'power_l')]
    for i, (sid, t, kind) in enumerate(combo):
        strike(sid, t, kind, 0, c1, NY, YN1, t - (0.07 if i == 0 else 0.10), t + 0.10, plant=False, big=0.7, keep_in=i > 0, keep_out=True)
    H.stiff(10.9, 11.95, 1.3)
    phF, _ = strike('f6', 12.078, 'power', 0, c1, NY, YN1, 11.97, 12.50, big=1.4, hold=0.40, plant=False, keep_out=True, keep_in=True)
    hitF = LAST['hit']
    H.key(12.46, P3.P(**{**hitF, 'nb': 0}), pos=tuple(phF - f * 0.005), yaw=y3, eyes=4.0)
    # ---- 12.51 the flick: thrown back into the wall (no tumbling - he flies back flat and hits it)
    ev('flick', 12.508, who='N')
    H.key(12.508, hitF, pos=tuple(phF - f * 0.01), yaw=y3)
    H.key(12.58, PS.THROWN, pos=tuple(phF - f * 0.75 + [0, 0.25, 0]), yaw=y3)
    H.key(12.80, PS.THROWN, pos=(WALLX + 2.1, 1.40, 0.0), yaw=y3, pitch=-30)
    H.key(12.937, P3.P(**{**PS.THROWN, 'afR': 140, 'afL': 140, 'aaR': 60, 'aaL': 60, 'lfR': 40, 'lfL': 30}), pos=(WALLX + 0.82, 1.45, 0.0), yaw=y3, pitch=-14)
    ev('wall_break', 12.937, x=WALLX, z=0.0)
    H.key(13.06, PS.THROWN, pos=(WALLX + 0.95, 1.10, 0.05), yaw=y3, pitch=0)
    H.key(13.26, PS.SUPERHERO, pos=(WALLX + 1.25, 0.40, 0.1), yaw=y3)
    RB = np.array([WALLX + 1.35, 0.1])
    rR, rL = feet(RB, y3, lead='L', fw=0.25, sd=0.16)
    H.plant('R', 13.22, 14.65, rR - [0.10, 0], rR); H.plant('L', 13.24, 14.65, rL - [0.10, 0], rL)
    H.key(13.80, PS.SUPERHERO, pos=(RB[0], 0.42, RB[1]), yaw=y3, eyes=2.0)
    # stands up straight out of the rubble, unhurt, fists low
    H.key(14.10, PS.RISE, pos=(RB[0], 0.66, RB[1]), yaw=y3, eyes=4.5)
    H.key(14.40, PS.WARRIOR, pos=(RB[0] - 0.02, 0.69, RB[1]), yaw=y3)
    H.key(14.60, P3.P(**{**PS.WARRIOR, 'pitch': 8}), pos=(RB[0], 0.68, RB[1]), yaw=y3)
    ev('rise', 13.8, who='H', dur=0.9)
    # ---- 14.66 gone; 15.09 in Notch's face in mid-air (Notch has drifted to NX2)
    tp(14.655, 14.99)
    c2 = (NX2, 0.0)
    ph4, y4 = strike('t1', 15.085, 'fly', 0, c2, NY, YN1, 14.99, 15.30, big=1.5, hold=0.12, keep_out=True)
    f4 = fwd(y4)
    jb4, _ = plan('jab', 0, c2, NY, YN1); FS4 = (jb4 - f4 * 0.20)[[0, 2]] + A('palmH')[[0, 2]]
    H.key(15.30, PS.WAR_FLOAT, pos=(FS4[0] + 0.1, 0.94, FS4[1]), yaw=y4)
    H.key(15.42, PS.WARRIOR, pos=(FS4[0], 0.715, FS4[1]), yaw=y4)
    g4R, g4L = feet(FS4 - f4[[0, 2]] * 0.04, y4, lead='L', fw=0.12, sd=0.17)
    H.plant('R', 15.40, 15.60, g4R); H.plant('L', 15.39, 15.60, g4L)
    # 15.51 Notch's palm into his stomach: folds over it and slides back
    HITS.append(dict(id='palm', t=15.514, who='N', part='fistR', kind='body', target='belly', big=1.4))
    H.key(15.514, PS.WARRIOR, pos=(FS4[0], 0.715, FS4[1]), yaw=y4)
    H.key(15.62, PS.DOUBLED, pos=(FS4[0] - 0.40, 0.62, FS4[1]), yaw=y4)
    H.key(15.80, PS.DOUBLED, pos=(FS4[0] - 0.62, 0.60, FS4[1]), yaw=y4)
    H.plant('R', 15.62, 15.86, g4R - [0.45, 0], g4R - [0.66, 0], ease='out'); H.plant('L', 15.64, 15.86, g4L - [0.43, 0], g4L - [0.64, 0], ease='out')
    H.key(15.90, P3.P(**{**PS.DOUBLED, 'sb': 30, 'nb': 0}), pos=(FS4[0] - 0.64, 0.64, FS4[1]), yaw=y4)
    # 15.94 the backhand: his head snaps round, he's thrown sideways and slides to a stop on his feet, upright, turning to face him
    HITS.append(dict(id='bh', t=15.944, who='N', part='fistR', kind='body', target='headC', big=1.6))
    xb = FS4[0] - 0.64
    LZ = np.array([xb - 0.45, 5.0])
    H.key(15.944, PS.HEAD_SNAP, pos=(xb, 0.72, 0.06), yaw=y4 + 8)
    H.key(16.06, PS.SIDE_SKID, pos=(xb - 0.05, 0.86, 0.95), yaw=y4 + 24)
    H.key(16.20, PS.SIDE_SKID, pos=(xb - 0.12, 0.58, 2.1), yaw=y4 + 50)
    lzR0, lzL0 = feet((xb - 0.12, 2.1), y4 + 50, lead='L', fw=0.22, sd=0.2); lzR1, lzL1 = feet(LZ, YL, lead='L', fw=0.16, sd=0.17)
    H.plant('R', 16.20, 17.00, lzR0, lzR1, ease='out'); H.plant('L', 16.22, 17.00, lzL0, lzL1, ease='out')
    for tk in (16.40, 16.65, 17.00):
        u = eo((tk - 16.20) / 0.80)
        H.key(tk, PS.SLIDE if tk > 16.5 else PS.SIDE_SKID, pos=(xb - 0.12 + (LZ[0] - xb + 0.12) * u, 0.67 if tk > 16.5 else 0.56, 2.1 + (LZ[1] - 2.1) * u),
              yaw=y4 + 50 + (YL - y4 - 50) * min(1, u * 1.3))
    ev('skid', 16.20, who='H', dur=0.80)
    # straightens; calls the storm down into his raised hand; then rises off the ground, arms open
    LZ2 = LZ
    zR, zL = feet(LZ2, YL, lead='L', fw=0.10, sd=0.17)
    H.plant('R', 17.00, 17.95, lzR1); H.plant('R', 18.10, 19.30, zR)
    H.plant('L', 17.00, 18.0, lzL1); H.plant('L', 18.18, 19.30, zL)
    H.key(17.40, PS.SLIDE, pos=(LZ2[0], 0.68, LZ2[1]), yaw=YL)
    H.key(17.80, PS.RISE, pos=(LZ2[0], 0.71, LZ2[1]), yaw=YL, eyes=3.0)
    H.key(18.20, PS.WARRIOR, pos=(LZ2[0], 0.72, LZ2[1]), yaw=YL)
    H.key(18.60, PS.SUMMON, pos=(LZ2[0], 0.735, LZ2[1]), yaw=YL)
    ev('summon', 18.3, who='H')
    H.key(18.95, P3.P(**{**PS.SUMMON, 'afR': 178}), pos=(LZ2[0], 0.74, LZ2[1]), yaw=YL, eyes=6.0)
    ev('bolt', 18.95, who='H')
    H.key(19.20, PS.SUMMON, pos=(LZ2[0], 0.735, LZ2[1]), yaw=YL)
    # lifts off: the power comes up into him (no roar, no crouch - he rises and hangs in the air)
    H.key(19.50, P3.P(**{**PS.ASCEND, 'aaR': 46, 'aaL': 46, 'afR': 16, 'afL': 16}), pos=(LZ2[0], 0.80, LZ2[1]), yaw=YL)
    H.key(19.809, PS.ASCEND, pos=(LZ2[0], 1.00, LZ2[1]), yaw=YL, eyes=7.0)
    ev('roar', 19.809, who='H')
    H.key(20.60, PS.ASCEND, pos=(LZ2[0], 1.10, LZ2[1]), yaw=YL)
    ev('charge', 20.67, who='H', dur=5.15)
    CH2 = P3.P(**{**PS.CHARGE, 'aaR': 32, 'aaL': 32, 'aeR': 26, 'aeL': 26})
    CH3 = P3.P(**{**PS.CHARGE, 'sb': 8, 'nb': 14})
    CH4 = P3.P(**{**PS.CHARGE, 'sb': 10, 'nb': 16, 'pitch': 6})
    H.key(21.6, PS.CHARGE, pos=(LZ2[0], 1.14, LZ2[1]), yaw=YL)
    H.key(22.386, CH2, pos=(LZ2[0], 1.17, LZ2[1]), yaw=YL, eyes=7.5)
    H.key(23.2, CH2, pos=(LZ2[0], 1.15, LZ2[1]), yaw=YL)
    H.key(24.10, CH3, pos=(LZ2[0], 1.19, LZ2[1]), yaw=YL, eyes=8.0)
    H.key(24.8, CH3, pos=(LZ2[0], 1.17, LZ2[1]), yaw=YL)
    H.key(25.39, CH4, pos=(LZ2[0], 1.20, LZ2[1]), yaw=YL)
    H.key(25.58, CH4, pos=(LZ2[0], 1.20, LZ2[1]), yaw=YL)
    # =================================================================================================================
    # DROP 2: now the barrier shows - every strike cracks it further
    cN = (NC2[0], NC2[2]); ny2 = NC2[1]
    tp(25.60, 25.74)
    ph5, y5 = strike('d1', 25.822, 'fly', 0, cN, ny2, YN2, 25.74, 26.0, big=2.0, hold=0.14, shell=True, crack=1, keep_out=True, npose=PS.FLOAT_CALM)
    f5 = fwd(y5)
    st5, _ = plan('power', 0, cN, ny2, YN2, shell=True, npose=PS.FLOAT_CALM); st5 = st5 + A('d2'); FS5 = (st5 - f5 * 0.17)[[0, 2]]
    H.key(26.04, PS.WAR_FLOAT, pos=(FS5[0] + f5[0] * 0.12, 0.92, FS5[1] + f5[2] * 0.12), yaw=y5)
    H.key(26.14, PS.WARRIOR, pos=(FS5[0] + f5[0] * 0.10, 0.70, FS5[1] + f5[2] * 0.10), yaw=y5)
    gR, gL = feet(FS5, y5, lead='L', fw=0.17, sd=0.17)
    H.plant('R', 26.12, 26.36, gR); H.plant('L', 26.11, 26.36, gL)
    strike('d2', 26.252, 'power', 0, cN, ny2, YN2, 26.16, 26.36, big=1.2, shell=True, crack=2, plant=False, npose=PS.FLOAT_CALM)
    tp(26.36, 26.60)
    strike('d3', 26.681, 'knee', -70, cN, ny2, YN2, 26.60, 26.78, big=1.2, shell=True, crack=3, npose=PS.FLOAT_CALM)
    tp(26.78, 27.03)
    strike('d4', 27.110, 'fly', 180, cN, ny2, YN2, 27.03, 27.20, big=1.2, shell=True, crack=4, npose=PS.FLOAT_CALM)
    tp(27.20, 27.47)
    strike('d5', 27.540, 'hay', 80, cN, ny2, YN2, 27.47, 27.64, big=1.2, shell=True, crack=5, npose=PS.FLOAT_CALM)
    tp(27.64, 27.89)
    strike('d6', 27.970, 'power_l', -20, cN, ny2, YN2, 27.89, 28.06, big=1.2, shell=True, crack=6, npose=PS.FLOAT_CALM)
    tp(28.06, 28.24)
    ph7, y7 = strike('d7', 28.399, 'hammer2', 0, cN, ny2, YN2, 28.24, 28.56, big=2.2, hold=0.05, shell=True, crack=7, shatter=True, keep_out=True, npose=PS.FLOAT_CALM)
    f7 = fwd(y7)
    # ---- drops through the pieces, lands right in front of him; the last power straight - caught
    K_ = n_pt('fistR', cN, ny2, YN2, PS.CATCH)          # Notch's open hand
    D_ = K_ + fwd(YN2) * 0.02
    drive, phD = solve_arm(PS.POWER, y7, 'fistR', D_, STANCE_Y - 0.01)
    phD = phD + A('catch')
    FSc = (phD - f7 * 0.17)[[0, 2]]
    H.key(28.56, PS.WAR_FLOAT, pos=tuple(ph7 - f7 * 0.15 + [0, -0.45, 0]), yaw=y7)
    H.key(28.74, P3.P(**{**PS.WARRIOR, 'sb': 10, 'pitch': 6}), pos=(FSc[0] + f7[0] * 0.04, 0.62, FSc[1] + f7[2] * 0.04), yaw=y7)
    cR, cL = feet(FSc, y7, lead='L', fw=0.17, sd=0.17)
    H.plant('R', 28.72, 33.0, cR); H.plant('L', 28.71, 33.0, cL)
    load = PS.POWER_LOAD
    H.key(28.829, load, pos=(FSc[0] + f7[0] * 0.02, 0.67, FSc[1] + f7[2] * 0.02), yaw=y7, eyes=8.0)
    H.key(29.10, load, pos=(FSc[0], 0.68, FSc[1]), yaw=y7)
    H.key(29.276, drive, pos=tuple(phD), yaw=y7)
    HITS.append(dict(id='catch', t=29.276, who='H', part='fistR', kind='catch', big=2.4))
    ev('catch', 29.276, who='N')
    H.key(29.70, drive, pos=tuple(phD + f7 * 0.02), yaw=y7)
    H.key(30.13, P3.P(**{**drive, 'pitch': drive['pitch'] + 4}), pos=tuple(phD + f7 * 0.04 + [0, -0.02, 0]), yaw=y7)
    H.key(30.56, drive, pos=tuple(phD + f7 * 0.03), yaw=y7)
    H.key(30.995, drive, pos=tuple(phD + f7 * 0.02), yaw=y7)
    # pressed down onto his right knee, still pushing up into the hand
    kp = n_pt('fistR', cN, ny2 - 0.06, YN2, PS.PRESS) + fwd(YN2) * 0.02
    kneel, phK = solve_arm(PS.GOD_KNEEL, y7, 'fistR', kp, 0.47, lo=80, hi=170)
    H.key(31.25, P3.P(**{**kneel, 'lkR': 90}), pos=tuple(np.r_[FSc[0], 0.58, FSc[1]] * 0.5 + phK * 0.5), yaw=y7)
    H.key(31.42, kneel, pos=tuple(phK), yaw=y7)
    ev('kneel', 31.42, who='H')
    H.key(32.0, kneel, pos=tuple(phK + [0, 0.01, 0]), yaw=y7, eyes=8.0)
    H.key(T_END, kneel, pos=tuple(phK + [0, 0.01, 0]), yaw=y7)
    ev('flare', 31.853, who='N')
    LOCK.update(t0=29.17, t1=29.27)
    shift(H, N, EV, HITS, LOCK)
    casual(H, N, EV, HITS, adj)
    return H, N, EV, HITS, LOCK


# =====================================================================================================================
def notch_look(B, A, ts):
    """Notch looks ahead, calm; his eyes follow Herobrine only while he is in front of him"""
    P = B['P']; Q = B['Q']; fk = R.fk(P, Q); fa = R.fk(A['P'], A['Q'])
    qp = Q[:, R.BI['pelvis']]; fw = R.qrot(qp, np.array([0, 0, 1.0])); fw[:, 1] = 0; fw /= np.linalg.norm(fw, axis=1, keepdims=True)
    head = fk['headC']; ah = fa['headC']
    d = ah - head; dh = d.copy(); dh[:, 1] = 0; dh /= np.linalg.norm(dh, axis=1, keepdims=True) + 1e-9
    cosang = np.sum(dh * fw, 1)
    see = (cosang > 0.42) & A['vis']
    w = np.convolve(see.astype(float), np.ones(25) / 25, 'same')
    ahead = head + fw * 3.0 + np.array([0, -0.15, 0])
    tgt = ahead * (1 - w[:, None]) + ah * w[:, None]
    tgt[ts < 4.3] = ah[ts < 4.3]
    chest = Q[:, R.BI['chest']]
    hp = fk['head'] + R.qrot(chest, np.array([0, 3 * K.PX, 0]))
    dl = R.qrot(R.qconj(chest), tgt - hp)
    nt = np.clip(np.degrees(np.arctan2(dl[:, 0], dl[:, 2])), -60, 60); nb = np.clip(np.degrees(np.arctan2(-dl[:, 1], np.hypot(dl[:, 0], dl[:, 2]))), -35, 45)
    C = B['C']; ww = np.clip((ts - 0.5) / 0.6, 0, 1)
    nt = C['nt'] * (1 - ww) + nt * ww; nb = C['nb'] * (1 - ww) + nb * ww
    nt = K.spring(nt, 3.0, 0.85, [0]); nb = K.spring(nb, 3.0, 0.85, [0])
    qh = K.mul(K.ry(nt), K.rx(nb), K.rz(-C['ns']))
    Q[:, R.BI['head']] = R.qmul(chest, qh)


def look_fist(A, ts):
    """FIST_LOOK window: Herobrine's head turns down to his own right fist (blended in and out)"""
    t0, t1 = FIST_LOOK; ramp = 0.16
    u = np.minimum(K.ss_((ts - t0) / ramp), K.ss_((t1 - ts) / ramp))
    m = u > 0
    if not m.any(): return
    P = A['P'][m]; Q = A['Q'][m]; f = R.fk(P, Q)
    head = f['head'] + R.qrot(Q[:, R.BI['chest']], np.array([0, 3 * K.PX, 0]))
    tgt = f['fistR'] + np.array([0, -0.02, 0])
    dl = R.qrot(R.qconj(Q[:, R.BI['chest']]), tgt - head)
    nt = np.clip(np.degrees(np.arctan2(dl[:, 0], dl[:, 2])), -70, 70); nb = np.clip(np.degrees(np.arctan2(-dl[:, 1], np.hypot(dl[:, 0], dl[:, 2]))), -40, 45)
    nt = K.spring(nt, 4.0, 0.8, [0]); nb = K.spring(nb, 4.0, 0.8, [0])
    qh = R.qmul(Q[:, R.BI['chest']], K.mul(K.ry(nt), K.rx(nb)))
    A['Q'][m, R.BI['head']] = R.qslerp(A['Q'][m, R.BI['head']], qh, u[m][:, None])


def bake(H, N, LOCK):
    ts = np.arange(0, TB + 1e-9, 1.0 / K.HZ)
    A = H.bake(ts); A['actor'] = H
    B = N.bake(ts); B['actor'] = N
    bob = 0.025 * np.sin(2 * np.pi * 0.42 * ts) + 0.01 * np.sin(2 * np.pi * 1.1 * ts + 1.0)
    B['P'][:, 1] += bob * np.clip((ts - 3.6) / 1.0, 0, 1) * np.clip((29.0 + SH - ts) / 0.3, 0, 1)
    if LOCK:
        # the catch: Notch's open hand stays on Herobrine's fist (his whole body floats with it)
        fa = R.fk(A['P'], A['Q']); fb = R.fk(B['P'], B['Q'])
        fist = fa['fistR']; arm = fist - fa['farmR']; arm /= np.linalg.norm(arm, axis=1, keepdims=True) + 1e-9
        want = fist + arm * 0.015
        w = K.ss_((ts - LOCK['t0']) / (LOCK['t1'] - LOCK['t0']))[:, None]
        B['P'] += w * (want - fb['fistR'])
    K.head_look(A, B, ts)
    look_fist(A, ts)
    notch_look(B, A, ts)
    return ts, A, B


def shell_target(p, c, rad=0.13):
    """the point just outside the barrier's ellipsoid on the ray from its centre through p"""
    d = p - c; sx = np.array([SHELL['rx'], SHELL['ry'], SHELL['rx']])
    s = np.linalg.norm(d / sx); surf = c + d / s
    n = (surf - c) / sx ** 2; n /= np.linalg.norm(n)
    return surf + n * rad


def align(iters=5):
    adj = {}
    for it in range(iters):
        H, N, EV, HITS, LOCK = build(adj); ts, A, B = bake(H, N, LOCK); errs = []
        for h in HITS:
            if h['kind'] == 'catch': continue
            i = int(round(h['t'] * K.HZ))
            att, de = (A, B) if h['who'] == 'H' else (B, A)
            if h['kind'] == 'shell':
                pp = R.fk(att['P'][i:i + 1], att['Q'][i:i + 1])[PT.get(h['part'], h['part'])][0]
                c = de['P'][i] + [0, SHELL['dy'], 0]
                vec = shell_target(pp, c) - pp
                cl, _, _ = K.part_closest(att, CAPS[h['part']], de); m = cl[i - 4:i + h.get('win', 13)].min()
                if m < GAP:                # never closer to his body than the invisible barrier
                    vec = vec + (GAP - m) * (pp - c) / (np.linalg.norm(pp - c) + 1e-9)
            else:
                cl, pa, pb = K.part_closest(att, CAPS[h['part']], de)
                j = i - 4 + int(np.argmin(cl[i - 4:i + h.get('win', 13)]))          # the closest it gets round the contact (arms lag a little)
                u = pb[j] - pa[j]; u /= np.linalg.norm(u) + 1e-9
                gp = GAP if h['kind'] == 'barrier' else 0.0
                vec = u * (cl[j] - gp)
            if h.get('ground') or h['who'] == 'N': vec = vec * np.array([1, 0, 1])
            errs.append(np.linalg.norm(vec))
            if it == iters - 1 and np.linalg.norm(vec) > 0.04: print('   err', h['id'], np.round(vec, 3))
            adj[h['id']] = np.asarray(adj.get(h['id'], np.zeros(3))) + vec * 0.95
        # the landing spots follow the first strike thrown from them
        print('align pass %d: max err %.3f mean %.3f' % (it, max(errs), np.mean(errs)))
    return adj


def report(ts, A, B, HITS):
    cl, pair, pt = K.clearance(A, B)
    print('--- hits')
    for h in HITS:
        i = int(round(h['t'] * K.HZ))
        att, de = (A, B) if h['who'] == 'H' else (B, A)
        pc = K.part_clearance(att, CAPS[h['part']], de)
        print('  %6.3f %-6s %s %-6s %-7s clearance %.3f  min(+-50ms) %.3f' % (h['t'], h['id'], h['who'], h['part'], h['kind'], pc[i], pc[max(0, i - 6):i + 7].min()))
    print('--- interpenetration (clearance < -0.02) outside contacts')
    bad = cl < -0.02
    for h in HITS:
        if h['kind'] in ('body', 'catch'): bad[int((h['t'] - 0.08) * K.HZ):int((h['t'] + (4 if h['kind'] == 'catch' else 0.12)) * K.HZ)] = False
    idx = np.where(bad)[0]
    if len(idx):
        runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
        for r in runs[:30]: print('  %.2f-%.2f  min %.2f  %s' % (ts[r[0]], ts[r[-1]], cl[r].min(), pair[r[np.argmin(cl[r])]]))
    else: print('  none')
    print('--- closest approach per second'); s = ''
    for k in range(int(TB)):
        m = (ts >= k) & (ts < k + 1); s += '%d:%.2f ' % (k, cl[m].min())
    print(' ', s)
    for nm, X in (('H', A), ('N', B)):
        f = R.fk(X['P'], X['Q']); low = np.min(np.stack([f[k][:, 1] for k in ('soleR', 'soleL', 'fistR', 'fistL', 'headC', 'shinR', 'shinL')], 1), 1)
        m = X['vis'] & (low < -0.04)
        if m.any(): print('  %s below ground at %s' % (nm, np.round(ts[m][::12], 2)))
    fb = R.fk(B['P'], B['Q']); ns = np.minimum(fb['soleR'][:, 1], fb['soleL'][:, 1])
    print('  Notch lowest sole after 4.4 s: %.2f' % ns[ts > 4.4].min())


if __name__ == '__main__':
    adj = align()
    H, N, EV, HITS, LOCK = build(adj)
    ts, A, B = bake(H, N, LOCK)
    report(ts, A, B, HITS)
    np.savez_compressed(os.path.join(HERE, 'story7.npz'), ts=ts, HP=A['P'], HQ=A['Q'], NP=B['P'], NQ=B['Q'], Hvis=A['vis'], Nvis=B['vis'], Heyes=A['eyes'], Neyes=B['eyes'])
    json.dump(dict(events=EV, hits=HITS, shell=SHELL, xs=INTRO.get('xs'), adj={k: list(map(float, v)) for k, v in adj.items()}), open(os.path.join(HERE, 'story7_events.json'), 'w'), indent=0)
    print('saved')
