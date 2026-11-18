"""v3: hand-keyed animation for the blocky rig.
Every joint is a few named angles (degrees) with hard anatomical limits: elbows and knees are pure hinges, shoulders and hips
swing + a limited twist, spine and neck bend / twist / tilt inside human ranges -> no limb can twist or bend the wrong way.
Keys are joined with shape-preserving cubic curves (no overshoot, no stops between keys that keep moving), then each joint
runs through a damped spring (follow-through and overlap: the head and limbs trail the body a little), then feet are
pinned to the ground with two-bone IK while planted, and heads track a target.  -> P (pelvis world position), Q (11 world quats).

Model faces +Z, up +Y, the character's left is +X.
  root  x y z (m)  yaw (deg, 0 = facing +Z, 90 = facing +X)  pitch (+ = lean forward)  roll (+ = lean to own left)
  spine sb (+ bend forward) st (+ turn to own left) ss (+ lean to own left)          [chest relative to pelvis]
  neck  nb (+ nod down) nt (+ turn left) ns (+ tilt left)                            [head relative to chest]
  arms  af (+ raise forward; 90 = pointing ahead, 180 = overhead)  aa (+ out to the side)  at (+ turn the arm's front inwards)
        ae elbow flex 0..150
  legs  lf (+ thigh forward)  la (+ out to the side)  lt (+ toe out)  lk knee flex 0..150
"""
import numpy as np
from scipy.interpolate import PchipInterpolator
import rig as R

HZ = 120
DEG = np.pi / 180
PX = R.PX
SIDES = ('R', 'L')
CH = ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'sb', 'st', 'ss', 'nb', 'nt', 'ns'] + \
     [a + s for s in SIDES for a in ('af', 'aa', 'at', 'ae')] + [a + s for s in SIDES for a in ('lf', 'la', 'lt', 'lk')]
LIM = {'sb': (-35, 75), 'st': (-55, 55), 'ss': (-32, 32), 'nb': (-45, 50), 'nt': (-80, 80), 'ns': (-30, 30),
       'af': (-75, 200), 'aa': (-25, 178), 'at': (-80, 80), 'ae': (0, 150),
       'lf': (-50, 135), 'la': (-15, 75), 'lt': (-35, 40), 'lk': (0, 155)}
SPR = {'pitch': (6.5, 0.75), 'roll': (6.5, 0.75), 'sb': (5.5, 0.62), 'st': (5.5, 0.62), 'ss': (5.0, 0.62),
       'nb': (4.2, 0.6), 'nt': (4.2, 0.66), 'ns': (4.2, 0.6),
       'af': (7.5, 0.56), 'aa': (7.5, 0.56), 'at': (7.0, 0.62), 'ae': (8.5, 0.52),
       'lf': (8.0, 0.62), 'la': (8.0, 0.62), 'lt': (8.0, 0.65), 'lk': (8.5, 0.58)}
NOISE = {'sb': 1.0, 'st': 1.2, 'ss': 0.8, 'nb': 1.4, 'nt': 1.6, 'ns': 1.0, 'af': 2.0, 'aa': 1.4, 'at': 1.5, 'ae': 2.5,
         'lf': 0.0, 'la': 0.0, 'lt': 0.0, 'lk': 0.0, 'pitch': 0.5, 'roll': 0.6}
L_LEG = 6 * PX


def base(c):
    return c[:-1] if c[-1] in SIDES and c[:-1] in LIM else c


def mirror(p):
    """left <-> right copy of a pose dict"""
    out = {}
    for c, v in p.items():
        if c[-1] in SIDES and c[:-1] in LIM: out[c[:-1] + ('L' if c[-1] == 'R' else 'R')] = v
        elif c in ('st', 'ss', 'nt', 'ns', 'roll'): out[c] = -v
        else: out[c] = v
    return out


# ---------------------------------------------------------------- rotations (vectorised, degrees)
def rx(a): return R.qaxis(np.array([1.0, 0, 0]), np.asarray(a) * DEG)
def ry(a): return R.qaxis(np.array([0, 1.0, 0]), np.asarray(a) * DEG)
def rz(a): return R.qaxis(np.array([0, 0, 1.0]), np.asarray(a) * DEG)
def mul(*qs):
    q = qs[0]
    for b in qs[1:]: q = R.qmul(q, b)
    return q


def local_quats(C):
    """C: dict channel -> (n,) -> dict bone -> local quaternion (n,4)"""
    L = {}
    L['pelvis'] = mul(ry(C['yaw']), rx(C['pitch']), rz(-C['roll']))
    L['chest'] = mul(ry(C['st']), rx(C['sb']), rz(-C['ss']))
    L['head'] = mul(ry(C['nt']), rx(C['nb']), rz(-C['ns']))
    for s, sg in (('R', -1), ('L', 1)):
        L['uarm' + s] = mul(rx(-C['af' + s]), rz(sg * C['aa' + s]), ry(-sg * C['at' + s]))
        L['farm' + s] = rx(-C['ae' + s])
        L['thigh' + s] = mul(rx(-C['lf' + s]), rz(sg * C['la' + s]), ry(sg * C['lt' + s]))
        L['shin' + s] = rx(C['lk' + s])
    return L


PARENT = {'chest': 'pelvis', 'head': 'chest', 'uarmR': 'chest', 'farmR': 'uarmR', 'uarmL': 'chest', 'farmL': 'uarmL',
          'thighR': 'pelvis', 'shinR': 'thighR', 'thighL': 'pelvis', 'shinL': 'thighL'}


def world_quats(L):
    W = {'pelvis': L['pelvis']}
    for b in R.BONES[1:]: W[b] = R.qmul(W[PARENT[b]], L[b])
    return W


# ---------------------------------------------------------------- springs and curves
def spring(x, f, z, resets, dt=1.0 / HZ, sub=4, fm=None):
    """second-order follower of x (n,), restarted on the target at reset indices; fm scales the frequency per sample"""
    h = dt / sub
    y = np.empty_like(x); p = x[0]; v = 0.0; rs = set(int(i) for i in resets)
    for i in range(len(x)):
        if i in rs: p = x[i]; v = 0.0
        tgt = x[i]; w = 2 * np.pi * f * (fm[i] if fm is not None else 1.0)
        for _ in range(sub):
            a = w * w * (tgt - p) - 2 * z * w * v; v += a * h; p += v * h
        y[i] = p
    return y


def curve(keys, ts, default=0.0):
    if not keys: return np.full(len(ts), float(default))
    d = {}
    for t, v in sorted(keys, key=lambda k: k[0]): d[round(t, 5)] = float(v)
    kt = np.array(sorted(d)); kv = np.array([d[t] for t in kt])
    if len(kt) == 1: return np.full(len(ts), kv[0])
    f = PchipInterpolator(kt, kv, extrapolate=False)
    y = f(np.clip(ts, kt[0], kt[-1]))
    return y


def smooth_noise(ts, seed, amp, freqs=(0.23, 0.41, 0.73)):
    rng = np.random.default_rng(seed); y = np.zeros(len(ts))
    for f in freqs:
        y += np.sin(2 * np.pi * f * rng.uniform(0.8, 1.25) * ts + rng.uniform(0, 6.3)) / len(freqs)
    return y * amp


def ss_(x):
    x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)


# ---------------------------------------------------------------- the actor
class Actor:
    def __init__(self, name, seed=0):
        self.name = name; self.seed = seed
        self.k = {c: [] for c in CH}; self.k['eyes'] = []; self.k['alive'] = []
        self.cuts = []                      # times where the body may jump (teleports): springs restart there
        self.plants = {'R': [], 'L': []}    # planted feet: dict(t0, t1, p0, p1)
        self.looks = []                     # (t0, t1, target, weight, ramp)
        self.hidden = []                    # (t0, t1)
        self.nolim = []
        self.stiffw = []                    # (t0, t1, k): springs k times stiffer (fast combos keep their snap)
        self.offs = []                      # (t0, t1, t2, t3, vec): root offset ramping in t0..t1, held, out t2..t3
        self.spr = {}                       # per-actor spring overrides: base channel -> (f, zeta)
        self.noise_k = 1.0

    def key(self, t, pose=None, **kw):
        d = {}
        if pose is not None:
            for p in (pose if isinstance(pose, (list, tuple)) else [pose]): d.update(p)
        d.update(kw)
        for c, v in d.items():
            if c == 'pos':
                for cc, vv in zip('xyz', v):
                    if vv is not None: self.k[cc].append((t, vv))
            elif c == 'xz':
                self.k['x'].append((t, v[0])); self.k['z'].append((t, v[1]))
            elif c in self.k: self.k[c].append((t, v))
            else: raise KeyError(c)
        return self

    def hold(self, t0, t1, chans=None):
        """re-key the curves' values at t0 at t1 (a moving hold is done by the noise/springs)"""
        self._holds = getattr(self, '_holds', []) + [(t0, t1, chans)]

    def cut(self, t): self.cuts.append(t)

    def stiff(self, t0, t1, k): self.stiffw.append((t0, t1, k))

    def offset(self, t0, t1, t2, t3, vec): self.offs.append((t0, t1, t2, t3, np.asarray(vec, float)))

    def plant(self, side, t0, t1, p0, p1=None, ease=True):
        self.plants[side].append(dict(t0=t0, t1=t1, p0=np.array(p0, float), p1=np.array(p1 if p1 is not None else p0, float), ease=ease))

    def look(self, t0, t1, target, w=1.0, ramp=0.25):
        self.looks.append((t0, t1, target, w, ramp))

    def hide(self, t0, t1): self.hidden.append((t0, t1))

    # ------------------------------------------------------------ bake channels
    def channels(self, ts):
        C = {}
        cuts_i = [int(np.searchsorted(ts, t)) for t in self.cuts]
        for c in CH:
            keys = self.k[c]
            if c in ('yaw', 'pitch', 'roll') and keys:      # angles: each key takes the turn nearest the previous one
                ks = sorted(keys, key=lambda k: k[0]); out = [ks[0]]
                for t, v in ks[1:]:
                    pv = out[-1][1]
                    while v - pv > 180.0001: v -= 360
                    while pv - v > 180.0001: v += 360
                    out.append((t, v))
                keys = out
            C[c] = curve(keys, ts)
        # holds: nothing to do (keys define them); noise keeps holds alive
        alive = curve(self.k['alive'], ts, 1.0) if self.k['alive'] else np.ones(len(ts))
        fm = np.ones(len(ts))
        for (a, b, kk) in self.stiffw:
            u = np.minimum(ss_((ts - a + 0.06) / 0.06), ss_((b + 0.06 - ts) / 0.06)); fm = np.maximum(fm, 1 + (kk - 1) * u)
        for (t0, t1, t2, t3, vec) in self.offs:
            u = np.where(ts < t1, ss_((ts - t0) / max(1e-3, t1 - t0)), np.where(ts <= t2, 1.0, 1 - ss_((ts - t2) / max(1e-3, t3 - t2))))
            for j, c in enumerate('xyz'): C[c] = C[c] + vec[j] * u
        for i, c in enumerate(CH):
            b = base(c)
            if b in NOISE and NOISE[b] > 0:
                C[c] = C[c] + smooth_noise(ts, self.seed * 100 + i, NOISE[b] * self.noise_k) * alive
            if b in SPR:
                f, z = self.spr.get(b, SPR[b]); C[c] = spring(C[c], f, z, [0] + cuts_i, fm=fm)
            if b in LIM:
                lo, hi = LIM[b]; C[c] = np.clip(C[c], lo, hi)
        C['eyes'] = curve(self.k['eyes'], ts, 0.0)
        return C

    # ------------------------------------------------------------ feet
    def foot_targets(self, side, ts):
        """-> target (n,3) and weight (n,) for one foot"""
        P = sorted(self.plants[side], key=lambda p: p['t0'])
        tgt = np.zeros((len(ts), 3)); w = np.zeros(len(ts))
        for i, t in enumerate(ts):
            for j, p in enumerate(P):
                if p['t0'] <= t <= p['t1']:
                    u = (t - p['t0']) / max(1e-6, p['t1'] - p['t0']); u = (1 - (1 - u) ** 2.2 if p['ease'] == 'out' else ss_(u)) if p['ease'] else u
                    q = p['p0'] + (p['p1'] - p['p0']) * u
                    tgt[i] = [q[0], 0.0, q[1]]; w[i] = 1.0; break
                if j + 1 < len(P) and p['t1'] < t < P[j + 1]['t0']:
                    nx = P[j + 1]; gap = nx['t0'] - p['t1']
                    if any(h0 <= t < h1 for h0, h1 in self.hidden): break          # gone (teleporting): no foot target
                    if gap <= 0.75 and not any(p['t1'] < c <= nx['t0'] for c in self.cuts):   # a step: lift, arc, set down
                        u = (t - p['t1']) / gap; a = p['p1']; b = nx['p0']
                        q = a + (b - a) * ss_(u); dist = np.linalg.norm(b - a)
                        hgt = np.clip(0.05 + 0.22 * dist, 0.05, 0.28) * np.sin(np.pi * u) ** 1.2
                        tgt[i] = [q[0], hgt, q[1]]; w[i] = 1.0; break
                    # airborne: fade the pin out after leaving / in before landing
                    if t - p['t1'] < 0.07:
                        tgt[i] = [p['p1'][0], 0.0, p['p1'][1]]; w[i] = 1 - ss_((t - p['t1']) / 0.07); break
                    if nx['t0'] - t < 0.07:
                        tgt[i] = [nx['p0'][0], 0.0, nx['p0'][1]]; w[i] = 1 - ss_((nx['t0'] - t) / 0.07); break
            else:
                if P and P[-1]['t1'] < t < P[-1]['t1'] + 0.07:
                    p = P[-1]; tgt[i] = [p['p1'][0], 0.0, p['p1'][1]]; w[i] = 1 - ss_((t - p['t1']) / 0.07)
                elif P and P[0]['t0'] - 0.07 < t < P[0]['t0']:
                    p = P[0]; tgt[i] = [p['p0'][0], 0.0, p['p0'][1]]; w[i] = 1 - ss_((p['t0'] - t) / 0.07)
        return tgt, w

    # ------------------------------------------------------------ bake body (no head look yet)
    def bake(self, ts):
        C = self.channels(ts)
        L = local_quats(C); W = world_quats(L)
        P = np.stack([C['x'], C['y'], C['z']], -1)
        # feet: pull the pelvis down if a planted leg can't reach, then two-bone IK
        yawq = ry(C['yaw'])
        tg = {}; wt = {}
        for s in SIDES: tg[s], wt[s] = self.foot_targets(s, ts)
        drop = np.zeros(len(ts))
        for s, sx in (('R', -2), ('L', 2)):
            hip = P + R.qrot(W['pelvis'], np.array([sx * PX, 0, 0]))
            d = tg[s] - hip; dh = np.linalg.norm(d[:, [0, 2]], axis=1)
            need = np.sqrt(np.maximum(0, (2 * L_LEG * 0.985) ** 2 - dh ** 2))       # max hip height above the target
            over = (hip[:, 1] - tg[s][:, 1]) - need
            drop = np.maximum(drop, np.where(wt[s] > 0.98, np.maximum(0, over), 0))
        if drop.any():
            k = int(0.08 * HZ); dr = np.convolve(np.pad(drop, k, mode='edge'), np.hanning(2 * k + 1) / np.hanning(2 * k + 1).sum(), 'same')[k:-k]
            P[:, 1] -= np.maximum(drop, dr)
        for s, sx in (('R', -2), ('L', 2)):
            if not wt[s].any(): continue
            hip = P + R.qrot(W['pelvis'], np.array([sx * PX, 0, 0]))
            pole = R.qrot(mul(yawq, ry((1 if s == 'L' else -1) * C['lt' + s])), np.array([0, 0, 1.0]))
            qt, qs = leg_ik(hip, tg[s], pole)
            m = wt[s] > 0
            W['thigh' + s][m] = R.qslerp(W['thigh' + s][m], qt[m], wt[s][m])
            W['shin' + s][m] = R.qslerp(W['shin' + s][m], qs[m], wt[s][m])
        Q = np.stack([W[b] for b in R.BONES], 1)
        vis = np.ones(len(ts), bool)
        for a, b in self.hidden: vis[(ts >= a) & (ts < b)] = False
        return dict(ts=ts, P=P, Q=Q, C=C, vis=vis, eyes=C['eyes'])


def leg_ik(hip, tgt, pole):
    """two equal bones from hip to the sole target; the knee bends towards `pole` -> thigh, shin world quats"""
    d = tgt - hip; L = np.linalg.norm(d, axis=1, keepdims=True); dn = d / (L + 1e-9)
    Lc = np.minimum(L, 2 * L_LEG * 0.999)
    pp = pole - np.sum(pole * dn, 1, keepdims=True) * dn; pp /= np.linalg.norm(pp, axis=1, keepdims=True) + 1e-9
    h = np.sqrt(np.maximum(0, L_LEG ** 2 - (Lc / 2) ** 2))
    knee = hip + dn * Lc / 2 + pp * h
    sole = hip + dn * Lc
    def frame(a, b, fwd):
        y = (a - b); y /= np.linalg.norm(y, axis=1, keepdims=True) + 1e-9          # bone +Y points from the lower end up
        z = fwd - np.sum(fwd * y, 1, keepdims=True) * y; z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
        x = np.cross(y, z)
        return R.qfrom_mat(np.stack([x, y, z], -1))
    return frame(hip, knee, pp), frame(knee, sole, pp)


def head_look(A, B, ts):
    """turn A's head (and a little of the chest via the head) towards targets: overrides the keyed neck with weight"""
    if not A['actor'].looks: return
    W = np.zeros(len(ts)); tgt = np.zeros((len(ts), 3))
    for (t0, t1, target, w, ramp) in A['actor'].looks:
        m = (ts >= t0 - ramp) & (ts < t1 + ramp)
        if not m.any(): continue
        u = np.minimum(ss_((ts[m] - (t0 - ramp)) / max(ramp, 1e-3)), ss_(((t1 + ramp) - ts[m]) / max(ramp, 1e-3)))
        if isinstance(target, str):
            fkb = R.fk(B['P'][m], B['Q'][m]); p = fkb[target]
        else:
            p = np.broadcast_to(np.asarray(target, float), (m.sum(), 3))
        tgt[m] = np.where((W[m] > 0)[:, None], tgt[m], p); W[m] = np.maximum(W[m], u * w)
    if not W.any(): return
    Q = A['Q']; P = A['P']
    fk = R.fk(P, Q)
    head = fk['head'] + R.qrot(Q[:, R.BI['chest']], np.array([0, 3 * PX, 0]))
    d = tgt - head
    dl = R.qrot(R.qconj(Q[:, R.BI['chest']]), d)
    nt = np.degrees(np.arctan2(dl[:, 0], dl[:, 2])); nb = np.degrees(np.arctan2(-dl[:, 1], np.hypot(dl[:, 0], dl[:, 2])))
    nt = np.clip(nt, -75, 75); nb = np.clip(nb, -40, 45)
    C = A['C']
    nt = C['nt'] * (1 - W) + nt * W; nb = C['nb'] * (1 - W) + nb * W
    cuts = [int(np.searchsorted(ts, t)) for t in A['actor'].cuts]
    nt = spring(nt, 5.0, 0.7, [0] + cuts); nb = spring(nb, 5.0, 0.7, [0] + cuts)
    qh = mul(ry(nt), rx(nb), rz(-C['ns']))
    Q[:, R.BI['head']] = R.qmul(Q[:, R.BI['chest']], qh)


# ---------------------------------------------------------------- collision check (capsules)
def capsules(P, Q):
    f = R.fk(P, Q); qp = Q[:, R.BI['pelvis']]; qc = Q[:, R.BI['chest']]
    caps = {}
    caps['head'] = (f['headC'], f['headC'], 0.26)
    for s, sx in (('a', -2), ('b', 2)):
        caps['torso' + s] = (f['pelvis'] + R.qrot(qp, np.array([sx * PX, 1 * PX, 0])), f['chest'] + R.qrot(qc, np.array([sx * PX, 5 * PX, 0])), 0.125)
    for s in SIDES:
        caps['uarm' + s] = (f['uarm' + s] + R.qrot(Q[:, R.BI['uarm' + s]], np.array([0, 1 * PX, 0])), f['farm' + s], 0.125)
        caps['farm' + s] = (f['farm' + s], f['fist' + s] - R.qrot(Q[:, R.BI['farm' + s]], np.array([0, -1.5 * PX, 0])), 0.125)
        caps['thigh' + s] = (f['thigh' + s], f['shin' + s], 0.125)
        caps['shin' + s] = (f['shin' + s], f['sole' + s] - R.qrot(Q[:, R.BI['shin' + s]], np.array([0, -1.5 * PX, 0])), 0.125)
    return caps


def seg_dist(a0, a1, b0, b1):
    """closest distance between segments, vectorised over rows"""
    d1 = a1 - a0; d2 = b1 - b0; r = a0 - b0
    a = np.sum(d1 * d1, 1); e = np.sum(d2 * d2, 1); f = np.sum(d2 * r, 1); c = np.sum(d1 * r, 1); b = np.sum(d1 * d2, 1)
    den = a * e - b * b
    s = np.where(den > 1e-9, np.clip((b * f - c * e) / np.where(den > 1e-9, den, 1), 0, 1), 0)
    t = np.where(e > 1e-9, (b * s + f) / np.where(e > 1e-9, e, 1), 0)
    s = np.where(t < 0, np.where(a > 1e-9, np.clip(-c / np.where(a > 1e-9, a, 1), 0, 1), 0), s)
    s = np.where(t > 1, np.where(a > 1e-9, np.clip((b - c) / np.where(a > 1e-9, a, 1), 0, 1), 0), s)
    t = np.clip(t, 0, 1)
    pa = a0 + d1 * s[:, None]; pb = b0 + d2 * t[:, None]
    return np.linalg.norm(pa - pb, axis=1), pa, pb


def clearance(A, B):
    """min surface clearance between two baked bodies per sample, and the pair of parts"""
    ca = capsules(A['P'], A['Q']); cb = capsules(B['P'], B['Q'])
    n = len(A['P']); best = np.full(n, 9.0); pair = np.empty(n, object); pt = np.zeros((n, 3))
    for ka, (a0, a1, ra) in ca.items():
        for kb, (b0, b1, rb) in cb.items():
            d, pa, pb = seg_dist(a0, a1, b0, b1); cl = d - ra - rb
            m = cl < best; best[m] = cl[m]; pair[m] = ka + '-' + kb; pt[m] = ((pa + pb) / 2)[m]
    both = A['vis'] & B['vis']; best[~both] = 9.0
    return best, pair, pt


def part_clearance(A, part, B):
    """clearance between one of A's parts (e.g. 'farmR', 'shinL') and all of B"""
    ca = capsules(A['P'], A['Q']); cb = capsules(B['P'], B['Q'])
    a0, a1, ra = ca[part]; best = np.full(len(A['P']), 9.0)
    for kb, (b0, b1, rb) in cb.items():
        d, pa, pb = seg_dist(a0, a1, b0, b1); best = np.minimum(best, d - ra - rb)
    return best


def part_closest(A, part, B):
    """clearance between one of A's parts and all of B, plus the closest points (on A's part, on B)"""
    ca = capsules(A['P'], A['Q']); cb = capsules(B['P'], B['Q'])
    a0, a1, ra = ca[part]; n = len(A['P']); best = np.full(n, 9.0); PA = np.zeros((n, 3)); PB = np.zeros((n, 3))
    for kb, (b0, b1, rb) in cb.items():
        d, pa, pb = seg_dist(a0, a1, b0, b1); cl = d - ra - rb
        m = cl < best; best[m] = cl[m]; PA[m] = pa[m]; PB[m] = pb[m]
    return best, PA, PB
