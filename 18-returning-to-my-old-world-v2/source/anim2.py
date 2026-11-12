"""v2 timeline, 60 fps: shots with their own smooth cameras (monotone splines, a breath of hand-held drift on the close
ones), crossfades between them, the dog and the player posed per frame, the typography cue sheet and the sound cues.
Camera yaw is the game's (0 = +z/south, 90 = west, 180 = north, -90 = east; pitch + = down).  Entity yaw is the
model's turn about y (0 = facing +z, 90 = facing +x)."""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

FPS = 60
T_END = 30.0
T_WORLD = 2.3
NF = int(round(T_END * FPS))
EYE, SNEAK = 1.62, 1.27


def track(keys):
    ts = np.array([k[0] for k in keys], float); vs = np.array([k[1] for k in keys], float)
    if len(ts) == 1:
        return lambda t: float(vs[0])
    f = PchipInterpolator(ts, vs, extrapolate=False)
    return lambda t: float(vs[0] if t <= ts[0] else (vs[-1] if t >= ts[-1] else f(t)))


def smooth(t, a, b):
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * u * (u * (6 * u - 15) + 10)


def drift(t, amp, seed):
    """slow hand-held breathing: a few incommensurate sines"""
    r = np.random.default_rng(seed); ph = r.random(6) * 6.28
    return amp * (0.5 * math.sin(t * 0.61 + ph[0]) + 0.3 * math.sin(t * 1.13 + ph[1]) + 0.2 * math.sin(t * 1.97 + ph[2]))


class Shot:
    def __init__(self, t0, t1, pos, yaw, pitch, fov=62, extra=None, dof=None, hand=0.0, seed=0):
        self.t0, self.t1 = t0, t1
        self.px, self.py, self.pz = (track([(t, p[i]) for t, p in pos]) for i in range(3))
        self.yaw = track(yaw); self.pitch = track(pitch)
        self.fov = fov if callable(fov) else (lambda t, f=fov: f)
        self.extra = extra or (lambda t: {})
        self.dof = dof; self.hand = hand; self.seed = seed

    def view(self, t):
        h = self.hand
        v = dict(pos=[self.px(t) + drift(t, 0.03 * h, self.seed), self.py(t) + drift(t, 0.02 * h, self.seed + 1), self.pz(t) + drift(t, 0.03 * h, self.seed + 2)],
                 yaw=self.yaw(t) + drift(t, 0.6 * h, self.seed + 3), pitch=self.pitch(t) + drift(t, 0.4 * h, self.seed + 4),
                 roll=drift(t, 0.35 * h, self.seed + 5), fov=self.fov(t), time=t)
        if self.dof:
            v['dof'] = self.dof(t) if callable(self.dof) else self.dof
        v.update(self.extra(t))
        return v


# ---- the dog: where he sits, how he turns, stands, trots to the doorway and sits again
DOG0 = (200.3, 191.4)
DOG1 = (198.6, 191.5)
PL_DOOR = (196.75, 191.5)
D_X = track([(0, DOG0[0]), (18.0, DOG0[0]), (19.2, DOG1[0]), (T_END, DOG1[0])])
D_Z = track([(0, DOG0[1]), (18.0, DOG0[1]), (19.2, DOG1[1]), (T_END, DOG1[1])])
D_YAW = track([(0, 90), (17.35, 90), (18.1, -90), (T_END, -90)])           # turns round to face the door
D_SIT = track([(0, 1), (16.9, 1), (17.35, 0), (19.2, 0), (19.65, 1), (T_END, 1)])
D_HYAW = track([(0, 0), (14.75, 0), (15.35, -58), (15.8, -66), (16.8, -60), (17.4, 0), (T_END, 0)])
D_HPITCH = track([(0, 10), (14.6, 10), (15.2, 2), (17.0, -2), (19.2, -4), (19.9, -24), (T_END, -27)])
D_HROLL = track([(0, 0), (21.4, 0), (21.9, 24), (23.0, 20), (24.0, -8), (25.0, -6), (T_END, -6)])
D_EARS = track([(0, 0.25), (14.55, 0.25), (14.75, 1.0), (T_END, 1.0)])
D_TAIL = track([(0, 0.1), (15.6, 0.1), (16.6, 0.8), (18.0, 1.0), (T_END, 1.0)])
D_WAG = track([(0, 0.0), (16.0, 0.0), (17.0, 0.5), (T_END, 0.55)])
_WD = None


def dog_dist(t):
    global _WD
    if _WD is None:
        ts = np.arange(0, T_END + 0.1, 1 / FPS); d = np.zeros(len(ts))
        for i in range(1, len(ts)):
            d[i] = d[i - 1] + math.hypot(D_X(ts[i]) - D_X(ts[i - 1]), D_Z(ts[i]) - D_Z(ts[i - 1])) + abs(D_YAW(ts[i]) - D_YAW(ts[i - 1])) / 360 * 0.8
        _WD = (ts, d)
    return float(np.interp(t, *_WD))


def dog(t):
    dt = 1 / FPS
    v = math.hypot(D_X(t + dt) - D_X(t - dt), D_Z(t + dt) - D_Z(t - dt)) / (2 * dt)
    turn = abs(D_YAW(t + dt) - D_YAW(t - dt)) / (2 * dt)
    walk = min(1.0, v / 1.2 + turn / 300)
    return dict(x=D_X(t), y=65.0, z=D_Z(t), yaw=D_YAW(t), sit=D_SIT(t), phase=dog_dist(t) * 2 * math.pi / 0.5, walkAmt=walk,
                headYaw=D_HYAW(t), headPitch=D_HPITCH(t), headRoll=D_HROLL(t), wag=D_WAG(t), wagPhase=t * 2 * math.pi * 3.0,
                tailUp=D_TAIL(t), ears=D_EARS(t))


# ---- the player (seen from inside the house): steps into the doorway, crouches when the dog arrives, reaches out
P_X = track([(0, 195.2), (17.0, 195.2), (17.9, PL_DOOR[0]), (T_END, PL_DOOR[0])])
P_SNEAK = track([(0, 0), (19.3, 0), (19.9, 1), (T_END, 1)])
P_REACH = track([(0, 0), (22.2, 0), (23.0, 0.65), (24.6, 0.55), (T_END, 0.55)])
P_HP = track([(0, 8), (19.1, 14), (19.9, 30), (T_END, 32)])


def player(t):
    dt = 1 / FPS
    v = abs(P_X(t + dt) - P_X(t - dt)) / (2 * dt)
    return dict(x=P_X(t), y=65.0, z=PL_DOOR[1], yaw=90, phase=(P_X(t) - 195.2) * 2 * math.pi / 1.6, walkAmt=min(1, v / 1.0),
                sneak=P_SNEAK(t), headPitch=P_HP(t), reach=P_REACH(t))


def hearts(t):
    out = []
    rng = np.random.default_rng(6)
    for i, t0 in enumerate((22.6, 22.95, 23.3, 23.75, 24.25)):
        ox, oz = rng.uniform(-0.3, 0.3, 2)
        if t0 <= t < t0 + 1.2:
            u = (t - t0) / 1.2
            out.append([DOG1[0] - 0.25 + ox * 0.5, 66.0 + 0.55 * u, DOG1[1] + oz * 0.8, 0.13])
    return out


# ---- first person: walk up the path, glance at the sign, stop at the door, it swings in
FP_X = track([(7.3, 180.6), (8.0, 181.0), (12.2, 193.3), (12.6, 193.4), (13.9, 194.35), (14.4, 194.45)])
DOOR = track([(0, 0), (12.55, 0), (13.15, 0.88), (13.5, 1.0), (T_END, 1.0)])


def fp_extra(t):
    dt = 1 / FPS
    v = abs(FP_X(t + dt) - FP_X(t - dt)) / (2 * dt)
    amt = min(1.0, v / 2.4) * 0.5
    ph = (FP_X(t) - 180.6) * math.pi / 1.2
    return dict(bob=(math.sin(ph) * 0.04 * amt, -abs(math.cos(ph)) * 0.05 * amt, math.sin(ph) * 1.1 * amt), door=DOOR(t), wolf=dog(t))


def fp_shot():
    s = Shot(7.05, 14.55, [(7.3, (180.6, 65 + EYE, 191.5)), (14.4, (180.6, 65 + EYE, 191.5))],
             [(7.3, -96), (8.6, -92), (9.8, -112), (10.7, -114), (11.6, -92), (13.0, -90), (14.4, -91)],
             [(7.3, 4), (9.8, 10), (10.7, 9), (11.6, 7), (13.0, 9), (14.4, 12)], fov=66, seed=10)
    base_view = s.view
    def view(t):
        v = base_view(t); e = fp_extra(t)
        bx, by, roll = e.pop('bob')
        yr = math.radians(v['yaw'])
        v['pos'] = [FP_X(t) + -math.cos(yr) * bx, 65 + EYE + by, 191.5 + -math.sin(yr) * bx]
        v['roll'] = roll; v.update(e)
        return v
    s.view = view
    return s


SHOTS = [
    # the world from the air at sunset: over the house, across the lake to the cliffs, the falls, the floating islands
    Shot(2.3, 7.55, [(2.3, (207, 97, 197)), (7.55, (191, 90.5, 193))], [(2.3, 98), (7.55, 94)], [(2.3, 13), (7.55, 9)], fov=62,
         extra=lambda t: dict(fogNear=110, fogFar=200, door=0, wolf=dog(t)), seed=1),
    fp_shot(),
    # close and low on the dog, waiting by the bed; the door's sound reaches him
    Shot(14.3, 17.95, [(14.3, (201.78, 65.6, 192.72)), (17.95, (201.62, 65.72, 192.55))], [(14.3, 141), (17.95, 138)], [(14.3, -3), (17.95, -1)],
         fov=50, extra=lambda t: dict(door=1, wolf=dog(t)), dof=lambda t: dict(focus=1.65, aperture=7, maxR=11), hand=1.0, seed=2),
    # from inside: you in the doorway against the sunset; he gets up and runs to you; you kneel
    Shot(17.75, 21.45, [(17.75, (201.45, 67.3, 191.5)), (21.45, (201.15, 67.2, 191.5))], [(17.75, 90), (21.45, 90)], [(17.75, 11), (21.45, 12)],
         fov=60, extra=lambda t: dict(door=1, wolf=dog(t), player=player(t)), hand=0.6, seed=3),
    # low and close: he sits, looks up at you, tilts his head; hearts
    # his view, kneeling: the dog looks up into the camera and tilts his head; hearts
    Shot(21.2, 26.2, [(21.2, (196.9, 66.27, 191.55)), (26.2, (197.0, 66.25, 191.52))], [(21.2, -91), (26.2, -90)], [(21.2, 22), (26.2, 24)],
         fov=56, extra=lambda t: dict(door=1, wolf=dog(t), hearts=hearts(t)),
         dof=lambda t: dict(focus=1.45, aperture=5.0, maxR=9), hand=0.8, seed=4),
    # outside at dusk: the door open, warm light inside, the stars coming out; the camera rises away
    Shot(25.4, T_END, [(25.4, (185.5, 67.2, 196.2)), (T_END, (180.5, 71.5, 199.0))], [(25.4, -112), (T_END, -114)], [(25.4, 2), (T_END, 9)],
         fov=60, extra=lambda t: dict(door=1, wolf=dog(t), player=player(t), dusk=0.35 + 0.45 * smooth(t, 25.4, T_END), sunElev=3 - 4 * smooth(t, 25.4, T_END)), seed=5),
]


def weights(t):
    """which shots are on screen at t, and how much (crossfades where shots overlap)"""
    act = [(i, s) for i, s in enumerate(SHOTS) if s.t0 <= t < s.t1]
    if len(act) <= 1:
        return [(i, 1.0) for i, s in act]
    (i0, a), (i1, b) = act[0], act[1]
    w = smooth(t, b.t0, a.t1)
    return [(i0, 1 - w), (i1, w)]


def views(k):
    t = k / FPS
    return [(SHOTS[i].view(t), w) for i, w in weights(t)]


# ---- typography: phrases of big serif words, each word landing on a note
PHRASES = [
    dict(lines=[['TEN'], ['YEARS']], times=[[2.85], [3.35]], end=4.6, y=0.30, size=230),
    dict(lines=[['SINCE', 'I'], ['LAST', 'PLAYED']], times=[[4.8, 5.1], [5.5, 5.9]], end=7.0, y=0.30, size=150),
    dict(lines=[['THE', 'GRASS'], ['GREW', 'OVER'], ['THE', 'PATH']], times=[[8.1, 8.4], [8.95, 9.3], [9.85, 10.15]], end=11.4, y=0.24, size=130),
    dict(lines=[['BUT', 'HE']], times=[[13.2, 13.55]], end=14.55, y=0.30, size=170),
    dict(lines=[['WAS', 'STILL'], ['HERE']], times=[[14.75, 15.1], [15.75]], end=17.1, y=0.22, size=150),
    dict(lines=[['HE', 'NEVER'], ['STOPPED'], ['WAITING']], times=[[22.5, 22.85], [23.45], [24.15]], end=25.9, y=0.18, size=140),
]
OUTRO = dict(text='my first world  ·  2016 – 2026', t0=27.0, t1=29.6)


# ---- sound cues
def fp_steps():
    out, last = [], None
    for t in np.arange(7.3, 14.5, 1 / FPS):
        n = math.floor((FP_X(t) - 180.6) / 1.2 + 0.5)
        if last is not None and n != last:
            out.append((float(t), 'stone' if 184 < FP_X(t) < 196 else 'grass'))
        last = n
    return out


def dog_steps():
    out, last = [], None
    for t in np.arange(17.6, 20.5, 1 / FPS):
        n = math.floor(dog_dist(t) / 0.25)
        if last is not None and n != last:
            out.append(float(t))
        last = n
    return out


if __name__ == '__main__':
    for t in (2.3, 5, 7.3, 10, 13.3, 15.5, 18.5, 19.5, 21.5, 23, 27, 29.9):
        vs = views(int(t * FPS))
        print(t, [(round(w, 2), [round(x, 2) for x in v['pos']], round(v['yaw'], 1)) for v, w in vs])
    print('steps', [(round(a, 2), b) for a, b in fp_steps()])
    print('dog steps', [round(x, 2) for x in dog_steps()])
