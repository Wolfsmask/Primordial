"""Everything that moves, per frame at 60 fps: the camera (smooth monotone splines through keyed values, a gentle
walking bob), the arm, the door, the wolf (turns, stands, trots over, sits, tilts its head), hearts, and the
timeline of sounds.  Times are seconds in the final video; the world starts at T_WORLD (before it, the menu)."""
import math
import numpy as np
from scipy.interpolate import PchipInterpolator

FPS = 60
T_END = 24.0
T_WORLD = 2.4
EYE, SNEAK = 1.62, 1.27
NF = int(round(T_END * FPS))


def track(keys):
    """keys: [(t, value), ...] -> smooth function (monotone cubic: no overshoot, flat at the holds)"""
    ts = np.array([k[0] for k in keys], float); vs = np.array([k[1] for k in keys], float)
    f = PchipInterpolator(ts, vs, extrapolate=False)
    def g(t):
        return float(vs[0] if t <= ts[0] else (vs[-1] if t >= ts[-1] else f(t)))
    return g


def smooth(t, a, b):
    u = min(1.0, max(0.0, (t - a) / (b - a)))
    return u * u * u * (u * (6 * u - 15) + 10)


# ---- the camera: where he stands (z along the path) and where he looks
Z_PATH = track([(T_WORLD, 128.5), (6.5, 128.5), (11.9, 136.25), (13.5, 136.25), (15.3, 139.35), (T_END, 139.35)])
X_PATH = track([(T_WORLD, 128.5), (T_END, 128.5)])
YAW = track([(T_WORLD, 114), (4.5, 66), (6.2, 10), (8.6, 4), (9.6, 30), (10.6, 28), (11.6, 0), (14.0, 0), (14.7, 30), (15.5, 22),
             (16.3, 17), (17.5, 15), (18.7, 3), (T_END, 1)])
PITCH = track([(T_WORLD, 2), (4.5, -3), (6.2, 6), (8.6, 7), (9.6, 16), (10.6, 15), (11.6, 9), (13.4, 9), (14.7, 17), (15.5, 18),
               (16.3, 19), (17.5, 20), (18.7, 21), (19.7, 17), (T_END, 18)])
CROUCH = track([(T_WORLD, 0), (18.9, 0), (19.7, 1), (T_END, 1)])
FOV = track([(T_WORLD, 70), (T_END, 70)])
DOOR = track([(T_WORLD, 0), (12.75, 0), (13.15, 0.82), (13.45, 1.0), (T_END, 1.0)])
SWING = [(12.35, 12.85)]              # arm use-swing (opens the door)
REACH = track([(T_WORLD, 0), (T_END, 0)])   # the hand goes down to him


def walk_dist(t):
    return abs(Z_PATH(t) - Z_PATH(T_WORLD)) + 0.0


def camera(t):
    z = Z_PATH(t); x = X_PATH(t)
    # walking: speed from the path, a soft bob (sideways sway, small dip, a little roll) that fades in and out with speed
    dt = 1 / FPS
    v = abs(Z_PATH(t + dt) - Z_PATH(t - dt)) / (2 * dt)
    amt = min(1.0, v / 1.9) * 0.55
    d = walk_dist(t)
    ph = d * math.pi / 1.15                       # one step every ~1.15 blocks
    bx = math.sin(ph) * 0.045 * amt
    by = -abs(math.cos(ph)) * 0.06 * amt
    roll = math.sin(ph) * 1.4 * amt
    pitch_bob = abs(math.cos(ph - 0.2)) * 1.6 * amt
    yaw = YAW(t); pitch = PITCH(t) + pitch_bob
    eye = EYE - (EYE - SNEAK) * CROUCH(t)
    yr = math.radians(yaw)
    # sideways bob along the camera's right vector
    rx, rz = -math.cos(yr), -math.sin(yr)
    pos = [x + rx * bx, 65 + eye + by, z + rz * bx]
    return dict(pos=pos, yaw=yaw, pitch=pitch, roll=roll, fov=FOV(t)), dict(ph=ph, amt=amt)


def arm(t, bob):
    ph, amt = bob['ph'], bob['amt']
    a = dict(bx=math.sin(ph) * 0.03 * amt, by=-abs(math.cos(ph)) * 0.035 * amt, bz=0.0)
    sw = 0.0
    for a0, a1 in SWING:
        if a0 <= t <= a1:
            u = (t - a0) / (a1 - a0)
            sw = math.sin(math.pi * u) ** 1.5
    a['swing'] = sw
    # idle breathing
    a['by'] += math.sin(t * 1.7) * 0.006
    r = REACH(t)
    a['bx'] += -0.12 * r; a['by'] += -0.02 * r; a['bz'] += -0.06 * r
    a['swing'] += 0.45 * r
    # lowered out of view while crouched before reaching
    a['drop'] = 0.10 * CROUCH(t) * (1 - r)
    return a


# ---- the wolf
W0 = (127.55, 141.65)                          # sitting where the bed is in front of him, facing it
W1 = (128.5, 141.0)                         # in front of us
YAW_BED = -60.0
YAW_US = math.degrees(math.atan2(128.5 - W1[0], 139.35 - W1[1]))       # facing the camera from W1
YAW_US0 = math.degrees(math.atan2(128.5 - W0[0], 139.35 - W0[1]))
YAW_US0 = YAW_US0 - 360 if YAW_US0 - YAW_BED > 180 else YAW_US0         # turn the short way round
WX = track([(0, W0[0]), (17.55, W0[0]), (18.65, W1[0]), (T_END, W1[0])])
WZ = track([(0, W0[1]), (17.55, W0[1]), (18.65, W1[1]), (T_END, W1[1])])
YAW_US_U = YAW_US0 + ((YAW_US - YAW_US0 + 180) % 360 - 180)
WYAW = track([(0, YAW_BED), (16.15, YAW_BED), (17.05, YAW_US0), (18.6, YAW_US_U), (T_END, YAW_US_U)])
SIT = track([(0, 1), (17.05, 1), (17.5, 0), (18.6, 0), (19.1, 1), (T_END, 1)])
HEAD_YAW = track([(0, 0), (15.75, 0), (16.2, -62), (16.55, -66), (17.05, 0), (T_END, 0)])
HEAD_PITCH = track([(0, 12), (15.6, 12), (16.1, 2), (17.1, -6), (18.6, -10), (19.3, -26), (T_END, -27)])
HEAD_ROLL = track([(0, 0), (19.5, 0), (20.0, 16), (20.9, 14), (21.6, -6), (22.2, -4), (T_END, -4)])
EARS = track([(0, 0.3), (15.55, 0.3), (15.75, 1.0), (T_END, 1.0)])
TAIL_UP = track([(0, 0.0), (16.0, 0.0), (16.6, 0.7), (17.3, 1.0), (T_END, 1.0)])
WAG = track([(0, 0.0), (16.2, 0.0), (16.8, 0.55), (T_END, 0.55)])


def wolf(t):
    x, z = WX(t), WZ(t)
    dt = 1 / FPS
    v = math.hypot(WX(t + dt) - WX(t - dt), WZ(t + dt) - WZ(t - dt)) / (2 * dt)
    turning = abs(WYAW(t + dt) - WYAW(t - dt)) / (2 * dt)
    walk_amt = min(1.0, v / 0.8 + turning / 260 * (1 - SIT(t) * 0.7))
    # the gait: phase from distance (and from turning on the spot)
    phase = (walk_dist_wolf(t)) * 2 * math.pi / 0.55
    wag_phase = t * 2 * math.pi * 3.2
    return dict(x=x, y=65.0, z=z, yaw=WYAW(t), sit=SIT(t), phase=phase, walkAmt=walk_amt, headYaw=HEAD_YAW(t),
                headPitch=HEAD_PITCH(t), headRoll=HEAD_ROLL(t), wag=WAG(t), wagPhase=wag_phase, tailUp=TAIL_UP(t), ears=EARS(t))


_WD = None


def walk_dist_wolf(t):
    global _WD
    if _WD is None:
        ts = np.arange(0, T_END + 0.1, 1 / FPS)
        d = np.zeros(len(ts))
        for i in range(1, len(ts)):
            d[i] = d[i - 1] + math.hypot(WX(ts[i]) - WX(ts[i - 1]), WZ(ts[i]) - WZ(ts[i - 1])) + abs(WYAW(ts[i]) - WYAW(ts[i - 1])) / 360 * 0.9
        _WD = (ts, d)
    ts, d = _WD
    return float(np.interp(t, ts, d))


HEART_T = [19.75, 20.05, 20.35, 20.7, 21.1]


def hearts(t):
    out = []
    rng = np.random.default_rng(4)
    for i, t0 in enumerate(HEART_T):
        ox, oz = rng.uniform(-0.3, 0.3, 2)
        if t0 <= t < t0 + 1.1:
            u = (t - t0) / 1.1
            out.append([W1[0] + ox, 66.15 + 0.5 * u, W1[1] + oz - 0.15, 0.2])
    return out


def frame(k):
    t = k / FPS
    cam, bob = camera(t)
    v = dict(cam, time=t, door=DOOR(t), arm=arm(t, bob), wolf=wolf(t), hearts=hearts(t))
    return v


# ---- sound cues (seconds)
def footsteps():
    """a step each half bob cycle while walking; on the path stones from z 130"""
    out = []
    ts = np.arange(T_WORLD, T_END, 1 / FPS)
    last = None
    for t in ts:
        _, b = camera(t)
        n = math.floor(b['ph'] / math.pi + 0.5)
        if last is not None and n != last and b['amt'] > 0.12:
            out.append((float(t), 'stone' if Z_PATH(t) > 129.6 and Z_PATH(t) < 138 else ('wood' if Z_PATH(t) >= 138 else 'grass'), b['amt']))
        last = n
    return out


def wolf_steps():
    out = []
    last = None
    for t in np.arange(16.0, 19.2, 1 / FPS):
        n = math.floor(walk_dist_wolf(t) / 0.275)
        if last is not None and n != last:
            out.append(float(t))
        last = n
    return out


if __name__ == '__main__':
    for k in (0, int(T_WORLD * 60), 500, 800, 1000, 1100, 1250, 1400):
        v = frame(k)
        print(k / 60, [round(x, 2) for x in v['pos']], round(v['yaw'], 1), round(v['pitch'], 1), 'door', round(v['door'], 2),
              'wolf', {q: round(v['wolf'][q], 2) for q in ('x', 'z', 'yaw', 'sit', 'walkAmt')})
    print('steps', [(round(a, 2), b) for a, b, c in footsteps()])
    print('wolf steps', [round(x, 2) for x in wolf_steps()])
