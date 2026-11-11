"""Pose authoring for the fist fight: arm angles are found from where the fists should be (local to the fighter:
forward, up, right in metres from the root), once, at authoring time. Everything else is plain joint angles."""
import math
import numpy as np
from scipy.optimize import least_squares
import rig

S_SCALE = 0.9375
N_SCALE = 0.9375 * 1.05
STATIC = rig.Fighter(slim=True, scale=S_SCALE)
NETH = rig.Fighter(slim=False, scale=N_SCALE)


def base(x=0.0, z=0.0, yaw=0.0):
    return dict(root=[x, 0.0, z, yaw, 0.0, 0.0], hips=[0, 0, 0, 0, 0, 0], waist=[0, 0, 0], head=[0, 0, 0],
                right_arm=[0, 0, 0], left_arm=[0, 0, 0], right_elbow=-0.1, left_elbow=-0.1,
                right_leg=[0, 0, 0], left_leg=[0, 0, 0], right_knee=0.05, left_knee=0.05, wrist=[0, 0, 0])


def axes(yaw_deg):
    y = math.radians(yaw_deg)
    return np.array([-math.sin(y), 0, -math.cos(y)]), np.array([0, 1.0, 0]), np.array([math.cos(y), 0, -math.sin(y)])


def to_world(p, local):
    f, u, r = axes(p['root'][3])
    return np.array(p['root'][:3]) + f * local[0] + u * local[1] + r * local[2]


def to_local(p, w):
    f, u, r = axes(p['root'][3])
    d = np.asarray(w) - np.array(p['root'][:3])
    return np.array([d @ f, d @ u, d @ r])


def hand(fighter, p, side):
    F = fighter.frames(p)
    return rig.pt(F['rh_end'] if side == 'right' else F['lh_end'])


def elbow_pt(fighter, p, side):
    F = fighter.frames(p)
    return rig.pt(F[side + '_arm'], (0, 4, 0))


def reach(fighter, p, side, target_w, prior=None, w_prior=0.25, elbow_hint=None, w_elbow=8.0):
    """arm (x, y, z) + elbow so this fist lands on target_w; stays near the prior arm angles"""
    key = side + '_arm'
    pr = np.array(list(prior if prior is not None else p[key]) + [p[side + '_elbow']])
    def res(v):
        q = dict(p); q[key] = list(v[:3]); q[side + '_elbow'] = v[3]
        h = hand(fighter, q, side)
        e = [] if elbow_hint is None else list((elbow_pt(fighter, q, side) - elbow_hint) * w_elbow)
        return np.concatenate([(h - target_w) * 30, (v - pr) * w_prior, [0.15 * v[1]], e])
    lo = [-3.4, -1.2, -1.6, -2.6]; hi = [1.6, 1.2, 1.6, 0.0]
    r = least_squares(res, np.clip(pr, np.array(lo) + 1e-6, np.array(hi) - 1e-6), bounds=(lo, hi), xtol=1e-10, max_nfev=400)
    p[key] = list(r.x[:3]); p[side + '_elbow'] = float(r.x[3])
    return float(np.linalg.norm(hand(fighter, p, side) - target_w))


def head_pt(fighter, p, off=(0, -4, 0)):
    return rig.pt(fighter.frames(p)['head'], off)


def chest_pt(fighter, p, off=(0, 3, -2.2)):
    return rig.pt(fighter.frames(p)['top'], off)


if __name__ == '__main__':
    for name, f in (('static', STATIC), ('neth', NETH)):
        p = base()
        F = f.frames(p)
        print(name, 'head', to_local(p, head_pt(f, p)).round(3), 'chin', to_local(p, head_pt(f, p, (0, -1, -4))).round(3),
              'shoulders', to_local(p, rig.pt(F['right_arm'])).round(3), to_local(p, rig.pt(F['left_arm'])).round(3),
              'chest', to_local(p, chest_pt(f, p)).round(3), 'hands down', to_local(p, hand(f, p, 'right')).round(3))
