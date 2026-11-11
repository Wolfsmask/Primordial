"""Forward kinematics that mirror render.html exactly, plus IK for planted feet and the two-handed grip.
Model px space (y down, front -z); a fighter's root sits on the ground at its feet."""
import math
import numpy as np
from scipy.optimize import least_squares

AXE_PX = 1.0                      # axe sprite px per model px


def rzyx(x, y, z):
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx


def M(R=None, t=(0, 0, 0), s=(1, 1, 1)):
    m = np.eye(4)
    m[:3, :3] = (np.eye(3) if R is None else R) @ np.diag(s)
    m[:3, 3] = t
    return m


def rx(a):
    return rzyx(a, 0, 0)


AXE_BASE = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)      # axe x -> -z, y -> +y, z -> +x (columns)


class Fighter:
    def __init__(self, slim, scale):
        self.slim, self.scale = slim, scale
        self.arm_y = 2.5 if slim else 2.0
        self.hand_x = -0.5 if slim else -1.0          # centre of the right arm box
        self.lhand_x = 0.5 if slim else 1.0

    # pose: dict with root [x,y,z,yaw,lean,roll], hips [hx,hy,hz, rx,ry,rz], waist [bx,by,bz],
    # head [3], right_arm/left_arm/right_leg/left_leg [3], *_elbow / *_knee scalars, wrist [3]
    def root(self, p):
        x, y, z, yaw, lean, roll = p['root']
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        tilt = rzyx(-lean, 0, roll)                   # three: plTilt.rotation.set(-lean, 0, roll), order XYZ default
        tilt = rzyx_xyz(-lean, 0, roll)
        sc = self.scale
        return M(Ry, (x, y, z)) @ M(tilt) @ M(None, (0, 0, 0), (-sc / 16, -sc / 16, sc / 16)) @ M(None, (0, -24.016, 0))

    def frames(self, p):
        F = {}
        C = self.root(p)
        h = p['hips']
        F['hips'] = C @ M(rzyx(*h[3:6]), (h[0], 12 + h[1], h[2]))
        for side, sx in (('right', -1.9), ('left', 1.9)):
            F[side + '_leg'] = F['hips'] @ M(rzyx(*p[side + '_leg']), (sx, 0, 0))
        w = p['waist']
        F['waist'] = F['hips'] @ M(rzyx(*w), (0, -6, 0))
        F['top'] = F['waist'] @ M(None, (0, -6, 0))
        F['head'] = F['top'] @ M(rzyx(*p['head']), (0, 0, 0))
        F['right_arm'] = F['top'] @ M(rzyx(*p['right_arm']), (-5, self.arm_y, 0))
        F['left_arm'] = F['top'] @ M(rzyx(*p['left_arm']), (5, self.arm_y, 0))
        F['rh_joint'] = F['right_arm'] @ M(rx(p['right_elbow']), (0, 4, 0))
        F['rh_end'] = F['rh_joint'] @ M(None, (self.hand_x, 6, 0))
        F['axe'] = F['rh_end'] @ M(rzyx(*p.get('wrist', (0, 0, 0))) @ AXE_BASE, (0, 0, 0), (AXE_PX,) * 3)
        F['lh_end'] = F['left_arm'] @ M(rx(p['left_elbow']), (0, 4, 0)) @ M(None, (self.lhand_x, 6, 0))
        for side in ('right', 'left'):
            F[side + '_foot'] = F[side + '_leg'] @ M(rx(p[side + '_knee']), (0, 6, 0)) @ M(None, (0, 6, 0))
        return F


def rzyx_xyz(x, y, z):
    """three.js default Euler order XYZ: matrix Rx Ry Rz"""
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rx @ Ry @ Rz


def pt(Mx, p=(0, 0, 0)):
    return (Mx @ np.array([p[0], p[1], p[2], 1.0]))[:3]


def solve_leg(fighter, pose, side, target_world, knee_hint=0.6):
    """leg rotation (x, z) + knee so the sole lands on target_world; knee bends backward (positive)"""
    def resid(v):
        q = dict(pose); q[side + '_leg'] = [v[0], 0.0, v[1]]; q[side + '_knee'] = v[2]
        f = pt(fighter.frames(q)[side + '_foot'])
        return np.concatenate([(f - target_world) * 20, [0.02 * v[1], 0.0 if v[2] >= 0 else 5 * v[2]]])
    x0 = list(pose[side + '_leg'][:1]) + [pose[side + '_leg'][2], max(0.05, pose[side + '_knee'] or knee_hint)]
    r = least_squares(resid, x0, bounds=([-2.6, -1.2, 0.0], [1.6, 1.2, 2.6]), xtol=1e-6, max_nfev=60)
    pose[side + '_leg'] = [r.x[0], 0.0, r.x[1]]
    pose[side + '_knee'] = r.x[2]
    return pose


def solve_left_hand(fighter, pose, target_world):
    """left arm (x, y, z) + elbow so the left fist closes on target_world (the axe handle)"""
    def resid(v):
        q = dict(pose); q['left_arm'] = [v[0], v[1], v[2]]; q['left_elbow'] = v[3]
        f = pt(fighter.frames(q)['lh_end'])
        return np.concatenate([(f - target_world) * 20, [0.05 * v[2], 0.0 if v[3] <= 0 else 5 * v[3]], (np.asarray(v) - x0) * 0.05])
    x0 = np.array(list(pose['left_arm']) + [min(-0.05, pose['left_elbow'])])
    x0 = np.clip(x0, [-3.6 + 1e-6, -2.4 + 1e-6, -1.6 + 1e-6, -2.6 + 1e-6], [1.8 - 1e-6, 2.4 - 1e-6, 1.6 - 1e-6, -1e-6])
    r = least_squares(resid, x0, bounds=([-3.6, -2.4, -1.6, -2.6], [1.8, 2.4, 1.6, 0.0]), xtol=1e-8, max_nfev=150)
    pose['left_arm'] = list(r.x[:3])
    pose['left_elbow'] = r.x[3]
    return pose


def axe_points(fighter, pose, pts_axe):
    """world positions of points given in axe-sprite space"""
    A = fighter.frames(pose)['axe']
    return [pt(A, (x, y, 0.0)) for x, y in pts_axe]
