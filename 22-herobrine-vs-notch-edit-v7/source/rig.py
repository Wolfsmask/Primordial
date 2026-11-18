"""The blocky fighter rig shared with web/edit2.html: 11 bones with world quaternions, pelvis world position.
Model faces +Z, up +Y, the character's left is +X.  Lengths in px (1/16 m)."""
import numpy as np

PX = 1.0 / 16
BONES = ['pelvis', 'chest', 'head', 'uarmR', 'farmR', 'uarmL', 'farmL', 'thighR', 'shinR', 'thighL', 'shinL']
BI = {n: i for i, n in enumerate(BONES)}
PARENT_OFF = {'chest': ('pelvis', (0, 6, 0)), 'head': ('chest', (0, 6, 0)), 'uarmR': ('chest', (-6, 4, 0)), 'farmR': ('uarmR', (0, -4, 0)),
              'uarmL': ('chest', (6, 4, 0)), 'farmL': ('uarmL', (0, -4, 0)), 'thighR': ('pelvis', (-2, 0, 0)), 'shinR': ('thighR', (0, -6, 0)),
              'thighL': ('pelvis', (2, 0, 0)), 'shinL': ('thighL', (0, -6, 0))}
UPPER_LEN = {'uarm': 4 * PX, 'farm': 6 * PX, 'thigh': 6 * PX, 'shin': 6 * PX}


# ---------------------------------------------------------------- quaternions [x, y, z, w]
def qmul(a, b):
    ax, ay, az, aw = np.moveaxis(np.asarray(a, float), -1, 0); bx, by, bz, bw = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def qconj(q):
    q = np.asarray(q, float).copy(); q[..., :3] *= -1; return q


def qrot(q, v):
    q = np.asarray(q, float); v = np.asarray(v, float)
    u = q[..., :3]; w = q[..., 3:4]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def qaxis(axis, ang):
    axis = np.asarray(axis, float); axis = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-12)
    ang = np.asarray(ang, float)[..., None]
    return np.concatenate([axis * np.sin(ang / 2), np.cos(ang / 2)], -1)


def qfrom_mat(m):
    """rotation matrices (...,3,3) whose COLUMNS are the rotated x, y, z axes -> quaternions"""
    m = np.asarray(m, float); sh = m.shape[:-2]; m = m.reshape(-1, 3, 3); out = np.zeros((len(m), 4))
    for i, r in enumerate(m):
        t = r[0, 0] + r[1, 1] + r[2, 2]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2; out[i] = [(r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2; out[i] = [0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s]
        elif r[1, 1] > r[2, 2]:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2; out[i] = [(r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s]
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2; out[i] = [(r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s]
    return out.reshape(*sh, 4)


def qto_mat(q):
    q = np.asarray(q, float)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def qslerp(a, b, t):
    a = np.asarray(a, float); b = np.asarray(b, float); t = np.asarray(t, float)
    d = np.sum(a * b, -1, keepdims=True); b = np.where(d < 0, -b, b); d = np.abs(d)
    t = t[..., None] if t.ndim == a.ndim - 1 else t
    lin = d > 0.9995
    th = np.arccos(np.clip(d, -1, 1)); s = np.sin(th) + 1e-12
    wa = np.where(lin, 1 - t, np.sin((1 - t) * th) / s); wb = np.where(lin, t, np.sin(t * th) / s)
    r = wa * a + wb * b
    return r / np.linalg.norm(r, axis=-1, keepdims=True)


def frame_from(xv, yv):
    """orthonormal frame (columns x, y, z) with y exactly along yv and x as close to xv as possible"""
    y = yv / (np.linalg.norm(yv, axis=-1, keepdims=True) + 1e-12)
    x = xv - np.sum(xv * y, -1, keepdims=True) * y; x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)
    z = np.cross(x, y)
    return np.stack([x, y, z], -1)


# ---------------------------------------------------------------- forward kinematics
def fk(P, Q):
    """P (...,3) pelvis world position (m); Q (...,11,4) world quaternions -> dict bone -> world position, plus end points"""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    pos = {'pelvis': P}
    for n in BONES[1:]:
        par, off = PARENT_OFF[n]
        pos[n] = pos[par] + qrot(Q[..., BI[par], :], np.array(off, float) * PX)
    pos['fistR'] = pos['farmR'] + qrot(Q[..., BI['farmR'], :], np.array([0, -6, 0]) * PX)
    pos['fistL'] = pos['farmL'] + qrot(Q[..., BI['farmL'], :], np.array([0, -6, 0]) * PX)
    pos['soleR'] = pos['shinR'] + qrot(Q[..., BI['shinR'], :], np.array([0, -6, 0]) * PX)
    pos['soleL'] = pos['shinL'] + qrot(Q[..., BI['shinL'], :], np.array([0, -6, 0]) * PX)
    pos['headC'] = pos['head'] + qrot(Q[..., BI['head'], :], np.array([0, 4, 0]) * PX)
    pos['chestC'] = pos['chest'] + qrot(Q[..., BI['chest'], :], np.array([0, 1, 0]) * PX)
    pos['belly'] = pos['pelvis'] + qrot(Q[..., BI['pelvis'], :], np.array([0, 3, 0]) * PX)
    pos['farmRm'] = pos['farmR'] + qrot(Q[..., BI['farmR'], :], np.array([0, -3, 0]) * PX)
    pos['farmLm'] = pos['farmL'] + qrot(Q[..., BI['farmL'], :], np.array([0, -3, 0]) * PX)
    return pos


def ident(n=None):
    q = np.zeros((11, 4)); q[:, 3] = 1
    return q if n is None else np.repeat(q[None], n, 0)


def pose_dict(P, Q, **kw):
    d = dict(P=[round(float(v), 5) for v in P], Q=[[round(float(v), 5) for v in q] for q in Q]); d.update(kw); return d
