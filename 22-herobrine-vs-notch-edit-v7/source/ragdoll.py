"""MuJoCo ragdoll of the blocky rig (same bones, box limbs, joint limits, human-like masses) and loose block debris.
simulate(P0, Q0, P1, Q1, dt, dur, push=..., spin=..., tone=...) -> (ts, P, Q) at fight.HZ, bodies in rig bone frames."""
import numpy as np
import mujoco
import rig as R

PX = R.PX
# body: parent, offset from parent (px), box half-size (px), box centre (px), mass (kg), joint
BODY = [('pelvis', None, (0, 0, 0), (4, 3, 2), (0, 3, 0), 12.0, 'free'),
        ('chest', 'pelvis', (0, 6, 0), (4, 3, 2), (0, 3, 0), 18.0, ('ball', 70)),
        ('head', 'chest', (0, 6, 0), (4, 4, 4), (0, 4, 0), 5.0, ('ball', 80)),
        ('uarmR', 'chest', (-6, 4, 0), (2, 3, 2), (0, -1, 0), 2.2, ('ball', 178)),
        ('farmR', 'uarmR', (0, -4, 0), (2, 3, 2), (0, -3, 0), 1.6, ('hinge', -150, 0)),
        ('uarmL', 'chest', (6, 4, 0), (2, 3, 2), (0, -1, 0), 2.2, ('ball', 178)),
        ('farmL', 'uarmL', (0, -4, 0), (2, 3, 2), (0, -3, 0), 1.6, ('hinge', -150, 0)),
        ('thighR', 'pelvis', (-2, 0, 0), (2, 3, 2), (0, -3, 0), 7.0, ('ball', 150)),
        ('shinR', 'thighR', (0, -6, 0), (2, 3, 2), (0, -3, 0), 4.0, ('hinge', 0, 150)),
        ('thighL', 'pelvis', (2, 0, 0), (2, 3, 2), (0, -3, 0), 7.0, ('ball', 150)),
        ('shinL', 'thighL', (0, -6, 0), (2, 3, 2), (0, -3, 0), 4.0, ('hinge', 0, 150))]


def _f(v, s=PX):
    return ' '.join('%.5f' % (x * s) for x in v)


def model_xml(obstacles=(), debris=0, tone=1.0):
    kids = {b[0]: [] for b in BODY}
    for b in BODY:
        if b[1]: kids[b[1]].append(b)
    def body(b, ind):
        name, par, off, half, cen, mass, jt = b
        s = ind + '<body name="%s" pos="%s">\n' % (name, _f(off))
        if jt == 'free':
            s += ind + '  <freejoint name="root"/>\n'
        elif jt[0] == 'ball':
            s += ind + '  <joint name="j_%s" type="ball" range="0 %d" damping="%.2f" stiffness="%.2f" armature="0.01"/>\n' % (name, jt[1], 1.2 * tone + 0.3, 6.0 * tone)
        else:
            s += ind + '  <joint name="j_%s" type="hinge" axis="1 0 0" range="%d %d" damping="%.2f" stiffness="%.2f" armature="0.01"/>\n' % (name, jt[1], jt[2], 0.6 * tone + 0.2, 2.5 * tone)
        s += ind + '  <geom type="box" size="%s" pos="%s" mass="%.2f" friction="0.85 0.02 0.002" condim="3" contype="1" conaffinity="2"/>\n' % (_f(half), _f(cen), mass)
        for k in kids[name]: s += body(k, ind + '  ')
        return s + ind + '</body>\n'
    obs = ''.join('<geom type="box" size="%s" pos="%s" friction="0.8 0.02 0.002" contype="2" conaffinity="1"/>\n' % (' '.join('%.3f' % v for v in o[3:6]), ' '.join('%.3f' % v for v in o[0:3])) for o in obstacles)
    deb = ''.join('<body name="d%d" pos="0 %d 0"><freejoint/><geom type="box" size="0.25 0.25 0.25" mass="18" friction="0.7 0.02 0.002"/></body>\n' % (i, -50 - i) for i in range(debris))
    return f'''<mujoco>
  <option timestep="0.0005" gravity="0 -9.81 0" iterations="50" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="80 80 0.1" zaxis="0 1 0" friction="0.9 0.02 0.002" contype="2" conaffinity="1"/>
    {obs}
{body(BODY[0], '    ')}
    {deb}
  </worldbody>
</mujoco>'''


def _qpos(m, P, Q):
    """rig pose -> qpos for the ragdoll's joints"""
    qp = np.zeros(m.nq)
    Qd = {n: Q[R.BI[n]] for n in R.BONES}
    for b in BODY:
        name, par, off, half, cen, mass, jt = b
        if jt == 'free':
            j = m.joint('root'); a = m.jnt_qposadr[j.id]
            qp[a:a + 3] = P; q = Qd['pelvis']; qp[a + 3:a + 7] = [q[3], q[0], q[1], q[2]]
            continue
        rel = R.qmul(R.qconj(Qd[par]), Qd[name])
        j = m.joint('j_' + name); a = m.jnt_qposadr[j.id]
        if jt[0] == 'ball':
            if rel[3] < 0: rel = -rel
            ang = 2 * np.arccos(np.clip(rel[3], -1, 1)); lim = np.radians(jt[1] - 3)
            if ang > lim:                       # pull the start pose inside the joint's cone
                ax = rel[:3] / (np.linalg.norm(rel[:3]) + 1e-12); rel = np.concatenate([ax * np.sin(lim / 2), [np.cos(lim / 2)]])
            qp[a:a + 4] = [rel[3], rel[0], rel[1], rel[2]]
        else:
            ang = 2 * np.arctan2(rel[0], rel[3]); ang = (ang + np.pi) % (2 * np.pi) - np.pi
            qp[a] = np.clip(np.degrees(ang), jt[1], jt[2]) * np.pi / 180
    return qp


def simulate(P0, Q0, P1, Q1, dt_in, dur, push=(0, 0, 0), push_body=None, spin=(0, 0, 0), tone=0.6, obstacles=(), hz=240, settle_tone=None, jvel=0.3, lvel=0.6):
    """start from the rig pose (P1, Q1) moving as it did from (P0, Q0) dt_in earlier; add a push (m/s) to the whole body
    (or to one body) and a spin (rad/s, world) -> ts, P (n,3), Q (n,11,4)"""
    m = mujoco.MjModel.from_xml_string(model_xml(obstacles, tone=tone)); d = mujoco.MjData(m)
    qa = _qpos(m, P0, Q0); qb = _qpos(m, P1, Q1)
    qv = np.zeros(m.nv); mujoco.mj_differentiatePos(m, qv, dt_in, qa, qb)
    qv[6:] *= jvel; qv[3:6] *= jvel; qv[0:3] *= lvel     # keep a little of the pose's own motion (the springs' quick wobble is not real momentum)
    d.qpos[:] = qb; d.qvel[:] = qv
    # spring reference: hold the joints loosely toward the start pose (muscle tone), so limbs don't go fully slack
    m.qpos_spring[:] = qb
    mujoco.mj_forward(m, d)
    # push: add linear velocity (world) to the free joint, and angular velocity
    ja = m.jnt_dofadr[m.joint('root').id]
    d.qvel[ja:ja + 3] += np.asarray(push, float)
    if np.any(spin):
        # free joint angular velocity is in the body frame
        xq = d.xquat[m.body('pelvis').id]; Rm = R.qto_mat([xq[1], xq[2], xq[3], xq[0]])
        d.qvel[ja + 3:ja + 6] += Rm.T @ np.asarray(spin, float)
    n = int(round(dur * hz)); steps = int(round(1.0 / hz / m.opt.timestep))
    P = np.zeros((n, 3)); Q = np.zeros((n, 11, 4))
    ids = [m.body(b).id for b in R.BONES]
    for i in range(n):
        if settle_tone is not None and i == int(n * 0.6):
            m.jnt_stiffness[:] *= settle_tone
        P[i] = d.xpos[ids[0]]
        for k, bi in enumerate(ids):
            w, x, y, z = d.xquat[bi]; Q[i, k] = [x, y, z, w]
        for _ in range(steps): mujoco.mj_step(m, d)
    for f in range(1, n):
        flip = np.sum(Q[f] * Q[f - 1], -1) < 0; Q[f, flip] *= -1
    return np.arange(n) / hz, P, Q


def debris_sim(blocks, dur, hz=240, obstacles=()):
    """blocks: list of (pos(3), vel(3), angvel(3), size) -> ts, pos (n,k,3), quat (n,k,4 xyzw)"""
    k = len(blocks)
    xml = '<mujoco><option timestep="0.001" gravity="0 -9.81 0"/><worldbody>' \
          '<geom type="plane" size="80 80 0.1" zaxis="0 1 0" friction="0.8 0.02 0.002"/>' + \
          ''.join('<geom type="box" size="%s" pos="%s"/>' % (' '.join('%.3f' % v for v in o[3:6]), ' '.join('%.3f' % v for v in o[0:3])) for o in obstacles) + \
          ''.join('<body name="b%d" pos="%.3f %.3f %.3f"><freejoint/><geom type="box" size="%.3f %.3f %.3f" mass="%.2f" friction="0.7 0.02 0.002"/></body>' %
                  (i, *b[0], b[3], b[3], b[3], 900 * (2 * b[3]) ** 3) for i, b in enumerate(blocks)) + '</worldbody></mujoco>'
    m = mujoco.MjModel.from_xml_string(xml); d = mujoco.MjData(m)
    for i, b in enumerate(blocks):
        a = m.jnt_dofadr[i]; d.qvel[a:a + 3] = b[1]; d.qvel[a + 3:a + 6] = b[2]
    n = int(round(dur * hz)); steps = int(round(1.0 / hz / m.opt.timestep))
    pos = np.zeros((n, k, 3)); quat = np.zeros((n, k, 4))
    for i in range(n):
        for j in range(k):
            bi = m.body('b%d' % j).id; pos[i, j] = d.xpos[bi]; w, x, y, z = d.xquat[bi]; quat[i, j] = [x, y, z, w]
        for _ in range(steps): mujoco.mj_step(m, d)
    return np.arange(n) / hz, pos, quat


def lying(Q):
    """+1 face up, -1 face down, 0 on the side: from the chest's forward axis"""
    z = R.qrot(Q[R.BI['chest']], [0, 0, 1.0])[1]
    return 1 if z > 0.5 else (-1 if z < -0.5 else 0)


def search(P0, Q0, P1, Q1, dt_in, dur, target_xz, face, base_push, base_spin, tries=60, seed=1, **kw):
    """randomised search round a push / spin for a tumble that ends near target_xz lying `face` (+1 up / -1 down)"""
    rng = np.random.default_rng(seed); best = None
    for k in range(tries):
        s = 0 if k == 0 else 1
        push = np.array(base_push) * (1 + s * rng.normal(0, 0.18, 3)); spin = np.array(base_spin) + s * rng.normal(0, 1.2, 3)
        ts, P, Q = simulate(P0, Q0, P1, Q1, dt_in, dur, push=push, spin=spin, **kw)
        err = np.linalg.norm(P[-1, [0, 2]] - np.asarray(target_xz)) + (0 if lying(Q[-1]) == face else 3.0)
        if best is None or err < best[0]: best = (err, push, spin, ts, P, Q)
    return best
