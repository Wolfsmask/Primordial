"""The player model (slim arms) as box meshes with standard 64x64 skin UVs, inner layer + outer layer, in each part's
own px space (y down, front -z), matching the renderer's part pivots.  Writes web/static.json."""
import base64, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

# name: (box min, box max, tex origin inner, tex origin outer, outer inflate)
PARTS = {
    'head': ((-4, -8, -4), (4, 0, 4), (0, 0), (32, 0), 0.5),
    'body': ((-4, 0, -2), (4, 12, 2), (16, 16), (16, 32), 0.25),
    'right_arm': ((-2, -2, -2), (1, 10, 2), (40, 16), (40, 32), 0.25),
    'left_arm': ((-1, -2, -2), (2, 10, 2), (32, 48), (48, 48), 0.25),
    'right_leg': ((-2, 0, -2), (2, 12, 2), (0, 16), (0, 32), 0.25),
    'left_leg': ((-2, 0, -2), (2, 12, 2), (16, 48), (0, 48), 0.25),
}


def box(lo, hi, tex, infl):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    w, h, d = x1 - x0, y1 - y0, z1 - z0
    u0, v0 = tex
    X0, Y0, Z0, X1, Y1, Z1 = x0 - infl, y0 - infl, z0 - infl, x1 + infl, y1 + infl, z1 + infl
    P, N, U = [], [], []

    def quad(corners, uvs, n):
        for i in (0, 1, 2, 0, 2, 3):
            P.append(corners[i]); N.append(n); U.append((uvs[i][0] / 64, uvs[i][1] / 64))
    # front (-z): u = u0+d+(x-x0), v = v0+d+(y-y0)
    quad([(X0, Y0, Z0), (X1, Y0, Z0), (X1, Y1, Z0), (X0, Y1, Z0)],
         [(u0 + d, v0 + d), (u0 + d + w, v0 + d), (u0 + d + w, v0 + d + h), (u0 + d, v0 + d + h)], (0, 0, -1))
    # back (+z): mirrored
    quad([(X1, Y0, Z1), (X0, Y0, Z1), (X0, Y1, Z1), (X1, Y1, Z1)],
         [(u0 + 2 * d + w, v0 + d), (u0 + 2 * d + 2 * w, v0 + d), (u0 + 2 * d + 2 * w, v0 + d + h), (u0 + 2 * d + w, v0 + d + h)], (0, 0, 1))
    # his right side (-x): u = u0+(z1-z)
    quad([(X0, Y0, Z1), (X0, Y0, Z0), (X0, Y1, Z0), (X0, Y1, Z1)],
         [(u0, v0 + d), (u0 + d, v0 + d), (u0 + d, v0 + d + h), (u0, v0 + d + h)], (-1, 0, 0))
    # his left side (+x): u = u0+d+w+(z-z0)
    quad([(X1, Y0, Z0), (X1, Y0, Z1), (X1, Y1, Z1), (X1, Y1, Z0)],
         [(u0 + d + w, v0 + d), (u0 + 2 * d + w, v0 + d), (u0 + 2 * d + w, v0 + d + h), (u0 + d + w, v0 + d + h)], (1, 0, 0))
    # top (y0, normal -y): u = u0+d+(x-x0), v = v0+(z1-z)
    quad([(X0, Y0, Z1), (X1, Y0, Z1), (X1, Y0, Z0), (X0, Y0, Z0)],
         [(u0 + d, v0), (u0 + d + w, v0), (u0 + d + w, v0 + d), (u0 + d, v0 + d)], (0, -1, 0))
    # bottom (y1, normal +y)
    quad([(X0, Y1, Z0), (X1, Y1, Z0), (X1, Y1, Z1), (X0, Y1, Z1)],
         [(u0 + d + w, v0 + d), (u0 + d + 2 * w, v0 + d), (u0 + d + 2 * w, v0), (u0 + d + w, v0)], (0, 1, 0))
    return P, N, U


def b64(a):
    return base64.b64encode(np.asarray(a, np.float32).tobytes()).decode()


bones = []
for name, (lo, hi, ti, to, infl) in PARTS.items():
    P, N, U = box(lo, hi, ti, 0.0)
    P2, N2, U2 = box(lo, hi, to, infl)
    bones.append(dict(name=name, pos=b64(P + P2), nrm=b64(N + N2), uv=b64(U + U2)))
json.dump(dict(bones=bones, arm_y=2.5, slim=True, tex='static.png'), open(os.path.join(HERE, 'web', 'static.json'), 'w'))
print('wrote static.json', [b['name'] for b in bones])
