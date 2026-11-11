"""v6: rebuild only the water surfaces (the hanging-valley streams follow the new gorges) and the fall bases"""
import json, os, time
import numpy as np
import hf4, mesher
import world4_build as W4
t0 = time.time()
mp = os.path.join(W4.WEB, 'meta.json'); meta = json.load(open(mp))
B = mesher.Builder(); W4.water(B)
mesher.save_groups({'water': B.build()}, W4.WEB, 'wat', meta['groups'])
falls = json.load(open(os.path.join(W4.WEB, 'waterfalls.json')))
# thin side strands: springs that break out of the rock just beside each big fall, a little below the cliff top
rng = np.random.default_rng(5)
for f in falls:
    if f['fade']:
        continue
    st = []
    ux, uz = f['dir']; tx, tz = -uz, ux
    for sgn in (-1, 1):
        for k in range(2 if f['width'] > 50 else 1):
            off = sgn * (f['width'] * 0.5 + 14 + 16 * k + rng.uniform(0, 8))
            x, z = f['x'] + tx * off - ux * 1.0, f['z'] + tz * off - uz * 1.0
            hb = float(hf4.height(np.array([x - ux * 4.0]), np.array([z - uz * 4.0]))[0])    # the rock just behind
            st.append([x, hb - 3.0 - rng.uniform(0, 6), z, f['width'] * rng.uniform(0.05, 0.09)])
    f['strands'] = st
json.dump(falls, open(os.path.join(W4.WEB, 'waterfalls.json'), 'w'))
# the spray rises where the water lands, ~40 m out from the foot of the cliff
meta['fall_bases'] = [[f['x'] + f['dir'][0] * 45, hf4.WATER_Y + 8, f['z'] + f['dir'][1] * 45, f['width']] for f in falls if not f['fade']]
json.dump(meta, open(mp, 'w'))
print('water', meta['groups']['wat_water']['tris'], '%.0fs' % (time.time() - t0))
