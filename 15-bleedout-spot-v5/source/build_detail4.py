"""mesh the detailed blocks round the ledge -> web/detail_*.bin + web/meta.json"""
import json, os, time
import numpy as np
import mesher
HERE = os.path.dirname(os.path.abspath(__file__)); WEB = os.path.join(HERE, 'web4')
t0 = time.time()
d = np.load(os.path.join(HERE, 'world4.npz'))
vol, org = d['vol'].copy(), d['origin']

g = mesher.mesh_volume(vol, (int(org[0]), int(org[1]), int(org[2])), mode='textured')
mp = os.path.join(WEB, 'meta.json')
meta = json.load(open(mp)) if os.path.exists(mp) else {}
meta.setdefault('groups', {})
mesher.save_groups(g, WEB, 'detail', meta['groups'])
meta['spot'] = d['spot'].tolist()
json.dump(meta, open(mp, 'w'))
for k, v in meta['groups'].items():
    print(k, v['verts'], 'verts', v['tris'], 'tris', len(v['chunks']), 'chunks')
print('detail meshed %.0fs' % (time.time() - t0))
