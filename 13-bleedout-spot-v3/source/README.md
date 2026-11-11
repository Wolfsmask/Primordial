# Bleedout spot v3: source

Scripts that build and render this video. Generated meshes, frames and audio are not stored here (they are rebuilt).

Order: `python3 make_model.py` (player model) → `python3 bloodskin.py` → `python3 -c "import textures; textures.build('web/atlas.png')"` →
`python3 terrain3.py` → `python3 build_detail.py` → `python3 objects3.py` → `python3 lod3.py` → `python3 motion3.py` →
`python3 fp3.py test` / `python3 fp3.py render`. The renderer page is `web/render3.html` (three.js r160 in `web/node_modules`,
served locally, driven by Playwright with SwiftShader). `fp3.py` reads the song bed from `audio/v2_bed.wav`
(the audio track of folder 12's video).

The shader techniques in `render3.html` are partly adapted from Photon Shaders by SixthSurge
(https://github.com/sixthsurge/photon, Photon Shaders License).
