# Bleedout spot v5: source

Scripts that build and render this video. Generated meshes, frames and audio are not stored here (they are rebuilt).

Order: `python3 make_model.py` (player model, writes web/static.json; copy to web4/) → `python3 bloodskin.py` →
`python3 -c "import textures; textures.build('web4/atlas.png')"` → `python3 terrain4.py` → `python3 build_detail4.py` →
`python3 world4_build.py` (water, falls, the floating rock) → `python3 lod5.py` (textured distant terrain and forests) → `python3 motion3.py` → `python3 fp5.py test` / `python3 fp5.py render`.
`objects3.py` (and the `hf3.py` it imports) provide the floating-rock builder and the voxel palette.
The renderer page is `web/render5.html` (served from web4/, three.js r160 in node_modules), driven by Playwright with
SwiftShader. `fp5.py` reads the song bed from `audio/v2_bed.wav` (the audio track of folder 12's video).

The shader techniques in `render5.html` are partly adapted from Photon Shaders by SixthSurge
(https://github.com/sixthsurge/photon, Photon Shaders License).
