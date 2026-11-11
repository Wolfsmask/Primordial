# Bleedout spot v6: source

Scripts that build and render this video. Generated meshes, frames and audio are not stored here (they are rebuilt).

What changed from v5:
- Distant mountains: snow caps and snow running down the gullies and couloirs, warped rock layers of several stone
  types, scree and alpine turf, darker hollows and lighter ridges, and a per-block shade grain so the far terrain keeps
  a block texture instead of flat colour (`lod5.py` tile rules, terrain shader in `web/render6.html`). Fixes a bug that
  had turned every far mountain bare grey stone and removed the snow.
- Waterfalls: each one leaves through a V-shaped gorge cut into the cliff top (`hf4.falls_carve`), from a straight
  4 m block cliff aligned with the fall (`lod5.fall_patches`). The water is an animated curtain: streaks and clumps
  that ride down with the water and speed up as they fall, a crowned lip, frayed edges, thin springs beside it, and a
  larger spray cloud at the base (`web/render6.html`).

Order: `python3 make_model.py` (player model, writes web/static.json; copy to web4/) → `python3 bloodskin.py` →
`python3 -c "import textures; textures.build('web4/atlas.png')"` → `python3 terrain4.py` → `python3 build_detail4.py` →
`python3 world4_build.py` (water, falls, the floating rock) → `python3 water6.py` (streams in the new gorges, spray
positions, spring strands) → `python3 lod5.py` (textured distant terrain, fall cliffs, forests) → `python3 motion3.py` →
`python3 fp6.py test` / `python3 fp6.py render`.
`objects3.py` (and the `hf3.py` it imports) provide the floating-rock builder and the voxel palette.
The renderer page is `web/render6.html` (served from web4/, three.js r160 in node_modules), driven by Playwright with
SwiftShader. The audio track is the same as v5's (song bed from folder 12's video plus foley).

The shader techniques in `render6.html` are partly adapted from Photon Shaders by SixthSurge
(https://github.com/sixthsurge/photon, Photon Shaders License).
