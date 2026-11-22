# The End Update: source

Everything is generated: the world, its textures, the black hole, the camera moves. The 3D is rendered with three.js
in headless Chromium; the phone, the caption, the title and the last notification are drawn in 2D with PIL on top.

1. `python3 end2_tex.py` paints the original 16 x 16 End textures into four atlases in `web/`: albedo (`atlas.png`),
   emission (`atlas_em.png`), normals (`atlas_n.png`) and roughness/metalness (`atlas_s.png`).
2. `python3 end2_world.py` builds the End (the main island and the Rift Gate, the End City, the crystal spires, the
   elder grove, the far End), meshes it with `end2_mesher.py` (face culling, ambient occlusion, a separate crystal
   group) into `web/*.bin` + `web/meta.json`, and writes the feature points to `web/scene.json`. `deps/lib.py` is the noise.
3. `python3 end2_bh.py` traces light past the black hole (Schwarzschild bending, backwards from the eye) and writes the
   lensing map the sky shader reads: `web/bh_a.bin`, `bh_b.bin`, `bh_c.bin` (half floats) and `web/bh.json`. About 3 min.
4. `web/end2.html` is the renderer. Put three.js in `web/node_modules/three` (`npm i three@0.160.0` inside `web/`). It
   exposes `renderView(params)`, which returns one finished frame.
5. `python3 end_audio.py` cuts the song (not included: put it next to the scripts as `song.wav`) into `audio_edit.wav`:
   it starts at "It's your undress rehearsal", tape-stops before the drop, pulls the break under water and echoes out on
   "start". `beats.npy` holds the song's beat times.
6. `end2_anim.py` is the edit on that cut: the shots, every cut on a beat, the punch-ins and whip pans, the crystals
   charging the gate, the title and caption timing, the lights and the grade per frame.
7. `python3 end2_film.py test 10,20,30` renders a contact sheet; `python3 end2_film.py render 0:2196 1080x1920` renders
   the 36.6 s at 60 fps into `frames/` (split it across processes, e.g. `0:800`, `800:1500`, `1500:2196`).
8. `bash encode2.sh` encodes the frames with the edited music.
