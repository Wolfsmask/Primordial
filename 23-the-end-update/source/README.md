# The End Update: source

Everything is generated: the world, its textures, the camera moves, the dragon's flight. The 3D is rendered with three.js
in headless Chromium, the phone, notifications and lyric subtitles are drawn in 2D with PIL on top.

1. `python3 end_tex.py` paints the original 16 x 16 End textures into `web/atlas.png` (and their glow into `web/atlas_em.png`).
2. `python3 end_world.py` builds the End: the main island (obsidian pillars, crystals, the exit portal) and the isles round
   it, meshes them with `end_mesher.py` (face culling, ambient occlusion) into `web/*.bin` + `web/meta.json`, and writes the
   feature points (crystals, portal, glows, lamps, void falls, the eclipse) to `web/scene.json`. `deps/lib.py` is the noise.
3. `web/end.html` is the renderer. Put three.js in `web/node_modules/three` (`npm i three@0.160.0` inside `web/`). It exposes
   `renderView(params)`, which returns one finished frame.
4. `python3 end_audio.py` cuts the song (not included: put it next to the scripts as `song.wav`) into `audio_edit.wav`:
   it starts at "It's your undress rehearsal", tape-stops before the drop, pulls the break under water and echoes out on
   "start".
5. `end_anim.py` is the timeline on that edit: the shots (splines), the dragon (its path and spine), crystal beams, lights,
   the edit pass, the title, the lyric word timings.
6. `python3 end_film.py test 10,20,30` renders a contact sheet; `python3 end_film.py render 0:2196 1080x1920` renders the
   36.6 s at 60 fps into `frames/` (split it across processes, e.g. `0:732`, `732:1464`, `1464:2196`).
7. `bash encode.sh` encodes the frames with the edited music.
