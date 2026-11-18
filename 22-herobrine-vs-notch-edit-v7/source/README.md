# Herobrine VS Notch (v7, beta): source

Everything in the video is hand-keyed (no motion capture) on a blocky rig with bendy elbows, knees and waist, and
rendered with three.js in headless Chromium.

How it is made, in order:

1. `python3 edit_world2.py` builds the night castle courtyard, the burning village, the forest and the mountains.
   It writes `web/world2.json` (generated, not committed) and the block texture strips `web/blocks2.png` and
   `web/blocks2_em.png`. `web/world_extra.json` places the fires.
2. `web/edit6.html` is the renderer (three.js r160). Put three.js in `web/node_modules/three` (`npm i three@0.160.0`
   inside `web/`). The page exposes `renderFrame(params)`, which returns a PNG of one frame. It draws:
   - the two skins (`web/herobrine2.png`, `web/notch2.png`, supplied by the channel);
   - glows on the models themselves (eyes, fists, rim light);
   - the line teleport, the barrier and its cracks, lightning, sparks, smoke and fire;
   - bloom and the anime edit pass.
3. `python3 story7.py` keys the whole fight and writes `story7.npz` and `story7_events.json`:
   - `rig.py` is the skeleton and forward kinematics;
   - `keyanim.py` is the keyframe system (joint limits, eased arcs, overlap springs, foot plants);
   - `poses.py`, `poses4.py`, `poses5.py` and `poses7.py` are the pose libraries. `poses7.py` holds the god-like
     opening (one hand behind the back) and the Thragg-style fighting: power straights, haymakers, flying punches,
     flying knees and two-fisted hammers.

   An alignment pass moves every strike so it stops a few centimetres short of Notch (the invisible barrier).
4. `anim7.py` turns the story into frames:
   - the cameras (`bcam7.json` caches the auto-searched teleport shots);
   - the effects timeline;
   - the speed ramps, impact frames, speed lines and grade.

   `ragdoll.py` (MuJoCo) simulated the wall debris in `debris3.npz`.
5. `python3 film7.py test 6.3,8.4,13.2` renders a contact sheet. `python3 film7.py render 0:2059 720x1280` renders all
   34.3 s at 60 fps (split it across processes, e.g. `0:687`, `687:1373` and `1373:2059`).
6. `bash encode.sh` upscales to 1080 x 1920 with Lanczos and encodes H.264 at 60 fps with the song.

`deps/` holds shared helpers: noise (`lib.py`), the procedural block textures and the pixel font.

The song is a commercial track and is not included here.
