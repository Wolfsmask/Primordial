# Herobrine VS Notch (beta rough draft): source

How the edit is made, in order:

1. `python3 edit_assets.py` paints the original skins, capes and blades into `web/`.
2. `python3 edit_world2.py` builds the castle courtyard, village, forest and mountains. It writes `web/world2.json`
   (generated, not committed) plus the block texture strip `web/blocks2.png` and its glow mask `web/blocks2_em.png`.
3. `web/edit.html` is the renderer (three.js r160). Put three.js in `web/node_modules/three` (`npm i three@0.160.0`
   inside `web/`). The page exposes `renderFrame(params)`, which returns a PNG of one frame.
4. `edit_anim.py` holds the whole edit on the song's beat grid (140 BPM, drop at 5.26 s):
   - the fighters' keyframed poses;
   - every camera move;
   - the effects timeline;
   - the 3D titles;
   - the edit pass (zoom punches, flashes, shake, chromatic aberration, radial blur, slow motion, the freeze and the grade).

   `beats.json` holds the detected kicks.
5. `python3 edit_film.py test` renders a contact sheet at 360 x 640. `python3 edit_film.py render 0:1860 720x1280` renders the
   full 31 s at 60 fps through headless Chromium (Playwright); the draft was rendered at 720 x 1280 and upscaled to 1080 x 1920 with Lanczos (omit the size to render natively at 1080 x 1920, about 3x slower in software GL).
6. Encode the frames with ffmpeg (H.264, 60 fps), muxed with the first 31 s of the song and a short fade-out.

`deps/` holds the shared helpers: noise (`lib.py`), and the procedural block textures and pixel font from the
old-world project.

The song is a commercial track and is not included here.
