# Returning to my old world (v2): source

v2 of folder 17, rebuilt to look like the old game and to play like a cinematic edit.

Order:
1. `python3 tex2.py`: block textures (web/atlas2.png), the wolf skin, the sun, the sign text, the painting.
2. `python3 world2.py`: old-generation terrain (3D noise), the waterfall, floating islands, the house and its details,
   the light flood-fill and the meshes (web/world2.bin, web/world2.json). Imports `fbm` from `lib.py` (folder 16 sources).
3. `python3 film2.py test` for a contact sheet, then `python3 film2.py render 0:1800` for the frames (60 fps).
4. `python3 audio2.py`: the music and the sound effects (audio2.wav). It reuses the instruments in `ow_audio.py`.
5. Encode the frames and audio2.wav with ffmpeg (H.264, 60 fps, AAC).

`anim2.py` is the whole edit: the shots and their cameras, the crossfades, the dog and the player, the words of the
typography with the times they land, and the sound cues. `web/ow2.html` is the renderer (three.js r160 in
web/node_modules), driven by Playwright with SwiftShader. The typography font is Cinzel (fonts/Cinzel.ttf, OFL).
