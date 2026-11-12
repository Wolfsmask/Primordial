# Returning to my old world: source

Scripts that build and render this video. The generated world, frames and audio are not stored here (they are rebuilt).

Order:
1. `python3 tex_ow.py`: block textures (web/atlas.png), the wolf skin, the sun and the sign text.
2. `python3 ow_world.py`: the voxel world, the house and the details, the light flood-fill and the meshes (web/world.bin, web/world.json).
3. `python3 ow_film.py thumb`: the world icon for the menu.
4. `python3 ow_film.py test` for a contact sheet, then `python3 ow_film.py render 0:1440` for the frames.
5. `python3 ow_audio.py`: the music and the sound effects (audio.wav).
6. Encode: `ffmpeg -framerate 60 -i frames/f_%05d.png -i audio.wav -c:v libx264 -pix_fmt yuv420p -c:a aac ...`.

`ow_anim.py` holds the whole timeline: camera splines (monotone cubic, so the camera never overshoots), the walking bob,
the door, the wolf (sit, turn, stand, walk, sit, head tilt), the hearts and the sound cues.
The renderer page is `web/ow.html` (three.js r160 in web/node_modules), driven by Playwright with SwiftShader.
`ow_world.py` imports `fbm` from `lib.py` in the bleedout-spot sources (folder 16).
The footstep samples come from the CC0 pack listed in CREDITS.txt (`sfx/` next to the bleedout sources).
