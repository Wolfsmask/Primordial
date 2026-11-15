# Returning to my old world (v4): source

v4: folder 19's film with the narrator's "Hey, buddy... I'm home." taken out (`voice_clips4.json`,
`audio4.py` -> audio4.wav). The picture is the same as v3; only the sound was remixed.

From v3: the film with a narrator, and the door fixed.

- Narrator: one ElevenLabs take with the voice "Brian" (eleven_multilingual_v2, stability 0.8, similarity 0.85, style 0,
  speed 0.88, with break tags between the lines), requested with per-character timestamps. `voice/brian_narration_raw.wav`
  is that take. It is deepened with ffmpeg:
  `asetrate=44100*0.933,aresample=44100,atempo=1.0718,highpass=f=50,bass=g=4:f=120:w=0.8,equalizer=f=320:t=q:w=1.2:g=-2,equalizer=f=3200:t=q:w=1:g=2.5,acompressor=threshold=-22dB:ratio=3:attack=8:release=160:makeup=2`
  -> voice_deep.wav. `voice_clips.json` says where each line goes in the film and when each word is spoken; the on-screen
  words (`anim3.PHRASES`) and their piano notes (`audio3.py`) land on those times, and the music dips under every line.
- Door: hinged on its inner edge (`web/ow3.html`), so it swings into the room and rests inside its own doorway instead
  of passing into the wall.

Order: `python3 tex2.py` -> `python3 world2.py` -> `python3 audio3.py` -> `python3 film3.py render 0:1800` -> encode the
frames with audio3.wav (60 fps). Everything else is as in folder 18's README.
