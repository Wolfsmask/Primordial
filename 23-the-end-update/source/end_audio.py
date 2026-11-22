"""The music edit for the End Update video.  Song time -> video time:
  [12.00, 20.90)  plays as is ("It's your undress rehearsal ... back seat")
  tape stop        20.90 -> 21.10 slows to a halt over 0.40 s
  silence          0.25 s
  [21.40, 46.13)  plays from just before the drop, with two bars of the drop spliced out (song 24.056 -> 29.234, on
                   the beat, where the two bars match); the break (39.6 - 41.4) is pulled under water (low-passed)
  the last word    "start" is thrown into echoes that ring out into the dark (+ a low boom)
python3 end_audio.py -> audio_edit.wav (48 kHz stereo) and prints the key times"""
import numpy as np, librosa, soundfile as sf
from scipy.signal import butter, sosfilt

SR = 48000
T0 = 12.00
A_END = 20.90; TS = 0.40; GAP = 0.25; B0 = 21.40; B_END = 46.13
CUT_A, CUT_B = 24.055873015873015, 29.23392290249433    # beats 33 and 41: song [CUT_A, CUT_B) is left out
OUT_B0 = (A_END - T0) + TS + GAP                 # video time where song B0 lands
def out_of(song_t):
    if song_t >= CUT_B: return song_t - (CUT_B - CUT_A) - B0 + OUT_B0
    return song_t - B0 + OUT_B0 if song_t >= B0 else song_t - T0


def main():
    y, _ = librosa.load('song.wav', sr=SR, mono=False)          # (2, n)
    S = lambda a, b: y[:, int(a * SR):int(b * SR)].copy()
    A = S(T0, A_END)
    A[:, :int(0.04 * SR)] *= np.linspace(0, 1, int(0.04 * SR))
    # tape stop: playback speed falls linearly 1 -> 0 over TS; it reads TS/2 seconds of song
    src = S(A_END, A_END + TS)
    n = int(TS * SR); tt = np.arange(n) / SR; pos = (tt - tt * tt / (2 * TS)) * SR
    tape = np.stack([np.interp(pos, np.arange(src.shape[1]), src[c]) for c in range(2)])
    tape *= np.linspace(1, 0, n) ** 0.5
    gap = np.zeros((2, int(GAP * SR)))
    Bm = S(B0, B_END)
    # the break under water: crossfade to a 700 Hz low-passed copy and back (opens up on "Girl")
    sos = butter(4, 700, 'lowpass', fs=SR, output='sos'); wet = np.stack([sosfilt(sos, Bm[c]) for c in range(2)]) * 1.35
    tb = B0 + np.arange(Bm.shape[1]) / SR
    w = np.clip((tb - 39.55) / 0.5, 0, 1) * np.clip((41.42 - tb) / 0.12, 0, 1)
    Bm = Bm * (1 - w) + wet * w
    # the splice: 8 ms ahead of both beats so the kick lands whole, a 15 ms equal-power crossfade
    ia = int((CUT_A - 0.008 - B0) * SR); ib = int((CUT_B - 0.008 - B0) * SR); x = int(0.015 * SR); q = np.linspace(0, 1, x) * np.pi / 2
    Bm = np.concatenate([Bm[:, :ia - x], Bm[:, ia - x:ia] * np.cos(q) + Bm[:, ib - x:ib] * np.sin(q), Bm[:, ib:]], 1)
    # the end: "start" thrown into echoes
    tail_len = 2.4; out = np.concatenate([A, tape, gap, Bm], 1)
    word = S(45.30, 45.95); echo = np.zeros((2, int((0.65 + tail_len) * SR)))
    for k, (dl, g) in enumerate([(0.0, 0.0), (0.33, 0.55), (0.66, 0.36), (0.99, 0.22), (1.32, 0.13), (1.65, 0.07)]):
        if g == 0: continue
        i = int(dl * SR); seg = word * g
        if k % 2: seg = seg[::-1] * 0 + seg            # (keep stereo)
        sl = sosfilt(butter(2, 3000 - k * 380, 'lowpass', fs=SR, output='sos'), seg)
        echo[:, i:i + sl.shape[1]] += sl[:, :max(0, min(sl.shape[1], echo.shape[1] - i))]
    t_cut = out_of(45.95)
    fade = int(0.06 * SR); ic = int(t_cut * SR)
    out[:, ic - fade:ic] *= np.linspace(1, 0, fade); out = out[:, :ic]
    boom_t = np.arange(int(1.8 * SR)) / SR
    boom = 0.55 * np.sin(2 * np.pi * (48 * boom_t - 10 * boom_t * boom_t)) * np.exp(-boom_t * 2.4)
    tail = echo.copy(); tail[:, :len(boom_t)] += boom
    out = np.concatenate([out, tail], 1)
    out = out / (np.abs(out).max() + 1e-9) * 0.95
    sf.write('audio_edit.wav', out.T, SR)
    print('duration %.3f' % (out.shape[1] / SR))
    for nm, t in (('drop 21.455', 21.455), ('Ooh 21.52', 21.52), ('break 40.26', 40.26), ('Girl 41.37', 41.37), ('start 45.04', 45.04), ('cut 45.95', 45.95)):
        print('%-12s -> video %.3f' % (nm, out_of(t)))


if __name__ == '__main__':
    main()
