"""Sound for the old world video: an original soft piano piece (synthesised, with a little string pad and room),
the menu click, footsteps on grass / stone / planks (CC0 samples), the door, and the dog (synthesised whimpers and
panting, his paws on the floor).  -> audio.wav"""
import math, os, wave
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
import ow_anim as A

HERE = os.path.dirname(os.path.abspath(__file__)); SFX = os.path.join(HERE, '..', 'bleed3', 'sfx')
SR = 48000
N = int(A.T_END * SR) + SR
rng = np.random.default_rng(12)


def note_hz(n):
    names = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    k = names[n[0]]; i = 1
    if n[i] == '#':
        k += 1; i += 1
    elif n[i] == 'b':
        k -= 1; i += 1
    octave = int(n[i:])
    return 440.0 * 2 ** ((k + 12 * (octave + 1) - 69) / 12)


def piano(f, dur, vel):
    n = int(dur * SR); t = np.arange(n) / SR
    out = np.zeros(n)
    tau0 = 3.2 * (261.6 / f) ** 0.45
    B = 0.00012
    for k in range(1, 11):
        fk = f * k * math.sqrt(1 + B * k * k)
        if fk > 16000:
            break
        a = vel / k ** 1.35 * (1.0 if k > 1 else 1.15) * (0.6 if k == 2 else 1.0)
        tau = tau0 / (1 + 0.55 * (k - 1)) * (0.8 + 0.4 * rng.random())
        out += a * np.sin(2 * np.pi * fk * t + rng.random() * 6.28) * np.exp(-t / tau)
    # a second, slightly detuned string for warmth
    out += 0.35 * vel * np.sin(2 * np.pi * f * 1.0012 * t) * np.exp(-t / (tau0 * 1.2))
    att = np.minimum(1, t / 0.004)
    hammer = rng.standard_normal(n) * np.exp(-t / 0.006) * 0.08 * vel
    b, a_ = butter(2, min(0.99, f * 6 / (SR / 2)))
    out = out * att + lfilter(b, a_, hammer)
    rel = np.ones(n); r0 = int(max(0, dur - 0.25) * SR); rel[r0:] = np.linspace(1, 0, n - r0)
    return out * rel


def pad(freqs, t0, t1, vol):
    n = int((t1 - t0) * SR); t = np.arange(n) / SR
    s = np.zeros(n)
    for f in freqs:
        for det in (-0.004, 0.0, 0.005):
            ph = 2 * np.pi * f * (1 + det) * t
            s += (2 * ((ph / (2 * np.pi)) % 1) - 1) * 0.33         # saw
    b, a = butter(2, 1100 / (SR / 2)); s = lfilter(b, a, s)
    env = np.minimum(1, t / 1.6) * np.minimum(1, (t1 - t0 - t) / 1.4)
    return s * env * vol


def place(buf, sig, t, gain=1.0, pan=0.0):
    i = int(t * SR)
    if i >= len(buf):
        return
    sig = sig[:len(buf) - i]
    l = math.cos((pan + 1) * math.pi / 4); r = math.sin((pan + 1) * math.pi / 4)
    buf[i:i + len(sig), 0] += sig * gain * l * 1.41
    buf[i:i + len(sig), 1] += sig * gain * r * 1.41


def music():
    m = np.zeros((N, 2))
    def P(t, notes, vel=0.5, dur=3.2, spread=0.0):
        for j, n in enumerate(notes.split()):
            f = note_hz(n)
            pan = max(-0.6, min(0.6, (math.log2(f / 262) * 0.25)))
            place(m, piano(f, dur, vel), t + j * spread, 1.0, pan)
    # the menu: three soft high notes, like opening an old save
    P(0.25, 'A5', 0.22, 2.4); P(0.75, 'F#5', 0.2, 2.4); P(1.25, 'D5', 0.22, 2.6); P(1.9, 'E5', 0.16, 2.2)
    # walking through the meadow: D maj7 - B m7 - G maj7 - A sus, a slow broken-chord left hand, a sparse tune
    chords = [(2.4, 'D3 A3 C#4 F#4'), (4.9, 'B2 F#3 A3 D4'), (7.4, 'G2 D3 F#3 B3'), (9.9, 'A2 E3 A3 D4')]
    for t, c in chords:
        P(t, c, 0.30, 3.2, spread=0.11)
    for t, n, v in [(2.6, 'F#5', 0.30), (3.45, 'E5', 0.26), (4.1, 'D5', 0.28), (5.1, 'C#5', 0.26), (5.75, 'D5', 0.24), (6.4, 'A4', 0.28),
                    (7.6, 'B4', 0.28), (8.3, 'D5', 0.26), (9.0, 'F#5', 0.30), (10.1, 'E5', 0.28), (10.9, 'D5', 0.24), (11.6, 'C#5', 0.22)]:
        P(t, n, v, 2.8)
    # the door, the room: G maj7 then E m9, held, almost nothing on top
    P(12.4, 'G2 D3 B3 F#4', 0.26, 3.4, spread=0.14); P(14.2, 'E2 B2 G3 D4 F#4', 0.24, 3.0, spread=0.16)
    P(13.5, 'B4', 0.18, 2.5)
    # he knows you: B m7 - G - D/F# - A, the tune climbing
    for t, c in [(15.6, 'B2 F#3 A3 D4'), (16.8, 'G2 D3 G3 B3'), (18.0, 'F#2 D3 A3 D4'), (19.2, 'A2 E3 A3 C#4')]:
        P(t, c, 0.36, 2.6, spread=0.09)
    for t, n, v in [(15.75, 'A4', 0.32), (16.2, 'B4', 0.32), (16.85, 'D5', 0.34), (17.5, 'F#5', 0.36), (18.1, 'E5', 0.34),
                    (18.6, 'D5', 0.32), (19.25, 'E5', 0.34), (19.75, 'A5', 0.38)]:
        P(t, n, v, 2.6)
    # home: D maj9, a slow arpeggio over it, let ring
    P(20.4, 'D2 A2 F#3 C#4 E4', 0.36, 3.6, spread=0.12)
    for j, n in enumerate('A4 D5 E5 F#5 A5 F#5 E5'.split()):
        P(20.7 + j * 0.42, n, 0.24 - j * 0.015, 3.0)
    pads = [((2.6, 12.6), 'D3 F#3 A3'), ((12.4, 15.8), 'G3 B3 D4'), ((15.6, 20.6), 'B2 D3 F#3 A3'), ((20.2, 24.0), 'D3 F#3 A3 E4')]
    for (a, b), c in pads:
        s = pad([note_hz(x) for x in c.split()], a, b, 0.016 if a < 15 else 0.024)
        place(m, s, a, 1.0, -0.25); place(m, s, a + 0.013, 1.0, 0.25)
    return m


def reverb(x, rt=2.4, mix=0.32):
    n = int(rt * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for ch in range(2):
        ir = rng.standard_normal(n) * np.exp(-6.9 * t / rt)
        b, a = butter(1, 5000 / (SR / 2)); ir = lfilter(b, a, ir); ir[:int(0.012 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        out[:, ch] = fftconvolve(x[:, ch], ir)[:len(x)]
    return x * (1 - mix) + out * mix * 1.6


def load(name):
    with wave.open(os.path.join(SFX, name)) as w:
        sr = w.getframerate(); ch = w.getnchannels(); d = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
    if ch == 2:
        d = d.reshape(-1, 2).mean(1)
    if sr != SR:
        d = np.interp(np.arange(int(len(d) * SR / sr)) * sr / SR, np.arange(len(d)), d)
    return d


def pitch(sig, r):
    return np.interp(np.arange(int(len(sig) / r)) * r, np.arange(len(sig)), sig)


def slices(sig, n_min=0.08):
    """split a sample of several steps into single hits at the loud onsets"""
    env = np.convolve(np.abs(sig), np.ones(256) / 256, 'same')
    thr = env.max() * 0.25
    on = np.nonzero((env[1:] > thr) & (env[:-1] <= thr))[0]
    out = []
    for i, s in enumerate(on):
        e = on[i + 1] if i + 1 < len(on) else len(sig)
        if (e - s) / SR >= n_min:
            seg = sig[max(0, s - 200):min(e, s + int(0.35 * SR))]
            seg = seg * np.minimum(1, np.linspace(1, 0, len(seg)) * 4)
            out.append(seg)
    return out or [sig[:int(0.3 * SR)]]


def creak(dur):
    """a dry hinge: stick-slip pulses at a wandering rate through a few wooden resonances"""
    n = int(dur * SR); t = np.arange(n) / SR
    rate = 55 + 35 * np.sin(2 * np.pi * 1.3 * t + 0.5) + 15 * rng.standard_normal(n).cumsum() / np.sqrt(np.arange(1, n + 1)) * 0.2
    ph = np.cumsum(rate / SR)
    pulses = (np.diff(np.floor(ph), prepend=0) > 0).astype(float) * (0.6 + 0.4 * rng.random(n))
    s = np.zeros(n)
    for fc, q, g in ((420, 18, 1.0), (930, 20, 0.7), (1750, 22, 0.45), (2600, 25, 0.25)):
        b, a = butter(2, [fc * (1 - 1 / q) / (SR / 2), fc * (1 + 1 / q) / (SR / 2)], 'band')
        s += lfilter(b, a, pulses) * g
    env = np.minimum(1, t / 0.05) * np.minimum(1, (dur - t) / 0.12) * (0.6 + 0.4 * np.sin(np.pi * t / dur))
    return s * env / (np.abs(s).max() + 1e-9)


def whimper(dur=0.55, f0=780, peak=1180, end=900):
    n = int(dur * SR); t = np.arange(n) / SR; u = t / dur
    f = np.where(u < 0.25, f0 + (peak - f0) * (u / 0.25) ** 0.7, peak + (end - peak) * (np.clip(u - 0.25, 0, None) / 0.75) ** 1.3)
    f = f * (1 + 0.018 * np.sin(2 * np.pi * 7.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = sum(np.sin(k * ph) / k ** 1.6 for k in range(1, 7))
    s += rng.standard_normal(n) * 0.18
    b, a = butter(2, [700 / (SR / 2), 4200 / (SR / 2)], 'band'); s = lfilter(b, a, s)
    env = np.minimum(1, t / 0.035) * (1 - u) ** 0.8
    return s * env / (np.abs(s).max() + 1e-9)


def pant(n_breaths, rate=2.7):
    out = np.zeros(int((n_breaths / rate + 0.3) * SR))
    for i in range(n_breaths):
        d = 0.16 + 0.03 * rng.random(); m = int(d * SR); t = np.arange(m) / SR
        s = rng.standard_normal(m)
        lo, hi = (700, 2600) if i % 2 == 0 else (550, 2000)
        b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], 'band'); s = lfilter(b, a, s)
        env = np.sin(np.pi * t / d) ** 1.5 * (1.0 if i % 2 == 0 else 0.7)
        i0 = int(i / rate * SR)
        out[i0:i0 + m] += s * env
    return out / (np.abs(out).max() + 1e-9)


def sfx():
    s = np.zeros((N, 2))
    # menu click
    n = int(0.06 * SR); t = np.arange(n) / SR
    click = (np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.006) + 0.4 * rng.standard_normal(n) * np.exp(-t / 0.003) + 0.6 * np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.02))
    place(s, click, 1.03, 0.22)
    grass = slices(load('leaves01.wav')) + slices(load('leaves02.wav')) + slices(load('mud02.wav'))
    stone = slices(load('stone01.wav')) + slices(load('gravel.wav'))
    wood = slices(load('wood01.wav')) + slices(load('wood02.wav')) + slices(load('wood03.wav'))
    for i, (t, kind, amt) in enumerate(A.footsteps()):
        pool = {'grass': grass, 'stone': stone, 'wood': wood}[kind]
        hit = pitch(pool[int(rng.integers(len(pool)))], 0.92 + 0.16 * rng.random())
        place(s, hit, t, (0.20 if kind != 'wood' else 0.16) * (0.6 + 0.4 * amt) / (np.abs(hit).max() + 1e-6), pan=(-0.12 if i % 2 else 0.12))
    # the door: the latch, the hinge, the door settling against the wall inside
    latch = pitch(wood[0], 1.35); place(s, latch, 12.74, 0.22 / (np.abs(latch).max() + 1e-6))
    place(s, creak(0.62), 12.8, 0.10, 0.1)
    thud = pitch(wood[2 % len(wood)], 0.8); place(s, thud, 13.42, 0.18 / (np.abs(thud).max() + 1e-6), 0.15)
    # the dog: ears up and a small whimper; another as he stands; his paws; panting as he sits in front of you
    place(s, whimper(0.5, 760, 1150, 880), 15.85, 0.075, -0.15)
    place(s, whimper(0.65, 820, 1320, 1000), 17.15, 0.09, -0.05)
    place(s, whimper(0.45, 900, 1250, 1050), 19.15, 0.06, 0.0)
    paw = [pitch(w, 1.8) for w in wood]
    for i, t in enumerate(A.wolf_steps()):
        p = paw[i % len(paw)]; place(s, p, t, 0.06 / (np.abs(p).max() + 1e-6), -0.1)
    pt = pant(10); place(s, pt, 19.7, 0.045, 0.0)
    pt2 = pant(8, 2.5); place(s, pt2, 21.6, 0.035, 0.0)
    return s


if __name__ == '__main__':
    mus = reverb(music(), 2.6, 0.34)
    fx = reverb(sfx(), 0.9, 0.14)
    mix = mus * 0.55 + fx
    t = np.arange(N) / SR
    mix *= np.minimum(1, t / 0.05)[:, None]
    mix *= (1 - np.clip((t - 23.0) / 1.0, 0, 1) ** 1.5)[:, None]          # out with the picture
    mix = mix[:int(A.T_END * SR)]
    mix /= np.abs(mix).max() / 0.89
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(os.path.join(HERE, 'audio.wav'), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print('audio', pcm.shape[0] / SR, 's')
