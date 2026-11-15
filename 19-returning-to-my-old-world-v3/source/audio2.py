"""v2 sound for the old world video: an original soft piano piece (synthesised, with a little string pad and room),
the menu click, footsteps on grass / stone / planks (CC0 samples), the door, and the dog (synthesised whimpers and
panting, his paws on the floor).  -> audio.wav"""
import math, os, wave
import numpy as np
from scipy.signal import butter, lfilter, fftconvolve
import anim2 as A

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



def sub_hit(t_len=1.6, f0=58):
    n = int(t_len * SR); t = np.arange(n) / SR
    f = f0 * (1 + 0.6 * np.exp(-t / 0.05))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.45) * np.minimum(1, t / 0.004)


def riser(dur):
    n = int(dur * SR); t = np.arange(n) / SR; u = t / dur
    s = rng.standard_normal(n)
    out = np.zeros(n)
    for i in range(0, n, 2400):
        fc = 300 + 5000 * (i / n) ** 2
        b, a = butter(2, [max(60, fc * 0.7) / (SR / 2), min(0.99 * SR / 2, fc * 1.3) / (SR / 2)], 'band')
        out[i:i + 2400] = lfilter(b, a, s[i:i + 2400])
    return out * u ** 2.5 / (np.abs(out).max() + 1e-9)


def crickets(dur):
    n = int(dur * SR); t = np.arange(n) / SR
    out = np.zeros(n)
    for k, (f, rate, ph) in enumerate(((4300, 2.1, 0.0), (4650, 1.7, 0.4), (4020, 2.6, 0.9))):
        chirp = (np.sin(2 * np.pi * rate * t + ph) > 0.55).astype(float)
        trem = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 38 * t + k))
        out += np.sin(2 * np.pi * f * t) * chirp * trem * (0.6 + 0.4 * np.sin(2 * np.pi * 0.13 * t + k))
    b, a = butter(2, 2500 / (SR / 2), 'high'); out = lfilter(b, a, out)
    return out / (np.abs(out).max() + 1e-9)


def music():
    m = np.zeros((N, 2))
    def P(t, notes, vel=0.5, dur=3.2, spread=0.0):
        for j, n in enumerate(notes.split()):
            f = note_hz(n); pan = max(-0.6, min(0.6, math.log2(f / 262) * 0.25))
            place(m, piano(f, dur, vel), t + j * spread, 1.0, pan)
    # the menu: a few soft high notes, like opening an old save
    P(0.25, 'A5', 0.2, 2.4); P(0.7, 'F#5', 0.18, 2.4); P(1.15, 'D5', 0.2, 2.4); P(1.7, 'E5', 0.15, 2.0)
    # the world opens
    P(2.3, 'D2 A2 F#3 C#4 E4', 0.42, 4.6, spread=0.06)
    P(4.7, 'B1 F#2 A3 D4 E4', 0.36, 3.0, spread=0.08)
    P(7.3, 'G1 D2 F#3 B3', 0.34, 3.0, spread=0.09)
    P(9.8, 'F#1 D2 A3 D4', 0.32, 1.8, spread=0.09)
    P(11.2, 'A1 E2 A3 D4', 0.3, 1.4, spread=0.1); P(11.9, 'A1 E2 A3 C#4', 0.28, 1.0, spread=0.1)
    P(12.55, 'E1 B1 G3 D4 F#4', 0.28, 2.4, spread=0.14)
    P(14.6, 'B1 F#2 A3 D4', 0.34, 1.6, spread=0.08); P(15.7, 'G1 D2 B3 D4 A4', 0.42, 1.4, spread=0.06)
    P(16.9, 'A1 E2 C#4 E4', 0.36, 1.2, spread=0.06)
    P(18.0, 'F#1 D2 A3 D4', 0.38, 1.6, spread=0.05)
    for j, n in enumerate('D5 F#5 A5 D6 A5 F#5 A5 D6'.split()):
        P(18.05 + j * 0.15, n, 0.24, 1.6)
    P(19.6, 'G1 D2 B3 F#4', 0.32, 1.4, spread=0.1)
    P(20.4, 'E1 B1 G3 D4 F#4', 0.24, 2.6, spread=0.16)
    P(24.15, 'D1 A1 F#3 C#4 E4 A4', 0.48, 5.5, spread=0.07)
    # the words, each on a note
    for t, n, v in [(2.85, 'F#5', 0.36), (3.35, 'A5', 0.40), (4.8, 'E5', 0.3), (5.1, 'D5', 0.28), (5.5, 'C#5', 0.3), (5.9, 'A4', 0.32),
                    (8.1, 'B4', 0.28), (8.4, 'D5', 0.3), (8.95, 'F#5', 0.32), (9.3, 'E5', 0.3), (9.85, 'D5', 0.28), (10.15, 'B4', 0.3),
                    (13.2, 'G4', 0.26), (13.55, 'B4', 0.28), (14.75, 'D5', 0.3), (15.1, 'E5', 0.32), (15.75, 'F#5', 0.4),
                    (22.5, 'A4', 0.3), (22.85, 'B4', 0.32), (23.45, 'D5', 0.36), (24.15, 'F#5', 0.44)]:
        P(t, n, v, 3.0)
    # home, at dusk: a slow broken chord or two, fading
    for j, n in enumerate('A4 D5 E5 F#5 A5 F#5 E5 D5'.split()):
        P(25.6 + j * 0.5, n, 0.22 - j * 0.012, 3.0)
    P(27.6, 'G1 D2 B3 F#4', 0.22, 2.5, spread=0.12)
    pads = [((2.3, 12.6), 'D3 F#3 A3'), ((12.4, 15.9), 'E3 G3 B3'), ((15.6, 20.6), 'B2 D3 F#3 A3'), ((20.2, 24.3), 'E3 G3 B3'), ((24.0, 30.0), 'D3 F#3 A3 E4')]
    for (a, b), c in pads:
        s = pad([note_hz(x) for x in c.split()], a, b, 0.016 if a < 15 else 0.022)
        place(m, s, a, 1.0, -0.25); place(m, s, a + 0.013, 1.0, 0.25)
    for t, g in ((2.3, 0.55), (15.75, 0.45), (24.15, 0.6)):
        place(m, sub_hit(), t, g)
    place(m, riser(0.7), 1.6, 0.05)
    return m


def sfx():
    s = np.zeros((N, 2))
    n = int(0.06 * SR); t = np.arange(n) / SR
    click = (np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.006) + 0.4 * rng.standard_normal(n) * np.exp(-t / 0.003) + 0.6 * np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.02))
    place(s, click, 0.98, 0.22)
    grass = slices(load('leaves01.wav')) + slices(load('leaves02.wav')) + slices(load('mud02.wav'))
    stone = slices(load('stone01.wav')) + slices(load('gravel.wav'))
    wood = slices(load('wood01.wav')) + slices(load('wood02.wav')) + slices(load('wood03.wav'))
    for i, (t, kind) in enumerate(A.fp_steps()):
        pool = grass if kind == 'grass' else stone
        hit = pitch(pool[int(rng.integers(len(pool)))], 0.92 + 0.16 * rng.random())
        place(s, hit, t, 0.17 / (np.abs(hit).max() + 1e-6), pan=(-0.12 if i % 2 else 0.12))
    latch = pitch(wood[0], 1.35); place(s, latch, 12.5, 0.2 / (np.abs(latch).max() + 1e-6))
    place(s, creak(0.62), 12.58, 0.09, 0.1)
    thud = pitch(wood[2 % len(wood)], 0.8); place(s, thud, 13.45, 0.15 / (np.abs(thud).max() + 1e-6), 0.15)
    for k, t in enumerate((17.15, 17.55)):                   # his steps into the house
        h = pitch(wood[k % len(wood)], 1.0); place(s, h, t, 0.12 / (np.abs(h).max() + 1e-6), -0.1)
    place(s, whimper(0.5, 760, 1150, 880), 14.85, 0.07, -0.1)
    place(s, whimper(0.65, 820, 1320, 1000), 17.05, 0.085, 0.0)
    place(s, whimper(0.45, 900, 1250, 1050), 19.75, 0.06, 0.0)
    paw = [pitch(w, 1.8) for w in wood]
    for i, t in enumerate(A.dog_steps()):
        p = paw[i % len(paw)]; place(s, p, t, 0.055 / (np.abs(p).max() + 1e-6), 0.05)
    place(s, pant(10), 20.3, 0.04, 0.0); place(s, pant(8, 2.5), 22.6, 0.03, 0.0)
    place(s, crickets(4.8), 25.2, 0.018, 0.2)
    return s


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


if __name__ == '__main__':
    mus = reverb(music(), 2.8, 0.36)
    fx = reverb(sfx(), 0.9, 0.14)
    mix = mus * 0.6 + fx
    t = np.arange(N) / SR
    mix *= np.minimum(1, t / 0.05)[:, None]
    mix *= (1 - np.clip((t - 28.8) / 1.15, 0, 1) ** 1.5)[:, None]
    mix = mix[:int(A.T_END * SR)]
    mix /= np.abs(mix).max() / 0.89
    pcm = (np.clip(mix, -1, 1) * 32767).astype(np.int16)
    with wave.open(os.path.join(HERE, 'audio2.wav'), 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    print('audio2', pcm.shape[0] / SR, 's')
