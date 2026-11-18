"""v7: Herobrine fights like a god, not a boxer.  No guard: he stands tall, chin level, the left hand behind his back.
Every strike is one-handed and effortless: open-palm thrusts, backhand swats, a one-arm smite from above.  He floats
instead of jumping (legs hang loose).  Reactions stay upright: he slides back on his feet without a hand down."""
from poses import P
from poses5 import *          # noqa: F401,F403
import poses as P3

# the left hand tucked behind the small of the back
BEHIND = dict(afL=-46, aaL=-10, atL=60, aeL=52)
# legs: planted square (not a fighter's stagger), or hanging loose while he floats
SQUARE = dict(lfR=2, laR=4, ltR=10, lkR=6, lfL=2, laL=4, ltL=10, lkL=6)
HANG = dict(lfR=10, laR=4, ltR=6, lkR=20, lfL=-4, laL=5, ltL=6, lkL=10)


def G(legs=SQUARE, back=True, **kw):
    d = dict(pitch=0, sb=2, st=0, nb=4, nt=0)
    d.update(legs)
    if back: d.update(BEHIND)
    else: d.update(afL=4, aaL=9, atL=6, aeL=14)
    d.update(kw)
    return P(**d)


# standing tall, the right arm loose
GOD = G(afR=4, aaR=9, atR=6, aeR=14)
GOD_FLOAT = G(HANG, afR=6, aaR=10, atR=6, aeR=16)
# palm: the hand comes up beside the shoulder, then goes straight out; the chest hardly turns
PALM_LOAD = G(st=-10, nt=10, afR=38, aaR=12, atR=14, aeR=118)
PALM_HIT = G(pitch=2, sb=4, st=14, nt=-12, afR=94, aaR=2, atR=-12, aeR=2)
FPALM_LOAD = G(HANG, st=-10, nt=10, afR=38, aaR=12, atR=14, aeR=118)
FPALM_HIT = G(HANG, pitch=3, sb=4, st=14, nt=-12, afR=94, aaR=2, atR=-12, aeR=2)
# backhand swat: the arm folds across the chest, then sweeps out flat to the right
SWAT_LOAD = G(st=-30, nt=24, afR=64, aaR=-20, atR=50, aeR=104)
SWAT = G(st=12, nt=-14, afR=88, aaR=44, atR=-24, aeR=8)          # contact, mid-sweep
SWAT_END = G(st=34, nt=-28, afR=84, aaR=84, atR=-40, aeR=8)    # follow-through
# smite: floating above him, the right arm high, then brought straight down
SMITE_UP = G(HANG, sb=-4, nb=-4, afR=172, aaR=8, atR=0, aeR=24)
SMITE = G(HANG, pitch=6, sb=16, nb=10, afR=58, aaR=4, atR=0, aeR=4)
# the one-handed barrage on the spot: chest height and face height
PALM_HIT_HI = G(pitch=1, sb=3, st=12, nt=-10, afR=104, aaR=2, atR=-12, aeR=2)
# pushed back: slides upright, arms loose and a little out, knees soft, looking at him
SLIDE = G(back=False, pitch=6, sb=8, nb=-6, afR=14, aaR=26, atR=8, aeR=26, afL=14, aaL=26, atL=8, aeL=26,
          lfR=24, laR=8, ltR=12, lkR=40, lfL=10, laL=8, ltL=12, lkL=30)
# rising from the wall: straightens, shoulders back
RISE = G(back=False, pitch=4, sb=10, nb=-8, afR=8, aaR=14, atR=8, aeR=24, afL=8, aaL=14, atL=8, aeL=24)
# lifts off the ground, arms open, palms out: power coming up
ASCEND = G(HANG, back=False, sb=-6, nb=-10, afR=24, aaR=78, atR=-40, aeR=10, afL=24, aaL=78, atL=-40, aeL=10)
# hovering, charging: arms low and a little out, head down a touch, eyes up
CHARGE = G(HANG, back=False, sb=4, nb=10, afR=10, aaR=24, atR=6, aeR=18, afL=10, aaL=24, atL=6, aeL=18)
# the last strike (caught): one-handed, the left hand stays down
GOD_LOAD = G(back=False, pitch=4, sb=6, st=-16, nt=12, afR=40, aaR=12, atR=14, aeR=118, afL=-8, aaL=12, atL=8, aeL=20,
             lfR=6, lkR=20, lfL=10, lkL=18)
GOD_DRIVE = G(back=False, pitch=12, sb=10, st=24, nt=-20, nb=-4, afR=104, aaR=4, atR=4, aeR=4, afL=-14, aaL=14, atL=8, aeL=18,
              lfR=-8, laR=6, ltR=14, lkR=10, lfL=24, laL=5, ltL=8, lkL=26)
GOD_KNEEL = P(**{**KNEEL, 'afL': 6, 'aaL': 18, 'atL': 8, 'aeL': 30})

ALL = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, dict) and k not in ('Z', 'V3', 'BEHIND', 'SQUARE', 'HANG')}

# =====================================================================================================================
# v7b: after the first few strikes he stops toying with him and fights like Thragg: no boxer's guard, no bounce, no jabs.
# Upright, chest out, fists low and loose; every blow a full-body power strike (the other fist pulls back to the hip);
# wide haymakers; he flies in (body stretched out, legs trailing) for flying punches and knees; two-fisted hammers from above.
from keyanim import mirror as _mir
WIDE = dict(lfR=4, laR=8, ltR=14, lkR=16, lfL=16, laL=7, ltL=8, lkL=18)
WARRIOR = P(pitch=3, sb=4, st=-6, nb=0, nt=6, afR=12, aaR=16, atR=10, aeR=34, afL=16, aaL=14, atL=10, aeL=40, **WIDE)
WAR_FLOAT = P(**{**WARRIOR, **HANG})
# the power straight: the right fist cocked back at the ribs, the left reaching out low; then the fist goes through, the left pulls back to the hip
POWER_LOAD = P(pitch=6, sb=8, st=-30, nb=0, nt=24, afR=-26, aaR=18, atR=20, aeR=100, afL=40, aaL=10, atL=10, aeL=34,
               lfR=6, laR=8, ltR=16, lkR=28, lfL=20, laL=6, ltL=8, lkL=24)
POWER = P(pitch=14, sb=10, st=32, nb=0, nt=-26, afR=96, aaR=4, atR=4, aeR=2, afL=-26, aaL=16, atL=20, aeL=100,
          lfR=-10, laR=6, ltR=16, lkR=8, lfL=30, laL=5, ltL=6, lkL=30)
POWER_LOAD_L = _mir(POWER_LOAD)
POWER_L = _mir(POWER)
# the haymaker: the arm drawn back wide, then swung round flat across in front, the whole chest behind it
HAY_LOAD = P(pitch=4, sb=6, st=-40, nb=0, nt=30, afR=36, aaR=82, atR=-24, aeR=56, afL=18, aaL=12, atL=10, aeL=40,
             lfR=6, laR=8, ltR=16, lkR=26, lfL=18, laL=6, ltL=8, lkL=22)
HAY = P(pitch=10, sb=10, st=26, nb=0, nt=-22, afR=88, aaR=40, atR=24, aeR=58, afL=-18, aaL=20, atL=20, aeL=90,
        lfR=-6, laR=6, ltR=16, lkR=12, lfL=26, laL=5, ltL=6, lkL=28)
HAY_END = P(**{**HAY, 'st': 42, 'nt': -34, 'afR': 80, 'aaR': 6, 'atR': 30, 'aeR': 76})
HAY_STOP = P(**{**HAY, 'st': 22, 'aeR': 52})      # stopped dead by the barrier: the arm recoils
# flying in: stretched out flat, legs trailing; the fist cocked at the chest, then thrown straight ahead
TRAIL = dict(lfR=-10, laR=4, ltR=4, lkR=18, lfL=-2, laL=4, ltL=4, lkL=34)
FLY_LOAD = P(pitch=50, sb=4, st=-16, nb=-40, nt=12, afR=40, aaR=22, atR=24, aeR=118, afL=10, aaL=16, atL=10, aeL=24, **TRAIL)
FLY_HIT = P(pitch=54, sb=6, st=12, nb=-44, nt=-10, afR=146, aaR=4, atR=4, aeR=2, afL=-24, aaL=14, atL=10, aeL=20, **TRAIL)
# two fists from above, hanging in the air (legs loose, not a jump)
THAMMER_UP = P(**{**HAMMER_UP, **HANG, 'pitch': -4})
THAMMER_DOWN = P(**{**HAMMER_DOWN, **HANG, 'pitch': 18, 'lfR': 30, 'lkR': 40})
ALL = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, dict) and k not in ('Z', 'V3', 'BEHIND', 'SQUARE', 'HANG', 'WIDE', 'TRAIL')}
