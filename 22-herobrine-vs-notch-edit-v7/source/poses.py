"""v3 pose library (channels of keyanim, degrees).  Right-handed versions; keyanim.mirror() gives the left."""
from keyanim import mirror

Z = dict(sb=0, st=0, ss=0, nb=0, nt=0, ns=0, pitch=0, roll=0,
         afR=0, aaR=6, atR=0, aeR=8, afL=0, aaL=6, atL=0, aeL=8,
         lfR=0, laR=3, ltR=6, lkR=2, lfL=0, laL=3, ltL=6, lkL=2)


def P(**kw):
    d = dict(Z); d.update(kw); return d


# --- Notch: floating, calm, arms loose, legs hanging a little bent, one knee higher
FLOAT = P(sb=4, nb=6, afR=6, aaR=12, atR=10, aeR=22, afL=8, aaL=12, atL=10, aeL=26,
          lfR=6, laR=4, lkR=22, lfL=-4, laL=6, lkL=12, ltR=4, ltL=4)
FLOAT_DOWN = P(sb=2, nb=24, afR=4, aaR=16, atR=8, aeR=18, afL=4, aaL=16, atL=8, aeL=18,     # descending, looking down at him
               lfR=6, laR=4, lkR=12, lfL=0, laL=5, lkL=6)
HAND_UP = P(sb=2, st=-14, nb=4, afR=86, aaR=4, atR=-10, aeR=12, afL=8, aaL=14, atL=10, aeL=28,   # palm out at him (right hand)
            lfR=14, laR=4, lkR=24, lfL=4, laL=6, lkL=14)
FINGER = P(sb=0, st=-20, nb=2, afR=92, aaR=0, atR=-20, aeR=4, afL=6, aaL=16, atL=10, aeL=24,     # a flick: arm straight, pointing
           lfR=12, laR=4, lkR=22, lfL=4, laL=6, lkL=14)
SKY_HAND = P(sb=-6, st=-8, nb=-14, afR=165, aaR=10, atR=0, aeR=14, afL=10, aaL=18, atL=10, aeL=24,
             lfR=12, laR=4, lkR=22, lfL=4, laL=6, lkL=14)
ARMS_OPEN = P(sb=-4, nb=2, afR=40, aaR=48, atR=-30, aeR=14, afL=40, aaL=48, atL=-30, aeL=14,
              lfR=10, laR=6, lkR=18, lfL=4, laL=8, lkL=10)

# --- Herobrine: a fighter's stance, punches, kicks
GUARD = P(pitch=6, sb=10, st=-14, nb=-6, nt=12, afR=58, aaR=18, atR=26, aeR=128, afL=72, aaL=16, atL=22, aeL=118,
          lfR=8, laR=6, lkR=26, lfL=18, laL=6, lkL=22, ltR=14, ltL=6)
GUARD_LOW = P(pitch=10, sb=16, st=-18, nb=-12, nt=16, afR=50, aaR=20, atR=26, aeR=124, afL=62, aaL=18, atL=22, aeL=112,
              lfR=20, laR=8, lkR=48, lfL=34, laL=8, lkL=44, ltR=16, ltL=8)
CROUCH = P(pitch=22, sb=26, nb=-22, afR=34, aaR=24, atR=16, aeR=70, afL=30, aaL=24, atL=16, aeL=64,
           lfR=74, laR=10, lkR=120, lfL=64, laL=12, lkL=112, ltR=14, ltL=14)
# a cross with the right: rear hand drives straight ahead, hips and chest turn into it
CROSS_LOAD = P(pitch=8, sb=10, st=-30, nb=-8, nt=26, afR=46, aaR=26, atR=30, aeR=136, afL=76, aaL=14, atL=20, aeL=112,
               lfR=10, laR=6, lkR=34, lfL=20, laL=6, lkL=26)
CROSS = P(pitch=14, sb=12, st=28, nb=-10, nt=-24, afR=92, aaR=4, atR=6, aeR=4, afL=60, aaL=22, atL=24, aeL=132,
          lfR=-6, laR=6, lkR=18, lfL=30, laL=6, lkL=30)
JAB = P(pitch=8, sb=8, st=-24, nb=-6, nt=20, afR=58, aaR=18, atR=26, aeR=130, afL=94, aaL=2, atL=8, aeL=4,
        lfR=8, laR=6, lkR=26, lfL=22, laL=6, lkL=24)
HOOK_L = P(pitch=8, sb=10, st=-40, nb=-6, nt=30, afR=60, aaR=18, atR=26, aeR=130, afL=86, aaL=72, atL=-12, aeL=92,
           lfR=10, laR=6, lkR=30, lfL=18, laL=6, lkL=28)
UPPER_R = P(pitch=4, sb=-6, st=22, nb=-14, nt=-16, afR=150, aaR=8, atR=10, aeR=70, afL=58, aaL=20, atL=24, aeL=128,
            lfR=-4, laR=6, lkR=12, lfL=24, laL=6, lkL=20)
BODY_L = P(pitch=18, sb=26, st=-26, nb=-16, nt=22, afR=64, aaR=18, atR=26, aeR=128, afL=76, aaL=6, atL=6, aeL=20,
           lfR=24, laR=8, lkR=56, lfL=40, laL=8, lkL=52)
PUSH2 = P(pitch=18, sb=14, nb=-10, afR=88, aaR=10, atR=4, aeR=28, afL=88, aaL=10, atL=4, aeL=28,   # both hands driving into the barrier
          lfR=-14, laR=6, lkR=20, lfL=40, laL=6, lkL=44)
# kicks (right leg)
KICK_CHAMBER = P(pitch=-6, sb=-4, st=-20, nb=-8, nt=24, afR=40, aaR=34, atR=20, aeR=110, afL=60, aaL=30, atL=20, aeL=100,
                 lfR=100, laR=24, lkR=140, ltR=10, lfL=6, laL=6, lkL=18)
ROUNDHOUSE = P(pitch=-4, roll=34, sb=2, st=-10, ss=16, nb=-6, nt=-60, afR=30, aaR=40, atR=10, aeR=90, afL=40, aaL=50, atL=20, aeL=96,   # side-on: the leg out to the right at head height
               lfR=14, laR=74, lkR=8, ltR=-10, lfL=0, laL=10, lkL=12)
FRONT_KICK = P(pitch=-16, sb=-6, nb=-12, afR=30, aaR=36, atR=16, aeR=100, afL=48, aaL=30, atL=16, aeL=104,
               lfR=104, laR=6, lkR=4, lfL=-4, laL=6, lkL=16)
AXE_UP = P(pitch=-14, sb=-10, nb=-20, afR=60, aaR=56, atR=10, aeR=40, afL=70, aaL=50, atL=10, aeL=40,
           lfR=128, laR=10, lkR=4, lfL=-10, laL=6, lkL=30)
AXE_DOWN = P(pitch=26, sb=24, nb=-6, afR=30, aaR=60, atR=10, aeR=40, afL=40, aaL=56, atL=10, aeL=40,
             lfR=70, laR=8, lkR=6, lfL=-20, laL=6, lkL=40)
BACKKICK = P(pitch=52, sb=18, st=10, nb=-40, nt=-50, afR=40, aaR=40, atR=20, aeR=90, afL=50, aaL=40, atL=20, aeL=100,
             lfR=-48, laR=8, lkR=8, lfL=48, laL=6, lkL=56)
# flying / thrown (body pitched back, limbs trail forward), a tuck for flips, a skid on landing
THROWN = P(pitch=-30, sb=-20, nb=34, afR=110, aaR=36, atR=-20, aeR=36, afL=100, aaL=40, atL=-20, aeL=46,
           lfR=56, laR=14, lkR=40, lfL=34, laL=16, lkL=60)
TUCK = P(pitch=10, sb=56, nb=30, afR=70, aaR=20, atR=10, aeR=120, afL=70, aaL=20, atL=10, aeL=120,
         lfR=120, laR=10, lkR=146, lfL=120, laL=10, lkL=146)
SKID = P(pitch=24, sb=30, st=-12, nb=-26, nt=14, afR=20, aaR=40, atR=10, aeR=40, afL=74, aaL=30, atL=10, aeL=10,
         lfR=86, laR=14, lkR=128, lfL=24, laL=22, lkL=76, ltR=16, ltL=22)
SUPERHERO = P(pitch=26, sb=34, nb=-34, nt=4, afR=-8, aaR=12, atR=10, aeR=8, afL=58, aaL=26, atL=20, aeL=120,
              lfR=96, laR=12, lkR=132, lfL=8, laL=20, lkL=96, ltR=10, ltL=20)
DASH = P(pitch=42, sb=16, nb=-34, afR=-40, aaR=14, atR=10, aeR=40, afL=-46, aaL=14, atL=10, aeL=34,     # ninja run lean, arms swept back
         lfR=60, laR=6, lkR=90, lfL=-20, laL=6, lkL=50)
FLY_PUNCH = P(pitch=68, sb=8, nb=-56, st=10, afR=172, aaR=2, atR=4, aeR=4, afL=-30, aaL=14, atL=10, aeL=30,   # Superman: fist first
              lfR=-6, laR=4, lkR=24, lfL=-14, laL=4, lkL=10)
ROAR = P(pitch=-6, sb=-18, nb=-32, afR=40, aaR=70, atR=-30, aeR=50, afL=40, aaL=70, atL=-30, aeL=50,
         lfR=14, laR=16, lkR=34, lfL=14, laL=16, lkL=34, ltR=18, ltL=18)
SUMMON = P(pitch=-4, sb=-10, nb=-30, afR=176, aaR=8, atR=0, aeR=6, afL=26, aaL=26, atL=10, aeL=50,
           lfR=10, laR=10, lkR=24, lfL=16, laL=10, lkL=28)
HUNCH = P(pitch=16, sb=30, nb=-24, ss=-6, afR=14, aaR=16, atR=10, aeR=30, afL=34, aaL=12, atL=10, aeL=70,
          lfR=24, laR=10, lkR=44, lfL=10, laL=8, lkL=24)
DOUBLED = P(pitch=30, sb=52, nb=30, afR=50, aaR=20, atR=10, aeR=90, afL=56, aaL=22, atL=10, aeL=100,      # winded over a knee
            lfR=30, laR=8, lkR=50, lfL=20, laL=8, lkL=40)
KNEE_UP = P(pitch=4, sb=10, nb=12, afR=40, aaR=40, atR=-10, aeR=70, afL=30, aaL=40, atL=-10, aeL=60,     # Notch's knee strike (right)
            lfR=104, laR=6, lkR=118, lfL=-6, laL=6, lkL=20)
BACKHAND = P(sb=4, st=40, nb=0, nt=-30, afR=84, aaR=86, atR=-40, aeR=10, afL=20, aaL=20, atL=10, aeL=40,
             lfR=14, laR=4, lkR=24, lfL=4, laL=6, lkL=14)
HANDS_IN = P(sb=-6, nb=10, afR=60, aaR=-8, atR=40, aeR=70, afL=60, aaL=-8, atL=40, aeL=70,           # gathering a sphere between the hands
             lfR=10, laR=6, lkR=18, lfL=4, laL=8, lkL=10)
POINT_DOWN = P(sb=12, nb=34, afR=128, aaR=0, atR=-10, aeR=4, afL=10, aaL=22, atL=10, aeL=30,          # pointing down at him from above
               lfR=12, laR=4, lkR=22, lfL=4, laL=6, lkL=14)

# --- v3 additions
STAND = P(sb=6, nb=14, afR=4, aaR=9, atR=6, aeR=16, afL=4, aaL=9, atL=6, aeL=16, lfR=2, lkR=5, lfL=2, lkL=5, ltR=10, ltL=10)
STAND_TENSE = P(pitch=4, sb=12, nb=-4, ss=-3, afR=10, aaR=14, atR=10, aeR=40, afL=12, aaL=14, atL=10, aeL=46,
                lfR=8, laR=6, lkR=16, lfL=8, laL=6, lkL=16, ltR=12, ltL=12)
CROSS_AIR = P(pitch=10, sb=10, st=26, nb=-12, nt=-22, afR=94, aaR=2, atR=6, aeR=2, afL=54, aaL=24, atL=24, aeL=130,
              lfR=-24, laR=6, lkR=56, lfL=52, laL=8, lkL=84)
X_BLOCK = P(pitch=6, sb=14, nb=-30, afR=150, aaR=-14, atR=40, aeR=70, afL=150, aaL=-14, atL=40, aeL=70,    # arms crossed overhead
            lfR=34, laR=20, lkR=62, lfL=34, laL=20, lkL=62, ltR=20, ltL=20)
HAMMER_UP = P(pitch=-12, sb=-16, nb=-26, afR=176, aaR=6, atR=10, aeR=40, afL=176, aaL=6, atL=10, aeL=40,
              lfR=40, laR=8, lkR=90, lfL=60, laL=8, lkL=110)
HAMMER_DOWN = P(pitch=30, sb=34, nb=-20, afR=96, aaR=-4, atR=30, aeR=20, afL=96, aaL=-4, atL=30, aeL=20,
                lfR=60, laR=8, lkR=100, lfL=40, laL=8, lkL=80)
PALM = P(sb=8, st=-24, nb=2, nt=18, afR=84, aaR=6, atR=-6, aeR=6, afL=-24, aaL=26, atL=10, aeL=50,             # Notch: palm strike with the right
         lfR=18, laR=4, lkR=30, lfL=-4, laL=6, lkL=12)
BACKHAND_LOAD = P(sb=2, st=-36, nb=0, nt=26, afR=60, aaR=-20, atR=50, aeR=100, afL=-30, aaL=30, atL=10, aeL=50,
                  lfR=14, laR=4, lkR=24, lfL=4, laL=6, lkL=14)
HIT_FACE = P(pitch=-14, sb=-22, ss=10, st=24, nb=-30, nt=-36, ns=20, afR=40, aaR=50, atR=10, aeR=50, afL=30, aaL=60, atL=10, aeL=40,
             lfR=20, laR=10, lkR=40, lfL=40, laL=14, lkL=60)
SPIN_FLY = P(pitch=0, sb=-10, ss=14, nb=-10, ns=14, afR=60, aaR=90, atR=-10, aeR=30, afL=40, aaL=110, atL=-10, aeL=20,
             lfR=30, laR=30, lkR=40, lfL=10, laL=36, lkL=20)
LOOK_UP = P(pitch=-4, sb=-10, nb=-40, afR=20, aaR=20, atR=10, aeR=60, afL=20, aaL=20, atL=10, aeL=60,
            lfR=14, laR=12, lkR=26, lfL=14, laL=12, lkL=26, ltR=14, ltL=14)
FLOAT_CHARGE = P(sb=-2, nb=4, afR=52, aaR=-2, atR=34, aeR=64, afL=52, aaL=-2, atL=34, aeL=64,
                 lfR=12, laR=4, lkR=22, lfL=4, laL=6, lkL=14)

ALL = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, dict) and k != 'Z'}
