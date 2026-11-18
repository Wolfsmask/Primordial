"""v4 pose library: Herobrine fights light and upright (orthodox: left foot and left hand lead), every strike is short and fast.
Channels as in keyanim (degrees).  Notch keeps his v3 floating poses, plus the catch."""
from poses import P, Z, ALL as V3
import poses as P3

# ---------------------------------------------------------------- Herobrine
# stance: chin down, rear hand on the chin, lead hand a little forward, chest turned a little side-on, knees soft
GUARD = P(pitch=3, sb=7, st=-16, nb=8, nt=14,
          afR=46, aaR=8, atR=36, aeR=138, afL=60, aaL=6, atL=28, aeL=124,
          lfR=4, laR=6, ltR=20, lkR=18, lfL=18, laL=5, ltL=6, lkL=20)
# jab: the lead arm snaps straight at the target, the shoulder rolls up to cover the chin, rear hand stays home
JAB = P(pitch=6, sb=7, st=-30, nb=8, nt=26,
        afR=46, aaR=8, atR=36, aeR=140, afL=104, aaL=6, atL=4, aeL=0,
        lfR=2, laR=6, ltR=20, lkR=16, lfL=24, laL=5, ltL=6, lkL=24)
# straight right: hips and chest turn into it, rear arm straight, lead hand pulled back to the chin
LOAD_R = P(pitch=4, sb=8, st=-26, nb=8, nt=24,
           afR=40, aaR=10, atR=38, aeR=142, afL=62, aaL=6, atL=28, aeL=120,
           lfR=6, laR=6, ltR=20, lkR=24, lfL=16, laL=5, ltL=6, lkL=22)
STRAIGHT = P(pitch=8, sb=8, st=30, nb=6, nt=-28,
             afR=104, aaR=6, atR=4, aeR=0, afL=52, aaL=10, atL=34, aeL=140,
             lfR=-8, laR=6, ltR=16, lkR=8, lfL=26, laL=5, ltL=6, lkL=28)
# the same thrown in the air (after a teleport): legs gathered under him
AIR_LOAD = P(**{**LOAD_R, 'pitch': 0, 'lfR': 30, 'lkR': 70, 'lfL': 50, 'lkL': 90})
AIR_STRAIGHT = P(**{**STRAIGHT, 'pitch': 6, 'lfR': -14, 'lkR': 60, 'lfL': 46, 'lkL': 88})
# right elbow, horizontal: chest whips round, the forearm folds flat across the face, the point of the elbow leads
ELBOW_LOAD = P(pitch=3, sb=6, st=-28, nb=8, nt=24, afR=60, aaR=40, atR=40, aeR=140, afL=60, aaL=6, atL=28, aeL=124,
               lfR=6, laR=6, ltR=20, lkR=22, lfL=18, laL=5, ltL=6, lkL=22)
ELBOW = P(pitch=6, sb=8, st=40, nb=6, nt=-36, afR=88, aaR=62, atR=76, aeR=150, afL=50, aaL=10, atL=34, aeL=140,
          lfR=-6, laR=6, ltR=16, lkR=10, lfL=24, laL=5, ltL=6, lkL=26)
AIR_ELBOW_LOAD = P(**{**ELBOW_LOAD, 'lfR': 34, 'lkR': 80, 'lfL': 50, 'lkL': 92})
AIR_ELBOW = P(**{**ELBOW, 'lfR': -10, 'lkR': 64, 'lfL': 44, 'lkL': 90})
# jumping right knee: the knee drives up and through, the body stays tall, arms snap down and back for the drive
KNEE_LOAD = P(pitch=0, sb=6, st=8, nb=6, afR=-6, aaR=20, atR=12, aeR=64, afL=-4, aaL=20, atL=12, aeL=60,
              lfR=84, laR=6, lkR=136, lfL=-14, laL=6, lkL=50)
KNEE = P(pitch=-8, sb=2, st=10, nb=14, afR=-24, aaR=20, atR=10, aeR=50, afL=-20, aaL=20, atL=10, aeL=46,
         lfR=118, laR=8, lkR=146, lfL=-26, laL=6, lkL=56)
# right snap kick (front kick): chamber, snap the shin out, ball of the foot in, re-chamber
SNAP_CHAMBER = P(pitch=-6, sb=2, st=4, nb=10, afR=40, aaR=24, atR=24, aeR=118, afL=56, aaL=14, atL=26, aeL=122,
                 lfR=92, laR=4, lkR=118, lfL=-2, laL=5, lkL=16)
SNAP = P(pitch=-18, sb=-4, st=6, nb=16, afR=36, aaR=30, atR=24, aeR=110, afL=54, aaL=18, atL=26, aeL=118,
         lfR=96, laR=4, lkR=4, lfL=-6, laL=5, lkL=14)
# hammer fists from above (after a teleport): both fists high, then straight down
HAMMER_UP = P(pitch=-6, sb=-10, nb=-6, afR=170, aaR=8, atR=10, aeR=50, afL=170, aaL=8, atL=10, aeL=50,
              lfR=30, laR=8, lkR=80, lfL=46, laL=8, lkL=96)
HAMMER_DOWN = P(pitch=22, sb=26, nb=8, afR=100, aaR=-2, atR=26, aeR=16, afL=100, aaL=-2, atL=26, aeL=16,
                lfR=56, laR=8, lkR=96, lfL=36, laL=8, lkL=76)
# moving at speed between strikes (flash steps are hidden, this is the pose either side of them)
DART = P(pitch=30, sb=12, st=-8, nb=-18, afR=-30, aaR=14, atR=10, aeR=70, afL=60, aaL=10, atL=24, aeL=110,
         lfR=50, laR=6, lkR=70, lfL=-14, laL=6, lkL=40)
# reactions: the push sends him sliding back on his feet (one hand down); the palm folds him; the backhand turns his head
SKID = P(pitch=22, sb=26, st=-10, nb=-22, nt=10, afR=30, aaR=34, atR=14, aeR=60, afL=72, aaL=26, atL=10, aeL=8,
         lfR=80, laR=12, ltR=16, lkR=120, lfL=24, laL=20, ltL=20, lkL=70)
DOUBLED = P(pitch=24, sb=44, nb=26, afR=50, aaR=18, atR=10, aeR=96, afL=56, aaL=20, atL=10, aeL=104,
            lfR=28, laR=8, lkR=46, lfL=20, laL=8, lkL=40)
HEAD_SNAP = P(pitch=-6, sb=-10, st=24, ss=12, nb=-16, nt=-50, ns=22, afR=30, aaR=40, atR=10, aeR=60, afL=40, aaL=46, atL=10, aeL=70,
              lfR=14, laR=10, lkR=30, lfL=30, laL=12, lkL=46)
SIDE_SKID = P(pitch=14, roll=-16, sb=20, ss=-14, nb=-14, nt=20, afR=20, aaR=60, atR=10, aeR=10, afL=50, aaL=20, atL=20, aeL=110,
              lfR=40, laR=40, ltR=10, lkR=80, lfL=50, laL=8, ltL=10, lkL=100)
THROWN = P(pitch=-24, sb=-16, nb=30, afR=100, aaR=30, atR=-10, aeR=40, afL=94, aaL=34, atL=-10, aeL=46,
           lfR=56, laR=12, lkR=50, lfL=36, laL=14, lkL=64)
SUPERHERO = P3.SUPERHERO
HUNCH = P3.HUNCH
STAND = P3.STAND
STAND_TENSE = P3.STAND_TENSE
SUMMON = P3.SUMMON
ROAR = P3.ROAR
CROUCH = P(pitch=14, sb=20, nb=-18, afR=40, aaR=22, atR=20, aeR=100, afL=40, aaL=22, atL=20, aeL=96,
           lfR=60, laR=10, lkR=104, lfL=54, laL=10, lkL=96, ltR=14, ltL=14)
# the end: driving the last straight into Notch's hand, then forced down onto his right knee, still pushing up
DRIVE = P(**{**STRAIGHT, 'pitch': 14, 'sb': 12, 'nb': -4, 'aeR': 4})
KNEEL = P(pitch=8, sb=14, st=20, nb=-24, nt=-20, afR=110, aaR=12, atR=4, aeR=10, afL=50, aaL=14, atL=30, aeL=130,
          lfR=-10, laR=6, ltR=10, lkR=120, lfL=80, laL=6, ltL=6, lkL=96)

# ---------------------------------------------------------------- Notch
FLOAT = P3.FLOAT
FLOAT_DOWN = P3.FLOAT_DOWN
HAND_UP = P3.HAND_UP
FINGER = P3.FINGER
PALM = P3.PALM
BACKHAND_LOAD = P3.BACKHAND_LOAD
BACKHAND = P3.BACKHAND
ARMS_OPEN = P3.ARMS_OPEN
# calm: hands loose, chin level (the breakdown)
FLOAT_CALM = P(sb=2, nb=2, afR=2, aaR=10, atR=8, aeR=14, afL=4, aaL=10, atL=8, aeL=16,
               lfR=6, laR=4, lkR=20, lfL=-4, laL=6, lkL=12, ltR=4, ltL=4)
# the catch: right hand up and open in front of his face, elbow soft
CATCH = P(sb=0, st=12, nb=2, afR=84, aaR=6, atR=0, aeR=16, afL=6, aaL=12, atL=10, aeL=24,
          lfR=8, laR=4, lkR=22, lfL=-2, laL=6, lkL=12)
PRESS = P(sb=10, st=14, nb=18, afR=50, aaR=6, atR=0, aeR=8, afL=6, aaL=12, atL=10, aeL=24,
          lfR=8, laR=4, lkR=22, lfL=-2, laL=6, lkL=12)

ALL = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, dict) and k not in ('Z', 'V3')}
