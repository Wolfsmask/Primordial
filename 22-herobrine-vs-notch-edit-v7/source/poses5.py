"""v5 additions: the casual walk up and the lazy straight that stops dead, then looking at his own fist."""
from poses import P
from poses4 import *          # noqa: F401,F403  (all the v4 poses)
import poses4 as _P4

# walking, relaxed, chin down a little; the arms swing loose (keyed at each footfall: the arm opposite the forward foot comes forward)
WALK_L = P(pitch=2, sb=5, st=6, nb=8, nt=-4, afR=18, aaR=9, atR=6, aeR=18, afL=-14, aaL=9, atL=6, aeL=12,
           lfR=-12, laR=3, ltR=8, lkR=10, lfL=20, laL=3, ltL=8, lkL=8)
WALK_R = P(pitch=2, sb=5, st=-6, nb=8, nt=4, afR=-14, aaR=9, atR=6, aeR=12, afL=18, aaL=9, atL=6, aeL=18,
           lfR=20, laR=3, ltR=8, lkR=8, lfL=-12, laL=3, ltL=8, lkL=10)
# the lazy wind-up: no guard, the right hand drops back a touch, the chest turns back with it
CASUAL_LOAD = P(pitch=2, sb=5, st=-14, nb=6, nt=10, afR=-8, aaR=12, atR=10, aeR=58, afL=10, aaL=9, atL=6, aeL=16,
                lfR=-10, laR=3, ltR=10, lkR=12, lfL=16, laL=3, ltL=8, lkL=10)
# the casual straight: the arm just goes out straight, the other hand stays down, hardly any body in it
CASUAL_PUNCH = P(pitch=4, sb=4, st=16, nb=4, nt=-14, afR=96, aaR=4, atR=4, aeR=2, afL=-6, aaL=10, atL=6, aeL=18,
                 lfR=-12, laR=3, ltR=12, lkR=10, lfL=18, laL=3, ltL=8, lkL=14)
# leaning on it: pushing into the wall that isn't there
PRESS_IN = P(**{**CASUAL_PUNCH, 'pitch': 7, 'sb': 7, 'st': 20, 'nb': 0, 'afL': -10, 'aeL': 30, 'lkR': 16, 'lkL': 20})
# draws the fist back in front of his face and turns it over, looking at the knuckles
LOOK_FIST = P(pitch=2, sb=8, st=8, nb=10, nt=-6, afR=80, aaR=0, atR=30, aeR=90, afL=-2, aaL=10, atL=6, aeL=20,
              lfR=-8, laR=3, ltR=12, lkR=12, lfL=14, laL=3, ltL=8, lkL=14)
# looks back up at him: the hand drops, the shoulders set
SET = P(pitch=5, sb=10, st=4, nb=-6, nt=0, afR=30, aaR=14, atR=10, aeR=70, afL=12, aaL=12, atL=8, aeL=40,
        lfR=-6, laR=4, ltR=12, lkR=16, lfL=14, laL=4, ltL=8, lkL=18)
ALL = {k: v for k, v in dict(globals()).items() if k.isupper() and isinstance(v, dict) and k not in ('Z', 'V3')}
