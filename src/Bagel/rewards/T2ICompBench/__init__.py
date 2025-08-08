# src/rewards/T2ICompBench/__init__.py

import os, sys

_root = os.path.dirname(__file__)

# 1) 让 BLIPvqa_eval/BLIP 下的模块可以被 import BLIP.*
blipvqa_root = os.path.join(_root, "BLIPvqa_eval")
blip_root     = os.path.join(blipvqa_root, "BLIP")
for p in (blipvqa_root, blip_root):
    if p not in sys.path:
        sys.path.insert(0, p)

# 2) 让 UniDet_eval/ 也能当 top-level
unidet_root = os.path.join(_root, "UniDet_eval")
if unidet_root not in sys.path:
    sys.path.insert(0, unidet_root)

# 3) 同理，CLIPScore_eval/
clipscore_root = os.path.join(_root, "CLIPScore_eval")
if clipscore_root not in sys.path:
    sys.path.insert(0, clipscore_root)
