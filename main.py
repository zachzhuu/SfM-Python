import os
from sfm import SfM

image_dir = './Fountain_Comp'
intrinsics_path = os.path.join(image_dir, 'K.txt')
use_ba = False

sfm = SfM(image_dir, intrinsics_path)
sfm.detect_features()
sfm.match_features()
sfm.filter_matches()
sfm.draw_matches()
sfm.reconstruct(use_ba=use_ba)
