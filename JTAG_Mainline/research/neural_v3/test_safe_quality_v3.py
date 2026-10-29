import unittest
import numpy as np
from PIL import Image
from safe_detail_v3 import protect

CONFIG={'dark_floor':.25,'bright_headroom':.25,'blend':.75,'max_delta':64}
class GuardContract(unittest.TestCase):
    def test_identical_candidate_is_identity(self):
        a=np.random.default_rng(31).integers(0,256,(53,47,3),dtype=np.uint8);pic=Image.fromarray(a)
        np.testing.assert_array_equal(np.asarray(protect(pic,pic,CONFIG)),a)
    def test_extreme_predictions_preserve_range_and_limit(self):
        a=np.tile(np.arange(256,dtype=np.uint8)[None,:,None],(16,1,3));base=Image.fromarray(a)
        for level in [0,255]:
            out=np.asarray(protect(base,Image.fromarray(np.full_like(a,level)),CONFIG),dtype=np.float64)
            self.assertLessEqual(np.max(np.abs(out-a)),48.501)
            self.assertTrue(np.all(out>=a*.4375-.501));self.assertTrue(np.all(out<=255-(255-a)*.4375+.501))
            self.assertTrue(np.array_equal(out[:,:,0],out[:,:,1]))
    def test_mismatched_geometry_rejected(self):
        with self.assertRaises(ValueError):protect(Image.new('RGB',(64,64)),Image.new('RGB',(32,64)),CONFIG)
    def test_invalid_guard_rejected(self):
        with self.assertRaises(ValueError):protect(Image.new('RGB',(64,64)),Image.new('RGB',(64,64)),{**CONFIG,'blend':2})
if __name__=='__main__':unittest.main()
