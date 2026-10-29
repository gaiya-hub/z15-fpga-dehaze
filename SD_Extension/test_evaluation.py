from pathlib import Path
import unittest,math,json
import numpy as np
from PIL import Image
from evaluation import image_scores,evaluate_quality,auto_ground_truth,demo_pair
ROOT=Path(__file__).parent
FIXTURES=ROOT/'qa/metric_fixtures'

class QualityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FIXTURES.mkdir(parents=True,exist_ok=True)
        cls.clear=Image.new('RGB',(16,16),(128,128,128))
        cls.hazy=Image.new('RGB',(16,16),(130,130,130))
        cls.gt=FIXTURES/'clear.png';cls.source=FIXTURES/'hazy.png'
        cls.clear.save(cls.gt);cls.hazy.save(cls.source)

    def test_known_metrics_and_identity(self):
        scores=image_scores(self.clear,self.hazy)
        self.assertAlmostEqual(scores['mae'],2.0)
        self.assertAlmostEqual(scores['mse'],4.0)
        self.assertAlmostEqual(scores['psnr_db'],10*math.log10(255**2/4),places=10)
        expected=(2*128*130+(0.01*255)**2)/(128**2+130**2+(0.01*255)**2)
        self.assertAlmostEqual(scores['ssim'],expected,places=9)
        identical=image_scores(self.clear,self.clear)
        self.assertEqual(identical['psnr_db'],'Infinity')
        self.assertAlmostEqual(identical['ssim'],1.0)

    def test_paired_improvement_and_json(self):
        quality,clear=evaluate_quality(self.source,self.hazy,self.clear,self.gt)
        self.assertTrue(quality['available'])
        self.assertEqual(quality['fpga']['psnr_db'],'Infinity')
        self.assertLess(quality['delta']['mae'],0)
        json.dumps(quality,allow_nan=False)

    def test_missing_gt_does_not_invent_metrics(self):
        quality,clear=evaluate_quality(self.source,self.hazy,self.clear,None)
        self.assertFalse(quality['available']);self.assertIsNone(clear)
        self.assertNotIn('fpga',quality)

    def test_different_original_geometry_rejected(self):
        path=FIXTURES/'wrong_geometry.png';Image.new('RGB',(17,16)).save(path)
        quality,clear=evaluate_quality(self.source,self.hazy,self.clear,path)
        self.assertFalse(quality['available']);self.assertIsNone(clear)
        self.assertIn('尺寸不一致',quality['reason'])

    def test_same_file_is_not_accepted_as_gt(self):
        quality,clear=evaluate_quality(self.source,self.hazy,self.clear,self.source)
        self.assertFalse(quality['available'])

    def test_known_dataset_pairing(self):
        root=FIXTURES/'pair_layout'
        for folder in ['natural_hazy','natural_gt','synthetic_hazy','synthetic_gt','output_images']:
            (root/folder).mkdir(parents=True,exist_ok=True)
        natural=root/'natural_hazy/scene_0.8_0.2.jpg'
        gt=root/'natural_gt/scene_0.8_0.2.jpg'
        self.hazy.save(natural);self.clear.save(gt)
        self.assertEqual(auto_ground_truth(natural),gt)
        synthetic=root/'synthetic_hazy/frame_hazy.jpg';clear=root/'synthetic_gt/frame.jpg'
        self.hazy.save(synthetic);self.clear.save(clear)
        self.assertEqual(auto_ground_truth(synthetic),clear)
        self.assertIsNone(auto_ground_truth(root/'output_images/frame.jpg'))

    def test_small_image_has_no_fabricated_ssim(self):
        scores=image_scores(Image.new('RGB',(1,1),(128,128,128)),Image.new('RGB',(1,1),(130,130,130)))
        self.assertIsNone(scores['ssim'])
        self.assertAlmostEqual(scores['mae'],2.0)

if __name__=='__main__':unittest.main(verbosity=2)
