from pathlib import Path
import tempfile,unittest,threading,json,hashlib
from PIL import Image
from batch_dehaze import ROOT,run_batch,digest

class BatchTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'qa').mkdir(exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=ROOT/'qa');self.root=Path(self.temp.name)
        self.src=self.root/'input';self.src.mkdir();self.out=self.root/'output'
        Image.new('RGB',(32,24),(130,150,170)).save(self.src/'scene.png')
        Image.new('RGB',(32,24),(140,160,180)).save(self.src/'scene.jpg')
    def tearDown(self):self.temp.cleanup()
    def test_outputs_resume_corruption_and_originals(self):
        original={p.name:digest(p) for p in self.src.iterdir()}
        (self.src/'broken.png').write_bytes(b'invalid image')
        result=run_batch(self.src,self.out,max_edge=640)
        self.assertEqual((result['succeeded'],result['failed']),(2,1))
        self.assertEqual(len(list((self.out/'images').glob('*.png'))),2)
        for p in self.src.iterdir():
            if p.name in original:self.assertEqual(digest(p),original[p.name])
        result=run_batch(self.src,self.out,max_edge=640)
        self.assertEqual((result['succeeded'],result['skipped'],result['failed']),(0,2,1))
        (self.out/'images/scene.jpg.dehazed.png').write_bytes(b'corrupt output')
        result=run_batch(self.src,self.out,max_edge=640)
        self.assertEqual((result['succeeded'],result['skipped']),(1,1))
        record=json.loads((self.out/'records/scene.png.json').read_text(encoding='utf-8'))
        self.assertFalse(record['result']['quality']['available'])
        self.assertEqual(record['result']['execution'],'PC software; not FPGA output')
        self.assertIsNone(record['result'].get('fpga_ms'))
    def test_stop_then_resume(self):
        stop=threading.Event()
        result=run_batch(self.src,self.out,max_edge=640,stop=stop,progress=lambda state:stop.set() if state['processed']==1 else None)
        self.assertEqual((result['status'],result['succeeded']),('stopped',1))
        result=run_batch(self.src,self.out,max_edge=640)
        self.assertEqual((result['status'],result['succeeded'],result['skipped']),('completed',1,1))
    def test_gt_recursion_and_config_guard(self):
        nested=self.src/'sub';nested.mkdir();Image.new('RGB',(24,16),(100,120,140)).save(nested/'x_hazy.png')
        gt=self.root/'gt/sub';gt.mkdir(parents=True);Image.new('RGB',(24,16),(80,110,135)).save(gt/'x.png')
        result=run_batch(self.src,self.out,max_edge=640,ground_truth_dir=gt.parent)
        self.assertEqual(result['succeeded'],3)
        record=json.loads((self.out/'records/sub/x_hazy.png.json').read_text(encoding='utf-8'))
        self.assertTrue(record['result']['quality']['available']);self.assertIn('output',record['result']['quality']);self.assertNotIn('fpga',record['result']['quality'])
        with self.assertRaises(ValueError):run_batch(self.src,self.out,max_edge=1280,ground_truth_dir=gt.parent)
    def test_reject_nested_or_unowned_output(self):
        with self.assertRaises(ValueError):run_batch(self.src,self.src/'restored')
        other=self.root/'existing';other.mkdir();(other/'keep.txt').write_text('keep')
        with self.assertRaises(ValueError):run_batch(self.src,other)
        self.assertEqual((other/'keep.txt').read_text(),'keep')
    def test_same_output_cannot_run_twice(self):
        attempted=[]
        def progress(state):
            if not attempted:
                attempted.append(True)
                with self.assertRaises(RuntimeError):run_batch(self.src,self.out,max_edge=640)
        result=run_batch(self.src,self.out,max_edge=640,progress=progress)
        self.assertEqual(result['succeeded'],2)

if __name__=='__main__':unittest.main()
