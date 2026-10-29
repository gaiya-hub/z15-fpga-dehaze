"""Offline protocol fixtures only: this suite is NOT actual SD/FPGA evidence."""
from pathlib import Path
import hashlib,json,shutil,sys,tempfile,unittest
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import sd_tasks as sd
from PIL import Image
import numpy as np
QA=ROOT/'qa/protocol_fixtures';QA.mkdir(parents=True,exist_ok=True)
class Tasks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=QA);self.root=Path(self.temp.name)
        assert self.root.resolve().is_relative_to(QA.resolve())
        self.source=self.root/'images';self.source.mkdir()
        rng=np.random.default_rng(728)
        for w,h in [(37,29),(37,32)]:Image.fromarray(rng.integers(20,245,(h,w,3),dtype=np.uint8)).save(self.source/f'{w}x{h}.png')
        self.target=self.root/'card';self.target.mkdir()
        self.task,self.manifest=sd.export_task(self.source,self.target)
    def tearDown(self):self.temp.cleanup()
    def fixture(self):
        for job in self.manifest['jobs']:
            head,regs,params,pixels=sd.read_input(self.task/'IN'/job['input_file']);w,h=job['processing_size'];n=w*h
            output,_=sd.get_model().process(sd.unpack(pixels,w,h));raw=sd.pack(output)
            r=[0]*128;r[:15]=[sd.RESULT_MAGIC,1,0,job['index'],w,h,n,self.manifest['model_tag'],head[5],head[6],head[7],sd.crc(raw),1000000,1,0xfeed0001]
            m=[0]*48;m[0]=0x600d600d;m[1]=w;m[2]=h;m[18]=15000;m[20]=10000;m[23]=1000000;m[24]=n;m[25]=2000;m[27]=4000;m[29]=9000;m[32]=8000
            m[31]=self.manifest['model_tag'];m[34]=0x504c5633;m[36]=1;m[37]=int(n%8==0 and h>=16);m[38]=16 if m[37] else 0;m[46]=0x4455414c;m[47]=1
            r[16:64]=m;r[74]=int(job['index']==1);r[75]=(1+r[74]) if m[37] else 0;r[77]=self.manifest['index_crc32'];r[64]=300;r[66]=400;r[68]=50;r[70]=20000;r[72]=15000
            (self.task/'OUT'/f'O{job["index"]:07d}.BIN').write_bytes(raw)
            (self.task/'OUT'/f'R{job["index"]:07d}.BIN').write_bytes(sd.sealed(r))
        (self.task/'SIMULATED_FIXTURE_DO_NOT_USE_AS_BOARD_PROOF.txt').write_text('Golden output synthesized for offline parser testing; no card or FPGA used.',encoding='utf-8')
        summary=[0]*64;summary[:10]=[sd.BATCH_MAGIC,1,len(self.manifest['jobs']),0,self.manifest['model_tag'],1000000,0xfeed0001,self.manifest['index_crc32'],123456,0]
        (self.task/'OUT/BATCH.BIN').write_bytes(sd.sealed(summary))
    def result_path(self):return self.task/'OUT/R0000001.BIN'
    def alter_result(self,slot):
        p=self.result_path();r=sd.words(p.read_bytes());r[slot]^=1;p.write_bytes(sd.sealed(r))
    def test_export_pixel_bytes_and_no_overwrite(self):
        for job in self.manifest['jobs']:
            h,regs,params,pixels=sd.read_input(self.task/'IN'/job['input_file'])
            with Image.open(job['source']) as image:expected=sd.pack(image.convert('RGB'))
            self.assertEqual(pixels,expected);self.assertEqual(h[4],regs[1]*regs[2]);self.assertEqual(h[8],params[22])
        with self.assertRaises(ValueError):sd.export_task(self.source,self.target)
    def test_input_corruption_rejected(self):
        p=self.task/'IN'/self.manifest['jobs'][0]['input_file'];data=bytearray(p.read_bytes());data[-1]^=1;p.write_bytes(data)
        with self.assertRaises(ValueError):sd.read_input(p)
    def test_simulated_roundtrip_preserves_raw_output_and_reference(self):
        self.fixture();dest,report=sd.import_task(self.task,self.root/'imported')
        self.assertEqual(len(report['records']),2)
        self.assertTrue(report['board_batch_completed']);self.assertEqual(report['batch_summary']['board_before_final_summary_ms'],123.456)
        for record in report['records']:
            r=record['result'];self.assertEqual(r['host_channel_mismatches'],0);self.assertIsNone(r['upload_ms']);self.assertEqual(r['transport'],'sd-card')
            i=int(Path(record['run']).name);self.assertEqual((Path(record['run'])/'fpga.bin').read_bytes(),(self.task/'OUT'/f'O{i:07d}.BIN').read_bytes())
    def test_partial_batch_reports_missing(self):
        dest,report=sd.import_task(self.task,self.root/'partial');self.assertEqual(report['missing_jobs'],[1,2]);self.assertEqual(report['records'],[])
    def test_result_crc_rejected(self):
        self.fixture();p=self.result_path();b=bytearray(p.read_bytes());b[50]^=1;p.write_bytes(b)
        with self.assertRaises(ValueError):sd.import_task(self.task,self.root/'bad')
    def test_output_crc_rejected(self):
        self.fixture();p=self.task/'OUT/O0000001.BIN';b=bytearray(p.read_bytes());b[1]^=1;p.write_bytes(b)
        with self.assertRaises(ValueError):sd.import_task(self.task,self.root/'bad')
    def test_wrong_task_even_with_valid_crc_rejected(self):
        self.fixture();self.alter_result(77)
        with self.assertRaises(ValueError):sd.import_task(self.task,self.root/'bad')
    def test_wrong_worker_job_count_rejected(self):
        self.fixture();self.alter_result(75)
        with self.assertRaises(ValueError):sd.import_task(self.task,self.root/'bad')
    def test_wrong_model_rejected(self):
        self.fixture();self.alter_result(7)
        with self.assertRaises(ValueError):sd.import_task(self.task,self.root/'bad')
    def test_wrong_batch_end_record_rejected(self):
        self.fixture();p=self.task/'OUT/BATCH.BIN';r=sd.words(p.read_bytes());r[7]^=1;p.write_bytes(sd.sealed(r))
        with self.assertRaises(ValueError):sd.import_task(self.task,self.root/'bad')
    def test_discontinuous_batch_timer_is_not_reported_as_valid_speed(self):
        self.fixture();p=self.task/'OUT/BATCH.BIN';r=sd.words(p.read_bytes());r[8]=1;r[9]=0;p.write_bytes(sd.sealed(r))
        _,report=sd.import_task(self.task,self.root/'bad_timer')
        self.assertTrue(report['board_batch_completed'])
        self.assertFalse(report['batch_summary']['timing_valid'])
        self.assertIsNone(report['batch_summary']['board_before_final_summary_ms'])
if __name__=='__main__':unittest.main()
