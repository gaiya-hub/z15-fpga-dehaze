from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import errno,sys,unittest
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(Path(__file__).parent));sys.path.insert(0,str(ROOT));sys.dont_write_bytecode=True
import sd_tasks as sd
import test_sd_tasks as fixtures
from PIL import Image
class Preflight(fixtures.Tasks):
    def test_no_space_writes_nothing_and_does_not_run_model(self):
        dest=self.root/'capacity_case';dest.mkdir();(dest/'keep.txt').write_text('keep',encoding='ascii')
        with patch('shutil.disk_usage',return_value=SimpleNamespace(free=100,total=1000,used=900)),patch.object(sd,'get_model',side_effect=AssertionError('Model ran before capacity guard')):
            with self.assertRaisesRegex(ValueError,'空间不足'):sd.export_task(self.source,dest)
        self.assertFalse((dest/'DEHAZE').exists());self.assertEqual((dest/'keep.txt').read_text(),'keep')
    def test_card_subfolder_is_rejected_before_writing(self):
        dest=self.root/'wrong_card_folder';dest.mkdir()
        with patch.object(sd,'_is_removable',return_value=True):
            with self.assertRaisesRegex(ValueError,'根目录'):sd.export_task(self.source,dest)
        self.assertFalse((dest/'DEHAZE').exists())
    def test_runtime_disk_full_rolls_back_own_files_only(self):
        dest=self.root/'io_failure';dest.mkdir();(dest/'keep.txt').write_text('old',encoding='ascii');pack=sd.pack;calls=[]
        def fail_second(image):
            calls.append(1)
            if len(calls)==2:raise OSError(errno.ENOSPC,'simulated storage change')
            return pack(image)
        with patch.object(sd,'pack',side_effect=fail_second):
            with self.assertRaisesRegex(ValueError,'半成品已撤销'):sd.export_task(self.source,dest)
        self.assertFalse((dest/'DEHAZE').exists());self.assertEqual((dest/'keep.txt').read_text(),'old')
    def test_exif_rotation_plan_matches_actual_payload(self):
        source=self.root/'rotated';source.mkdir();image=Image.new('RGB',(96,32),(100,140,180));exif=image.getexif();exif[274]=6;image.save(source/'rotated.jpg',exif=exif)
        dest=self.root/'rotated_task';dest.mkdir();task,manifest=sd.export_task(source,dest,max_edge=64)
        h,_,_,_=sd.read_input(task/'IN/I0000001.BIN');self.assertEqual(h[2:4],[21,64]);self.assertEqual(manifest['jobs'][0]['processing_size'],[21,64])
if __name__=='__main__':unittest.main()
