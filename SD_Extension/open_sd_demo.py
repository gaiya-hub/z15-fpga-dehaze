from pathlib import Path
import argparse,json
from app import Demo
from sd_ui import open_window
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--latest',action='store_true');args=parser.parse_args()
app=Demo();app.title('高清彩色去雾 · SD卡批量版（双核）')
initial=None
if args.latest:
    reports=sorted((ROOT/'outputs').glob('sd_import_*/sd_import.json'),key=lambda p:p.stat().st_mtime,reverse=True)
    for path in reports:
        try:
            report=json.loads(path.read_text(encoding='utf-8'))
            if report.get('verified') and report.get('board_batch_completed') and (report.get('batch_summary') or {}).get('timing_valid'):
                initial=(path.parent,report);break
        except (OSError,ValueError):continue
app.after(150,lambda:open_window(app,initial_import=initial));app.mainloop()
