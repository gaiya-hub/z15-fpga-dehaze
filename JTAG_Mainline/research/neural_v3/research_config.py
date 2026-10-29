from pathlib import Path
import os,json
def read_manifest(root):
    raw=os.environ.get('DEHAZE_DATASET')
    if not raw:raise RuntimeError('Set DEHAZE_DATASET to the folder containing natural_hazy/natural_gt/synthetic_hazy/synthetic_gt')
    dataset=Path(raw).resolve()
    rows=json.loads((Path(__file__).parent/'dataset_manifest.json').read_text(encoding='utf-8'))['manifest']
    for row in rows:
        for key in ['source','gt']:row[key]=str(dataset/row[key])
    return rows
