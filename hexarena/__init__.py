import os
import yaml
from pathlib import Path

__version__ = '0.1.2'

def _update_pths(dir_strs: list[str]) -> list[Path]:
    dir_pths = []
    for s in dir_strs:
        p = Path(s)
        dir_pths.append(p if p.is_absolute() else Path(__file__).parent.parent/p)
    return dir_pths

with open(Path(__file__).parent/'dirs.yaml', 'r') as f:
    _dirs = yaml.safe_load(f)
_data_dirs = _update_pths(_dirs['data_dirs'])
DATA_DIR = next((d for d in _data_dirs if os.path.exists(d)), None)
_store_dirs = _update_pths(_dirs['store_dirs'])
STORE_DIR = next((d for d in _store_dirs if os.path.exists(d)), _store_dirs[0])
