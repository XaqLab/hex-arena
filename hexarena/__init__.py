import os
from pathlib import Path

__version__ = '0.1.2'

_data_dirs = [
    Path(__file__).parent.parent/'data',
    Path('/mnt/c/Projects.Data/foraging/from.Panos_Jun2024'),
    Path('/mnt/d/Projects.Data/foraging/from.Panos_Jun2024'),
]
DATA_DIR = next((d for d in _data_dirs if os.path.exists(d)), None)
_store_dirs = [
    Path(__file__).parent.parent/'store',
    Path('/mnt/d/foraging'),
]
STORE_DIR = next((d for d in _store_dirs if os.path.exists(d)), _store_dirs[0])
