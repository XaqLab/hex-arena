import os
from pathlib import Path

__version__ = '0.1.2'

_DATA_DIRS = [
    Path(__file__).parent.parent/'data',
    Path('/mnt/c/Projects.Data/foraging/from.Panos_Jun2024'),
]
DATA_DIR = next((d for d in _DATA_DIRS if os.path.exists(d)), None)
STORE_DIR = Path(__file__).parent.parent/'store'
