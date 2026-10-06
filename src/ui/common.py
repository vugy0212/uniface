import os
import sys
import config
from image_utils import imread_unicode, imwrite_unicode, save_image_dedup

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
SNAPSHOTS_DIR = config.get_snapshot_dir()

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
