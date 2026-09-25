import os
import sys

APP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
SRC_DIR = os.path.join(APP_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.live_cam import run_live_camera

if __name__ == "__main__":
    run_live_camera(camera_idx=0, threshold=0.45, process_interval=2)
