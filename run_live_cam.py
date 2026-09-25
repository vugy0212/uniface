import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, os.path.join(CURRENT_DIR, "src"), PARENT_DIR, os.path.join(PARENT_DIR, "src")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import argparse
from src.live_cam import run_live_camera

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniFace Live Face Recognition")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or RTSP/HTTP URL")
    parser.add_argument("--threshold", type=float, default=0.45, help="Recognition cosine similarity threshold")
    parser.add_argument("--skip", type=int, default=2, help="Process every N frames")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device (CPU or CUDA)")
    parser.add_argument("--start", type=int, default=0, help="Start position in seconds for video/youtube (default: 0)")
    parser.add_argument("--log-events", action="store_true", default=False, help="Enable automatic detection event logging")
    parser.add_argument("--cooldown", type=int, default=30, help="Cooldown in seconds between re-logging same person (default: 30)")
    args = parser.parse_args()

    run_live_camera(
        camera_source=args.source,
        threshold=args.threshold,
        process_interval=args.skip,
        device=args.device,
        start_sec=args.start,
        log_events=args.log_events,
        cooldown_sec=args.cooldown
    )
