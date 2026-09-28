import os
import sys
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, os.path.join(CURRENT_DIR, "src"), PARENT_DIR, os.path.join(PARENT_DIR, "src")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from src.multicam import run_multicam_grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ArgusFace Multi-Camera 2x2 Grid Live Recognition")
    
    # Camera 1
    parser.add_argument("--cam1", type=str, default="0", help="Camera 1 source (USB index 0, 1 or RTSP URL)")
    parser.add_argument("--name1", type=str, default="USB Web Kamera", help="Camera 1 label")
    parser.add_argument("--off1", action="store_true", default=False, help="Disable Camera 1")
    
    # Camera 2
    parser.add_argument("--cam2", type=str, default="rtsp://admin:admin@192.168.50.236:554/11", help="Camera 2 source")
    parser.add_argument("--name2", type=str, default="Denver IP Kamera", help="Camera 2 label")
    parser.add_argument("--off2", action="store_true", default=False, help="Disable Camera 2")

    # Camera 3
    parser.add_argument("--cam3", type=str, default="", help="Camera 3 source (optional)")
    parser.add_argument("--name3", type=str, default="Kamera 3", help="Camera 3 label")
    parser.add_argument("--off3", action="store_true", default=False, help="Disable Camera 3")

    # Camera 4
    parser.add_argument("--cam4", type=str, default="", help="Camera 4 source (optional)")
    parser.add_argument("--name4", type=str, default="Kamera 4", help="Camera 4 label")
    parser.add_argument("--off4", action="store_true", default=False, help="Disable Camera 4")

    # Global options
    parser.add_argument("--threshold", type=float, default=0.45, help="Face recognition threshold (default: 0.45)")
    parser.add_argument("--skip", type=int, default=3, help="Process every N frames (default: 3)")
    parser.add_argument("--log-events", action="store_true", default=False, help="Enable automatic event logging into database")
    parser.add_argument("--cooldown", type=int, default=30, help="Event logging cooldown in seconds (default: 30)")
    parser.add_argument("--device", type=str, default="AUTO", help="Inference device: AUTO, DIRECTML, CUDA, OPENVINO, or CPU (default: AUTO)")
    parser.add_argument("--record-nvr", action="store_true", default=False, help="Enable continuous NVR MP4 segment recording")
    parser.add_argument("--segment-min", type=int, default=5, help="Duration of each MP4 video segment in minutes (default: 5)")
    parser.add_argument("--max-gb", type=float, default=20.0, help="Maximum disk storage quota in GB for FIFO cleanup (default: 20)")

    args = parser.parse_args()

    camera_configs = [
        {"id": 1, "source": args.cam1, "label": args.name1, "enabled": not args.off1 and bool(args.cam1)},
        {"id": 2, "source": args.cam2, "label": args.name2, "enabled": not args.off2 and bool(args.cam2)},
        {"id": 3, "source": args.cam3, "label": args.name3, "enabled": not args.off3 and bool(args.cam3)},
        {"id": 4, "source": args.cam4, "label": args.name4, "enabled": not args.off4 and bool(args.cam4)},
    ]

    run_multicam_grid(
        camera_configs=camera_configs,
        threshold=args.threshold,
        process_interval=args.skip,
        log_events=args.log_events,
        cooldown_sec=args.cooldown,
        device=args.device,
        record_nvr=args.record_nvr,
        segment_duration_sec=args.segment_min * 60,
        max_storage_gb=args.max_gb
    )

