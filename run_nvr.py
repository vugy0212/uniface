import os
import sys
import time
import signal
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for p in [CURRENT_DIR, os.path.join(CURRENT_DIR, "src"), PARENT_DIR, os.path.join(PARENT_DIR, "src")]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

import db
import face_engine
import config
from image_utils import imwrite_unicode
from src.multicam import CameraWorker
from src.nvr_recorder import get_nvr_manager

DATA_DIR = os.path.join(CURRENT_DIR, "data")
EVENTS_DIR = os.path.join(DATA_DIR, "events")
os.makedirs(EVENTS_DIR, exist_ok=True)

is_running = True

def signal_handler(sig, frame):
    global is_running
    print("\n[INFO] Primljen signal za zaustavljanje NVR poslužitelja...")
    is_running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def run_headless_nvr(
    camera_configs: list,
    threshold: float = 0.45,
    process_interval: int = 3,
    cooldown_sec: int = 30,
    segment_duration_sec: int = 300,
    max_storage_gb: float = 20.0,
    device: str = "CPU"
):
    global is_running
    print("=================================================================")
    print("      UniFace 24/7 NVR Sustav Snimanja i Biometrijske Evidencije")
    print(f"      [Segment: {segment_duration_sec}s | Kvota diska: {max_storage_gb} GB | Prag: {threshold}]")
    print("=================================================================")

    embeddings = db.get_all_embeddings()
    face_idx = face_engine.FaceIndex(embeddings)
    print(f"[INFO] Baza lica: {len(face_idx.person_ids)} osoba, {face_idx.total_samples} vektora.")

    active_configs = [c for c in camera_configs if c.get("enabled", True) and str(c.get("source", "")).strip()]
    if not active_configs:
        print("❌ [GREŠKA] Nijedna kamera nije omogućena ili unesena!")
        return False

    # Initialize NVR Manager
    nvr = get_nvr_manager()
    nvr.max_storage_gb = max_storage_gb

    # Start Camera Workers
    workers = []
    for cfg in active_configs[:4]:
        cid = cfg.get("id", len(workers) + 1)
        src = str(cfg.get("source")).strip()
        lbl = str(cfg.get("label", f"Kamera {cid}")).strip()
        w = CameraWorker(cid, src, lbl)
        w.start()
        workers.append(w)
        nvr.configure_channel(cid, lbl, segment_duration_sec=segment_duration_sec, fps=20.0)
        print(f" -> Pokrenut NVR kanal za Cam #{cid}: {lbl} ({src})")

    last_seen_times = {} # (cam_id, person_name) -> timestamp
    total_events_logged = 0
    frame_counter = 0

    print("\n🟢 [NVR POKRENUT] 24/7 snimanje u tijeku. Pritisnite Ctrl+C za sigurno gašenje.\n")

    t_last_stat = time.time()

    try:
        while is_running:
            frame_counter += 1
            time.sleep(0.04) # ~25 FPS pacing

            for idx, w in enumerate(workers):
                conn, frame = w.get_latest_frame(target_w=640, target_h=360)
                if not conn or frame is None:
                    continue

                # Write frame to NVR video segment
                nvr.write_frame(w.cam_id, frame)

                # Periodic face recognition
                if frame_counter % process_interval == (idx % process_interval):
                    try:
                        detected = face_engine.extract_faces_from_image(frame, device=device, with_attributes=False)
                        for f in detected:
                            emb = f.get("embedding")
                            if emb is None:
                                continue
                            match = face_idx.match(emb, threshold=threshold)
                            if match.get("matched", False):
                                p_name = match.get("person_name") or match.get("best_name", "Osoba")
                                sim = float(match.get("similarity", 0.0))
                                
                                now_ts = time.time()
                                key = (w.cam_id, p_name)
                                if (now_ts - last_seen_times.get(key, 0.0)) >= cooldown_sec:
                                    last_seen_times[key] = now_ts
                                    t_stamp = time.strftime("%Y%m%d_%H%M%S")
                                    safe_pname = "".join(c for c in p_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
                                    
                                    # Save crop
                                    crop_bgr = f.get("crop_bgr")
                                    crop_path = ""
                                    if crop_bgr is not None and getattr(crop_bgr, "size", 0) > 0:
                                        crop_filename = f"ev_crop_cam{w.cam_id}_{safe_pname}_{t_stamp}.jpg"
                                        crop_path = os.path.join(EVENTS_DIR, crop_filename)
                                        try:
                                            imwrite_unicode(crop_path, crop_bgr)
                                        except Exception:
                                            crop_path = ""

                                    # Save snapshot
                                    snap_filename = f"ev_cam{w.cam_id}_{safe_pname}_{t_stamp}.jpg"
                                    snap_path = os.path.join(EVENTS_DIR, snap_filename)
                                    try:
                                        imwrite_unicode(snap_path, frame)
                                    except Exception:
                                        snap_path = ""

                                    # Get precise video segment path and time offset
                                    vid_path, vid_offset = nvr.get_bookmark(w.cam_id)

                                    db.log_detection_event(
                                        person_name=p_name,
                                        similarity=sim,
                                        source_label=w.label,
                                        crop_path=crop_path,
                                        snapshot_path=snap_path,
                                        video_path=vid_path,
                                        video_offset_sec=vid_offset
                                    )
                                    total_events_logged += 1
                                    bmark_str = f" [Video: {os.path.basename(vid_path)} @ {vid_offset:.1f}s]" if vid_path else ""
                                    print(f"[{time.strftime('%H:%M:%S')}] 📋 [EVIDENCIJA] Cam #{w.cam_id} ({w.label}): {p_name} ({sim*100:.1f}%){bmark_str}")
                    except Exception as err:
                        pass

            # Log periodic status every 30 seconds
            if time.time() - t_last_stat > 30.0:
                t_last_stat = time.time()
                active_cams = sum(1 for w in workers if w.is_connected)
                print(f"[{time.strftime('%H:%M:%S')}] ℹ️  [NVR STATUS] Aktivnih kamera: {active_cams}/{len(workers)} | Zabilježeno prolazaka ukupno: {total_events_logged}")

    finally:
        print("[INFO] Zaustavljam NVR snimače i zatvaram video segmente...")
        nvr.stop_all()
        for w in workers:
            w.stop()
        print("[OK] NVR servis uspješno zaustavljen.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniFace 24/7 NVR Background Recorder & Face Indexer")
    parser.add_argument("--cam1", type=str, default="0", help="Camera 1 source")
    parser.add_argument("--name1", type=str, default="USB Web Kamera", help="Camera 1 label")
    parser.add_argument("--off1", action="store_true", default=False, help="Disable Camera 1")

    parser.add_argument("--cam2", type=str, default="rtsp://admin:admin@192.168.50.236:554/11", help="Camera 2 source")
    parser.add_argument("--name2", type=str, default="Denver IP Kamera", help="Camera 2 label")
    parser.add_argument("--off2", action="store_true", default=False, help="Disable Camera 2")

    parser.add_argument("--cam3", type=str, default="", help="Camera 3 source (optional)")
    parser.add_argument("--name3", type=str, default="Kamera 3", help="Camera 3 label")
    parser.add_argument("--off3", action="store_true", default=False, help="Disable Camera 3")

    parser.add_argument("--cam4", type=str, default="", help="Camera 4 source (optional)")
    parser.add_argument("--name4", type=str, default="Kamera 4", help="Camera 4 label")
    parser.add_argument("--off4", action="store_true", default=False, help="Disable Camera 4")

    parser.add_argument("--threshold", type=float, default=0.45, help="Recognition threshold (default: 0.45)")
    parser.add_argument("--skip", type=int, default=3, help="Process every N frames (default: 3)")
    parser.add_argument("--cooldown", type=int, default=30, help="Event cooldown in seconds (default: 30)")
    parser.add_argument("--segment-min", type=int, default=5, help="MP4 video segment length in minutes (default: 5)")
    parser.add_argument("--max-gb", type=float, default=20.0, help="Maximum storage quota in GB (default: 20)")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device: CPU or CUDA (default: CPU)")

    args = parser.parse_args()

    camera_configs = [
        {"id": 1, "source": args.cam1, "label": args.name1, "enabled": not args.off1 and bool(args.cam1)},
        {"id": 2, "source": args.cam2, "label": args.name2, "enabled": not args.off2 and bool(args.cam2)},
        {"id": 3, "source": args.cam3, "label": args.name3, "enabled": not args.off3 and bool(args.cam3)},
        {"id": 4, "source": args.cam4, "label": args.name4, "enabled": not args.off4 and bool(args.cam4)},
    ]

    run_headless_nvr(
        camera_configs=camera_configs,
        threshold=args.threshold,
        process_interval=args.skip,
        cooldown_sec=args.cooldown,
        segment_duration_sec=args.segment_min * 60,
        max_storage_gb=args.max_gb,
        device=args.device
    )
