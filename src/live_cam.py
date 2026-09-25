import os
import sys
import time
import argparse
import cv2
import numpy as np

# Prevent Windows console cp1250 UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure src and root are in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import db
import face_engine
from image_utils import imwrite_unicode

DATA_DIR = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)

def draw_corner_box(img, pt1, pt2, color, thickness=2, corner_len=20):
    """Draws a modern bounding box with stylized corner accents."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Subtle full box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
    
    # Top-Left
    cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, thickness)
    
    # Top-Right
    cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, thickness)
    
    # Bottom-Left
    cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, thickness)
    
    # Bottom-Right
    cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, thickness)

def draw_hud(frame, fps, num_faces, threshold, total_persons, paused=False, status_msg=""):
    """Renders sleek top and bottom HUD panels with live telemetry."""
    h, w = frame.shape[:2]
    
    # Top HUD background
    top_bar = frame[0:55, 0:w].copy()
    overlay = np.zeros_like(top_bar)
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 23, 42), -1) # Dark slate
    cv2.addWeighted(overlay, 0.75, top_bar, 0.25, 0, top_bar)
    frame[0:55, 0:w] = top_bar
    cv2.line(frame, (0, 55), (w, 55), (59, 130, 246), 2) # Blue accent line
    
    # Top HUD text
    cv2.putText(frame, "UniFace Live Camera", (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Logitech C270 HD (720p)", (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
    
    # Stats on top right
    stats_x = max(260, w - 460)
    status_indicator = "|| PAUZIRANO" if paused else f"{fps:.1f} FPS"
    fps_color = (0, 165, 255) if paused else (16, 185, 129) # Orange or green
    cv2.putText(frame, f"Brzina: {status_indicator}", (stats_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Lica u kadru: {num_faces}", (stats_x, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Prag: {threshold:.2f}", (stats_x + 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 158, 11), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Baza: {total_persons} osoba", (stats_x + 180, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

    # Bottom status / help bar
    bot_y = h - 35
    bot_bar = frame[bot_y:h, 0:w].copy()
    b_overlay = np.zeros_like(bot_bar)
    cv2.rectangle(b_overlay, (0, 0), (w, 35), (15, 23, 42), -1)
    cv2.addWeighted(b_overlay, 0.75, bot_bar, 0.25, 0, bot_bar)
    frame[bot_y:h, 0:w] = bot_bar
    
    if status_msg:
        cv2.putText(frame, status_msg, (15, h - 12), cv2.FONT_HERSHEY_DUPLEX, 0.55, (34, 197, 94), 1, cv2.LINE_AA)
    else:
        controls_txt = "[Q/ESC] Izlaz   [S] Spremi kadar   [+/-] Prag   [R] Osvjezi bazu   [SPACE] Pauza"
        cv2.putText(frame, controls_txt, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

def run_live_camera(camera_idx=0, threshold=0.45, process_interval=2, device="CPU"):
    """
    Main loop for live face recognition from webcam.
    """
    print("===================================================")
    print("      UniFace Live Camera - Logitech C270")
    print("===================================================")
    print(f"Otvaram kameru indeks {camera_idx} (DirectShow backend)...")
    
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[GRESKA] Ne mogu otvoriti kameru na indeksu {camera_idx}. Provjerite je li kamera spojena.")
        return False
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    window_name = "UniFace Live Recognition - Logitech C270"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass
    
    print("[INFO] Ucitavam biometrijski FaceIndex u memoriju...")
    face_idx = face_engine.get_face_index()
    stats = db.get_stats()
    total_persons = stats["total_persons"]
    print(f"[OK] Ucitano {total_persons} osoba i {stats['total_samples']} uzoraka.")
    
    print("[INFO] Zagrijavam AI modele za prepoznavanje...")
    face_engine.get_analyzer(device=device, with_attributes=False)
    print("[OK] AI modeli spremni za rad!")
    print("Pokrecem video prikaz. Za izlaz pritisnite tipku 'Q' ili 'ESC' u prozoru kamere...")

    frame_count = 0
    fps = 0.0
    t_start = time.perf_counter()
    fps_history = []
    
    current_faces_tracked = []
    paused = False
    paused_frame = None
    status_notification = ""
    status_notification_time = 0
    
    possible_threshold_delta = 0.08
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("⚠️ Gubitak signala s kamere.")
                    time.sleep(0.1)
                    continue
                frame_count += 1
            else:
                frame = paused_frame.copy()
                
            now = time.perf_counter()
            elapsed = now - t_start
            t_start = now
            if elapsed > 0:
                fps_history.append(1.0 / elapsed)
                if len(fps_history) > 15:
                    fps_history.pop(0)
                fps = sum(fps_history) / len(fps_history)

            # Face recognition on process_interval frames (smooth performance)
            if not paused and (frame_count % process_interval == 0):
                # Analyze frame with RetinaFace + ArcFace
                detected = face_engine.extract_faces_from_image(frame, device=device, with_attributes=False)
                new_tracked = []
                
                for f in detected:
                    bbox = f["bbox"]
                    emb = f["embedding"]
                    
                    # Sub-millisecond BLAS matching against database
                    match = face_idx.match_sample_hybrid(emb, threshold=threshold)
                    
                    sim = match["similarity"]
                    sim_pct = sim * 100
                    p_name = match["person_name"]
                    
                    if match["matched"]:
                        # Verified match -> Emerald Green
                        color = (16, 185, 129)
                        label = f"{p_name} ({sim_pct:.1f}%)"
                        status_type = "match"
                    elif sim >= max(0.30, threshold - possible_threshold_delta):
                        # Possible match -> Amber Orange
                        color = (11, 158, 245)
                        label = f"Moguce: {p_name} ({sim_pct:.1f}%)"
                        status_type = "possible"
                    else:
                        # Unknown -> Crimson Red
                        color = (68, 68, 239)
                        label = f"Nepoznato ({sim_pct:.1f}%)" if sim > 0.15 else "Nepoznata osoba"
                        status_type = "unknown"
                        
                    new_tracked.append({
                        "bbox": bbox,
                        "label": label,
                        "color": color,
                        "status": status_type,
                        "crop": f.get("crop_bgr", None),
                        "emb": emb
                    })
                current_faces_tracked = new_tracked

            # Render tracked faces onto frame
            display_frame = frame.copy()
            for face in current_faces_tracked:
                x1, y1, x2, y2 = map(int, face["bbox"])
                color = face["color"]
                label = face["label"]
                
                # Draw corner box
                draw_corner_box(display_frame, (x1, y1), (x2, y2), color, thickness=3, corner_len=24)
                
                # Label badge
                (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                badge_y1 = max(60, y1 - th - 12)
                badge_y2 = max(60 + th + 8, y1)
                badge_x2 = min(display_frame.shape[1], x1 + tw + 16)
                
                # Filled badge background
                cv2.rectangle(display_frame, (x1, badge_y1), (badge_x2, badge_y2), (15, 23, 42), -1)
                cv2.rectangle(display_frame, (x1, badge_y1), (badge_x2, badge_y2), color, 1)
                cv2.putText(display_frame, label, (x1 + 8, badge_y2 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Notification timeout (2 seconds)
            active_msg = ""
            if status_notification and (time.time() - status_notification_time < 2.5):
                active_msg = status_notification
            else:
                status_notification = ""

            # Draw HUD
            draw_hud(
                display_frame,
                fps=fps,
                num_faces=len(current_faces_tracked),
                threshold=threshold,
                total_persons=total_persons,
                paused=paused,
                status_msg=active_msg
            )
            
            cv2.imshow(window_name, display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')): # ESC or Q -> Exit
                break
            elif key in (ord(' '), ): # Space -> Pause
                paused = not paused
                if paused:
                    paused_frame = frame.copy()
                    status_notification = "⏸️ Slika zamrznuta (Pauza). Pritisnite SPACE za nastavak."
                else:
                    status_notification = "▶️ Nastavak snimanja uzivo."
                status_notification_time = time.time()
            elif key in (ord('s'), ord('S')): # S -> Snapshot
                t_str = time.strftime("%Y%m%d_%H%M%S")
                snap_path = os.path.join(UPLOADS_DIR, f"live_snap_{t_str}.jpg")
                imwrite_unicode(snap_path, frame)
                status_notification = f"Kadar spremljen u: data/uploads/live_snap_{t_str}.jpg"
                status_notification_time = time.time()
                print(f"[OK] {status_notification}")
            elif key in (ord('+'), ord('=')): # Increase threshold
                threshold = min(0.95, round(threshold + 0.02, 2))
                status_notification = f"Prag povecan na: {threshold:.2f} (stroze)"
                status_notification_time = time.time()
            elif key in (ord('-'), ord('_')): # Decrease threshold
                threshold = max(0.20, round(threshold - 0.02, 2))
                status_notification = f"Prag smanjen na: {threshold:.2f} (blaze)"
                status_notification_time = time.time()
            elif key in (ord('r'), ord('R')): # Refresh DB cache
                db.invalidate_cache()
                face_idx = face_engine.get_face_index()
                stats = db.get_stats()
                total_persons = stats["total_persons"]
                status_notification = f"Baza osvjezena! Osoba: {total_persons}"
                status_notification_time = time.time()
                print(f"[OK] {status_notification}")
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera uspjesno oslobodena i ugasena.")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniFace Live Face Recognition")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.45, help="Recognition cosine similarity threshold (default: 0.45)")
    parser.add_argument("--skip", type=int, default=2, help="Process every N frames (default: 2)")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device: CPU or CUDA (default: CPU)")
    args = parser.parse_args()
    
    run_live_camera(
        camera_idx=args.camera,
        threshold=args.threshold,
        process_interval=args.skip,
        device=args.device
    )
