import os
import sys
import time
import threading
import cv2
import numpy as np

# Ensure proper paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import db
import face_engine
import config
import hardware
from image_utils import imwrite_unicode

DATA_DIR = os.path.join(APP_DIR, "data")
SNAPSHOTS_DIR = config.get_snapshot_dir()
EVENTS_DIR = os.path.join(DATA_DIR, "events")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)

# ----------------- THREADED CAMERA WORKER -----------------
class CameraWorker:
    """
    Dedicated background thread reader for a single camera (USB or RTSP/Network).
    Provides zero-latency latest-frame retrieval and auto-reconnection.
    """
    def __init__(self, cam_id: int, source: str, label: str):
        self.cam_id = cam_id
        self.source = str(source).strip()
        self.label = label
        
        self.is_running = False
        self.is_connected = False
        self.cap = None
        self.thread = None
        
        self.last_frame = None
        self.lock = threading.Lock()
        self.fps = 0.0
        self.frame_count = 0
        self.last_error = ""

    def start(self):
        self.is_running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.is_running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _open_capture(self):
        src = self.source
        is_num = src.isdigit()
        if is_num:
            idx = int(src)
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
        else:
            if src.startswith("rtsp://") or src.startswith("http://"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|reorder_queue_size;0|buffer_size;1024000"
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(src)
            else:
                cap = cv2.VideoCapture(src)
        return cap

    def _read_loop(self):
        while self.is_running:
            if self.cap is None or not self.cap.isOpened():
                self.is_connected = False
                try:
                    self.cap = self._open_capture()
                except Exception as e:
                    self.last_error = str(e)
                    time.sleep(2.0)
                    continue

                if not self.cap or not self.cap.isOpened():
                    self.is_connected = False
                    time.sleep(2.5)
                    continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.is_connected = False
                time.sleep(0.5)
                continue

            self.is_connected = True
            with self.lock:
                self.last_frame = frame
                self.frame_count += 1

            time.sleep(0.005) # Yield briefly to avoid 100% core saturation

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def get_latest_frame(self, target_w=640, target_h=360):
        """Returns the latest frame resized to target dimensions or a cyber placeholder."""
        with self.lock:
            frame = self.last_frame.copy() if self.last_frame is not None else None
            connected = self.is_connected

        if frame is not None and connected:
            h, w = frame.shape[:2]
            if (w, h) != (target_w, target_h):
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return True, frame
        else:
            # Generate sleek cyber placeholder
            placeholder = np.zeros((target_h, target_w, 3), dtype=np.uint8)
            placeholder[:] = (15, 23, 42) # Slate dark
            
            # Subtle grid lines
            for y in range(0, target_h, 40):
                cv2.line(placeholder, (0, y), (target_w, y), (26, 37, 60), 1)
            for x in range(0, target_w, 40):
                cv2.line(placeholder, (x, 0), (x, target_h), (26, 37, 60), 1)
                
            # Status icon / text
            txt_status = f"[CAM {self.cam_id}] Povezivanje / Nema signala..."
            cv2.putText(placeholder, txt_status, (target_w // 2 - 160, target_h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, (239, 68, 68), 1, cv2.LINE_AA)
            cv2.putText(placeholder, self.label[:35], (target_w // 2 - 130, target_h // 2 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
            return False, placeholder


# ----------------- MULTI-CAM ORCHESTRATOR -----------------
def draw_cam_cell_hud(cell_img, cam_id: int, label: str, is_connected: bool, num_faces: int, is_solo=False):
    """Renders cyber HUD badges over each individual camera cell."""
    h, w = cell_img.shape[:2]
    
    # Outer cell border: Glowing cyan if connected, red if disconnected
    border_color = (212, 182, 6) if is_connected else (68, 68, 239) # BGR
    cv2.rectangle(cell_img, (0, 0), (w - 1, h - 1), border_color, 2)
    
    # Top mini-badge bar
    badge_w = min(w - 20, 320)
    overlay = cell_img[2:32, 2:badge_w].copy()
    bg = np.zeros_like(overlay)
    cv2.rectangle(bg, (0, 0), (overlay.shape[1], overlay.shape[0]), (13, 20, 36), -1)
    cv2.addWeighted(bg, 0.85, overlay, 0.15, 0, overlay)
    cell_img[2:32, 2:badge_w] = overlay
    
    # Dot indicator
    dot_color = (129, 185, 16) if is_connected else (68, 68, 239)
    cv2.circle(cell_img, (14, 16), 5, dot_color, -1)
    
    tag = f"CAM {cam_id}: {label[:20]}"
    if is_solo:
        tag += " [SOLO]"
    cv2.putText(cell_img, tag, (26, 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (248, 250, 252), 1, cv2.LINE_AA)
    
    # Right side of cell: faces count badge
    if is_connected and num_faces > 0:
        face_txt = f"{num_faces} lica"
        (tw, th), _ = cv2.getTextSize(face_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        fx1 = w - tw - 24
        f_overlay = cell_img[2:32, fx1:w-2].copy()
        f_bg = np.zeros_like(f_overlay)
        cv2.rectangle(f_bg, (0, 0), (f_overlay.shape[1], f_overlay.shape[0]), (16, 185, 129), -1)
        cv2.addWeighted(f_bg, 0.35, f_overlay, 0.65, 0, f_overlay)
        cell_img[2:32, fx1:w-2] = f_overlay
        cv2.putText(cell_img, face_txt, (fx1 + 6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (16, 185, 129), 1, cv2.LINE_AA)


from nvr_recorder import get_nvr_manager

def draw_grid_master_hud(master_frame, fps, active_count, total_count, solo_cam=0, log_events=False, record_nvr=False, status_msg=""):
    """Draws top and bottom master bars for the entire 2x2 multi-cam window."""
    h, w = master_frame.shape[:2]
    
    # Top bar
    top_h = 44
    top_bar = master_frame[0:top_h, 0:w].copy()
    overlay = np.zeros_like(top_bar)
    cv2.rectangle(overlay, (0, 0), (w, top_h), (11, 17, 32), -1)
    cv2.addWeighted(overlay, 0.85, top_bar, 0.15, 0, top_bar)
    master_frame[0:top_h, 0:w] = top_bar
    cv2.line(master_frame, (0, top_h), (w, top_h), (212, 182, 6), 2) # Cyan neon line
    
    # Title & Telemetry
    cv2.putText(master_frame, "Argusface Multi-Cam 2x2 Grid", (14, 28), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
    
    ev_txt = "● EVIDENCIJA" if log_events else "EVIDENCIJA: OFF"
    ev_color = (129, 185, 16) if log_events else (148, 163, 184)
    cv2.putText(master_frame, ev_txt, (340, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.46, ev_color, 1, cv2.LINE_AA)
    
    # NVR Recording indicator
    if record_nvr:
        cv2.circle(master_frame, (490, 24), 6, (0, 0, 235), -1)
        cv2.putText(master_frame, "REC [NVR]", (502, 28), cv2.FONT_HERSHEY_DUPLEX, 0.50, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(master_frame, "NVR: OFF (Tipka 'R')", (490, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 116, 139), 1, cv2.LINE_AA)

    # Right stats
    stats_txt = f"{fps:.1f} FPS  |  Aktivno: {active_count}/{total_count} kamera"
    (stw, _), _ = cv2.getTextSize(stats_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
    cv2.putText(master_frame, stats_txt, (w - stw - 18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (6, 182, 212), 1, cv2.LINE_AA)
    
    # Bottom bar
    bot_h = 32
    bot_y = h - bot_h
    bot_bar = master_frame[bot_y:h, 0:w].copy()
    b_overlay = np.zeros_like(bot_bar)
    cv2.rectangle(b_overlay, (0, 0), (w, bot_h), (11, 17, 32), -1)
    cv2.addWeighted(b_overlay, 0.85, bot_bar, 0.15, 0, bot_bar)
    master_frame[bot_y:h, 0:w] = bot_bar
    
    if status_msg:
        cv2.putText(master_frame, status_msg, (14, h - 10), cv2.FONT_HERSHEY_DUPLEX, 0.52, (34, 197, 94), 1, cv2.LINE_AA)
    else:
        controls = "[1-4] Solo  |  [0/ESC] Mreza  |  [R] NVR Snimanje  |  [E] Evidencija  |  [S] Snimi kadar  |  [Q] Izlaz"
        cv2.putText(master_frame, controls, (14, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)


def run_multicam_grid(
    camera_configs: list,
    threshold: float = 0.45,
    process_interval: int = 3,
    log_events: bool = False,
    cooldown_sec: int = 30,
    device: str = "AUTO",
    record_nvr: bool = False,
    segment_duration_sec: int = 300,
    max_storage_gb: float = 20.0
):
    """
    Main loop for Multi-Camera 2x2 Grid with real-time biometric face recognition and NVR recording.
    """
    accel_type, accel_status = hardware.get_onnx_acceleration_status()
    print("=========================================================")
    print("      Argusface Multi-Camera 2x2 Grid - Nadzor Uživo")
    print(f"      [{accel_status} ({accel_type})]")
    print(f"      [Prag: {threshold} | Interval: {process_interval} | Dnevnik: {log_events} | NVR: {record_nvr}]")
    print("=========================================================")
    
    # 1. Initialize Face Index
    embeddings = db.get_all_embeddings()
    face_idx = face_engine.FaceIndex(embeddings)
    print(f"[INFO] Baza lica ucitana: {len(face_idx.person_ids)} osoba, {face_idx.total_samples} vektora.")

    # 2. Filter active cameras and start background workers
    active_configs = [c for c in camera_configs if c.get("enabled", True) and str(c.get("source", "")).strip()]
    if not active_configs:
        print("❌ [GREŠKA] Nijedna kamera nije omogućena ili unesena!")
        return False

    workers = []
    for cfg in active_configs[:4]:
        cid = cfg.get("id", len(workers) + 1)
        src = str(cfg.get("source")).strip()
        lbl = str(cfg.get("label", f"Kamera {cid}")).strip()
        worker = CameraWorker(cid, src, lbl)
        worker.start()
        workers.append(worker)
        print(f" -> Pokrenuta radna dretva za Cam #{cid}: {lbl} ({src})")

    # 3. Window configuration
    window_name = "Argusface Multi-Camera 2x2 Grid"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 760)

    # Resolution per cell in 2x2 mode
    cell_w, cell_h = 640, 360
    grid_w, grid_h = cell_w * 2, cell_h * 2
    top_hud_h = 44
    bot_hud_h = 32

    # State
    solo_cam_id = 0 # 0 means 2x2 grid, 1..4 means solo full view
    last_seen_times = {} # { (cam_id, person_name): timestamp }
    status_notification = ""
    status_notification_time = 0.0
    frame_counter = 0
    t_prev = time.perf_counter()
    fps = 0.0

    # NVR Manager
    nvr = get_nvr_manager()
    nvr.max_storage_gb = max_storage_gb
    if record_nvr:
        for w in workers:
            nvr.configure_channel(w.cam_id, w.label, segment_duration_sec=segment_duration_sec, fps=20.0)
        print(f"[NVR] Pokrenuto 24/7 NVR snimanje za {len(workers)} kanala (segment: {segment_duration_sec}s).")

    # Store tracked detections per camera: { cam_id: [ {"bbox": ..., "label": ..., "color": ..., "crop": ...} ] }
    cam_tracked_faces = {w.cam_id: [] for w in workers}

    def record_event_if_eligible(cam_id, cam_label, person_name, similarity, crop_bgr, full_frame):
        if not log_events:
            return
        now_ts = time.time()
        key = (cam_id, person_name)
        last_ts = last_seen_times.get(key, 0.0)
        if (now_ts - last_ts) >= cooldown_sec:
            last_seen_times[key] = now_ts
            t_stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_pname = "".join(c for c in person_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
            
            crop_path = ""
            if crop_bgr is not None and getattr(crop_bgr, "size", 0) > 0:
                crop_filename = f"ev_crop_cam{cam_id}_{safe_pname}_{t_stamp}.jpg"
                crop_path = os.path.join(EVENTS_DIR, crop_filename)
                try:
                    imwrite_unicode(crop_path, crop_bgr)
                except Exception:
                    crop_path = ""

            snap_path = ""
            if full_frame is not None and getattr(full_frame, "size", 0) > 0:
                snap_filename = f"ev_cam{cam_id}_{safe_pname}_{t_stamp}.jpg"
                snap_path = os.path.join(EVENTS_DIR, snap_filename)
                try:
                    imwrite_unicode(snap_path, full_frame)
                except Exception:
                    snap_path = ""
            elif crop_path:
                snap_path = crop_path

            vid_path, vid_offset = nvr.get_bookmark(cam_id) if record_nvr else ("", 0.0)

            try:
                db.log_detection_event(
                    person_name=person_name,
                    similarity=similarity,
                    source_label=cam_label,
                    crop_path=crop_path,
                    snapshot_path=snap_path,
                    video_path=vid_path,
                    video_offset_sec=vid_offset
                )
                bmark_str = f" [Video bookmark: {os.path.basename(vid_path)} @ {vid_offset:.1f}s]" if vid_path else ""
                print(f"[EVIDENCIJA] Cam #{cam_id} ({cam_label}): Zabilježen {person_name} ({similarity*100:.1f}%){bmark_str}")
            except Exception as err:
                print(f"[UPOZORENJE] Greška pri spremanju u dnevnik: {err}")

    try:
        while True:
            frame_counter += 1
            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Clear expired banner message
            if status_notification and (time.time() - status_notification_time) > 3.0:
                status_notification = ""

            active_worker_count = sum(1 for w in workers if w.is_connected)
            processed_cells = []

            # Process each camera
            for idx, w in enumerate(workers):
                # Desired resolution depends on solo or grid
                req_w = 1280 if (solo_cam_id == w.cam_id) else cell_w
                req_h = 720 if (solo_cam_id == w.cam_id) else cell_h

                conn, cell_img = w.get_latest_frame(target_w=req_w, target_h=req_h)

                # Periodic face detection (staggered across cameras)
                if conn and (frame_counter % process_interval == (idx % process_interval)):
                    try:
                        detected = face_engine.extract_faces_from_image(cell_img, device=device, with_attributes=False)
                        new_tracked = []
                        for f in detected:
                            bbox = f["bbox"]
                            emb = f.get("embedding")
                            if emb is None:
                                continue
                            match = face_idx.match(emb, threshold=threshold)
                            sim = float(match.get("similarity", 0.0))
                            sim_pct = sim * 100.0
                            p_name = match.get("person_name") or match.get("best_name", "Nepoznato")

                            if match.get("matched", False):
                                color = (16, 185, 129) # Green
                                label = f"{p_name} ({sim_pct:.0f}%)"
                                record_event_if_eligible(w.cam_id, w.label, p_name, sim, f.get("crop_bgr"), cell_img)
                            elif sim >= max(0.30, threshold - 0.10):
                                color = (11, 158, 245) # Orange
                                label = f"{p_name}? ({sim_pct:.0f}%)"
                            else:
                                color = (68, 68, 239) # Red
                                label = "Nepoznato"

                            new_tracked.append({
                                "bbox": bbox,
                                "label": label,
                                "color": color,
                                "crop": f.get("crop_bgr")
                            })
                        cam_tracked_faces[w.cam_id] = new_tracked
                    except Exception as err:
                        pass

                # Draw tracked face boxes on this cell
                current_faces = cam_tracked_faces.get(w.cam_id, [])
                for face_item in current_faces:
                    x1, y1, x2, y2 = face_item["bbox"]
                    col = face_item["color"]
                    lbl = face_item["label"]

                    # Draw stylized bounding box
                    cv2.rectangle(cell_img, (x1, y1), (x2, y2), col, 2)
                    (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tag_y = max(y1, lh + 6)
                    cv2.rectangle(cell_img, (x1, tag_y - lh - 6), (x1 + lw + 8, tag_y), col, -1)
                    cv2.putText(cell_img, lbl, (x1 + 4, tag_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                # Draw cell-level HUD (label, dot, faces count)
                draw_cam_cell_hud(cell_img, w.cam_id, w.label, conn, len(current_faces), is_solo=(solo_cam_id == w.cam_id))
                
                # Write to NVR recording if enabled
                if record_nvr and conn and cell_img is not None:
                    nvr.write_frame(w.cam_id, cell_img)

                processed_cells.append((w.cam_id, cell_img))

            # ---------------- COMPOSITING THE CANVAS ----------------
            if solo_cam_id > 0:
                # Find the solo cell
                solo_item = next((c for cid, c in processed_cells if cid == solo_cam_id), None)
                if solo_item is None and processed_cells:
                    solo_item = processed_cells[0][1]
                
                # Compose master window with top and bottom HUD
                content_h = solo_item.shape[0]
                content_w = solo_item.shape[1]
                canvas = np.zeros((content_h + top_hud_h + bot_hud_h, content_w, 3), dtype=np.uint8)
                canvas[top_hud_h:top_hud_h + content_h, 0:content_w] = solo_item
            else:
                # 2x2 Grid Mode
                grid_cells = [c for _, c in processed_cells]
                # Pad to 4 cells if fewer cameras configured
                while len(grid_cells) < 4:
                    empty_cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
                    empty_cell[:] = (15, 23, 42)
                    cid_empty = len(grid_cells) + 1
                    cv2.putText(empty_cell, f"[CAM {cid_empty}] Nije konfigurirano", (cell_w // 2 - 120, cell_h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (71, 85, 105), 1, cv2.LINE_AA)
                    cv2.rectangle(empty_cell, (0, 0), (cell_w - 1, cell_h - 1), (30, 41, 59), 1)
                    grid_cells.append(empty_cell)

                row1 = np.hstack((grid_cells[0], grid_cells[1]))
                row2 = np.hstack((grid_cells[2], grid_cells[3]))
                grid_matrix = np.vstack((row1, row2))

                # Canvas with top & bottom HUD
                canvas = np.zeros((grid_h + top_hud_h + bot_hud_h, grid_w, 3), dtype=np.uint8)
                canvas[top_hud_h:top_hud_h + grid_h, 0:grid_w] = grid_matrix

            # Draw master top and bottom HUD
            draw_grid_master_hud(
                canvas,
                fps=fps,
                active_count=active_worker_count,
                total_count=len(workers),
                solo_cam=solo_cam_id,
                log_events=log_events,
                record_nvr=record_nvr,
                status_msg=status_notification
            )

            cv2.imshow(window_name, canvas)

            # ---------------- KEYBOARD SHORTCUTS ----------------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27): # Q or ESC
                if solo_cam_id > 0:
                    solo_cam_id = 0 # Return to grid on ESC
                else:
                    break
            elif key == ord('1'):
                solo_cam_id = 1 if (solo_cam_id != 1 and len(workers) >= 1) else 0
            elif key == ord('2'):
                solo_cam_id = 2 if (solo_cam_id != 2 and len(workers) >= 2) else 0
            elif key == ord('3'):
                solo_cam_id = 3 if (solo_cam_id != 3 and len(workers) >= 3) else 0
            elif key == ord('4'):
                solo_cam_id = 4 if (solo_cam_id != 4 and len(workers) >= 4) else 0
            elif key == ord('0'):
                solo_cam_id = 0 # Return to 2x2 grid
            elif key in (ord('r'), ord('R')):
                record_nvr = not record_nvr
                if record_nvr:
                    for w in workers:
                        nvr.configure_channel(w.cam_id, w.label, segment_duration_sec=segment_duration_sec, fps=20.0)
                    status_notification = "🔴 NVR Snimanje: AKTIVNO"
                else:
                    nvr.stop_all()
                    status_notification = "NVR Snimanje: ISKLJUCENO"
                status_notification_time = time.time()
            elif key in (ord('e'), ord('E')):
                log_events = not log_events
                status_notification = "📋 Evidencija prolazaka: AKTIVNA" if log_events else "Evidencija: ISKLJUCENA"
                status_notification_time = time.time()
            elif key in (ord('s'), ord('S')):
                # Save snapshot of full grid
                snap_dir = config.get_snapshot_dir()
                t_str = time.strftime("%Y%m%d_%H%M%S")
                snap_filename = f"multicam_grid_{t_str}.jpg"
                snap_path = os.path.join(snap_dir, snap_filename)
                imwrite_unicode(snap_path, canvas)
                status_notification = f"📸 Snimljen kadar mreze: {snap_filename}"
                status_notification_time = time.time()
                print(f"[SNAPSHOT] Snimljen kadar: {snap_path}")

    finally:
        print("[INFO] Zaustavljam NVR snimanje i radne dretve...")
        try:
            nvr.stop_all()
        except Exception:
            pass
        for w in workers:
            w.stop()
        cv2.destroyAllWindows()
        print("[INFO] Multi-Camera prozor zatvoren.")
