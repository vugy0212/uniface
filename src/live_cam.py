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
import config
from image_utils import imwrite_unicode

DATA_DIR = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
SNAPSHOTS_DIR = config.get_snapshot_dir()
EVENTS_DIR = os.path.join(DATA_DIR, "events")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
os.makedirs(EVENTS_DIR, exist_ok=True)

DEBUG_LOG_PATH = os.path.join(DATA_DIR, "live_cam_debug.log")
def log_debug(msg):
    try:
        t = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{t}] {msg}\n")
    except Exception:
        pass

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

def draw_hud(frame, fps, num_faces, threshold, total_persons, paused=False, status_msg="", camera_label="USB Web Kamera", cur_sec=0, total_sec=0, log_events=False):
    """Renders sleek top and bottom HUD panels with live telemetry and timeline."""
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
    
    # Subtitle: camera label, timeline position and event log status
    sub_title = camera_label
    if total_sec > 0:
        c_min, c_s = int(cur_sec) // 60, int(cur_sec) % 60
        t_min, t_s = int(total_sec) // 60, int(total_sec) % 60
        sub_title = f"{camera_label} | [{c_min:02d}:{c_s:02d} / {t_min:02d}:{t_s:02d}]"
    if log_events:
        sub_title += " | [📋 EVIDENCIJA AKTIVNA]"
    else:
        sub_title += " | [EVIDENCIJA: ISKLJUČENA (Tipka 'E')]"
    cv2.putText(frame, sub_title, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
    
    # Stats on top right
    stats_x = max(260, w - 460)
    status_indicator = "|| PAUZIRANO" if paused else f"{fps:.1f} FPS"
    fps_color = (0, 165, 255) if paused else (16, 185, 129) # Orange or green
    cv2.putText(frame, f"Brzina: {status_indicator}", (stats_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Lica u kadru: {num_faces}", (stats_x, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)
    
    cv2.putText(frame, f"Prag: {threshold:.2f}", (stats_x + 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 158, 11), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Baza: {total_persons} osoba", (stats_x + 180, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

    # Timeline progress bar (if video has duration)
    if total_sec > 0:
        bar_y1 = h - 39
        bar_y2 = h - 35
        # Background bar
        cv2.rectangle(frame, (0, bar_y1), (w, bar_y2), (30, 41, 59), -1)
        # Progress fill
        pct = max(0.0, min(1.0, float(cur_sec) / float(total_sec)))
        fill_w = int(w * pct)
        if fill_w > 0:
            cv2.rectangle(frame, (0, bar_y1), (fill_w, bar_y2), (59, 130, 246), -1)

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
        if total_sec > 0:
            controls_txt = "[Q] Izlaz   [SPACE] Pauza   [A/D] Premotaj   [E] Evidencija ON/OFF   [S] Kadar   [O] Mapa"
        else:
            controls_txt = "[Q/ESC] Izlaz   [E] Evidencija ON/OFF   [S] Spremi kadar   [O] Mapa   [SPACE] Pauza"
        cv2.putText(frame, controls_txt, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

def resolve_stream_source(source_input):
    """
    Resolves various video inputs:
    - YouTube / web video platform URL -> extracted direct stream via yt-dlp
    - Local video file (.mp4, .mkv, .avi, .mov) -> file path
    - Network RTSP / HTTP video stream -> URL
    - Local USB webcam index -> integer (0, 1, 2)
    """
    source_str = str(source_input).strip()
    
    # 1. YouTube / Web video platform (handled by yt-dlp)
    is_web_video = any(domain in source_str.lower() for domain in ["youtube.com", "youtu.be", "vimeo.com"])
    if is_web_video:
        try:
            print(f"[INFO] Dohvaćam video stream s YouTubea: {source_str} ...")
            log_debug(f"Pokusavam yt-dlp ekstrakciju za: {source_str}")
            import yt_dlp
            
            # Format strategies: for face recognition we need video stream.
            # YouTube serves separate video and audio DASH streams for 720p/1080p, so 'bestvideo' is mandatory.
            format_candidates = [
                'bestvideo[height<=720][ext=mp4]/bestvideo[height<=720]/bestvideo/best[height<=720]/best',
                'bestvideo/best',
                'best'
            ]
            
            stream_url = None
            title = 'YouTube Video'
            
            for fmt in format_candidates:
                try:
                    ydl_opts = {
                        'format': fmt,
                        'quiet': True,
                        'no_warnings': True,
                        'noplaylist': True,
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(source_str, download=False)
                        if info:
                            stream_url = info.get('url')
                            title = info.get('title', 'YouTube Video')
                            if stream_url:
                                log_debug(f"yt-dlp uspjeh s formatom '{fmt}': naslov='{title}', stream_url={stream_url[:60]}...")
                                break
                except Exception as fmt_err:
                    log_debug(f"yt-dlp format '{fmt}' nije uspio: {fmt_err}")
                    continue
                    
            if stream_url:
                safe_title = "".join(c for c in title if c.isprintable())
                lbl = f"YouTube: {safe_title[:28]}..." if len(safe_title) > 28 else f"YouTube: {safe_title}"
                return stream_url, lbl, "youtube"
            else:
                log_debug("yt-dlp nije pronasao direktni video url.")
                print("[UPOZORENJE] yt-dlp nije uspio ekstrahirati stream link za navedeni video.")
                return None, "YouTube Greška (nedostupan stream)", "error"
        except Exception as e:
            print(f"[UPOZORENJE] Greška pri dohvaćanju YouTube streama: {e}")
            log_debug(f"yt-dlp greska: {e}")
            return None, f"YouTube Greška ({e})", "error"

    # 2. Local video file
    clean_path = source_str.strip('"\'')
    if os.path.isfile(clean_path):
        bname = os.path.basename(clean_path)
        return clean_path, f"Video: {bname[:26]}", "file"

    # 3. Network RTSP / HTTP video stream
    if (source_str.startswith("rtsp://") or 
        source_str.startswith("http://") or 
        source_str.startswith("https://")):
        return source_str, f"IP Stream ({source_str[:26]}...)", "network"

    # 4. Local USB camera index
    try:
        cam_idx = int(source_str)
    except Exception:
        cam_idx = 0
    return cam_idx, f"USB Web Kamera (indeks {cam_idx})", "usb"

def run_live_camera(camera_source=0, threshold=0.45, process_interval=2, device="CPU", start_sec=0, log_events=False, cooldown_sec=30):
    """
    Main loop for live face recognition from webcam, IP/RTSP camera, YouTube, or video file.
    """
    source_str = str(camera_source).strip()
    log_debug(f"run_live_camera pokrenut: camera_source={source_str}, threshold={threshold}, interval={process_interval}, start_sec={start_sec}, log_events={log_events}, cooldown={cooldown_sec}")
    print("===================================================")
    print("      UniFace Live Camera - Prepoznavanje Lica")
    if log_events:
        print(f"      [📋 Evidencija prolazaka: AKTIVNA | Cooldown: {cooldown_sec}s]")
    print("===================================================")

    stream_target, source_label, source_kind = resolve_stream_source(source_str)

    if stream_target is None or source_kind == "error":
        log_debug(f"Prekid rada: neispravan izvor '{source_str}' ({source_label})")
        print(f"\n❌ [GREŠKA] Nije moguće pokrenuti video prikaz: {source_label}")
        print("Pričekajte nekoliko sekundi...")
        time.sleep(3)
        return False

    if source_kind in ("youtube", "network"):
        print(f"Povezujem se na mrežni video stream ({source_kind}): {source_label} ...")
        # Optimized RTSP/HLS parameters: TCP transport avoids packet drops
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|reorder_queue_size;0|buffer_size;1024000"
        cap = cv2.VideoCapture(stream_target, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(stream_target)
    elif source_kind == "file":
        print(f"Otvaram lokalnu video datoteku: {stream_target} ...")
        cap = cv2.VideoCapture(stream_target)
    else:
        cam_idx = stream_target
        print(f"Otvaram lokalnu USB kameru indeks {cam_idx} (DirectShow backend)...")
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx)
            
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'M', 'J', 'P', 'G'))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        log_debug(f"GRESKA: Ne mogu otvoriti video izvor: {source_str}")
        print(f"[GRESKA] Ne mogu otvoriti video izvor: {source_str}. Provjerite izvor, datoteku ili URL.")
        time.sleep(3)
        return False
        
    # Get native video FPS for local files and YouTube to maintain natural playback speed
    native_fps = cap.get(cv2.CAP_PROP_FPS) if source_kind in ("file", "youtube") else 0
    target_frame_time = (1.0 / native_fps) if (native_fps and 5.0 <= native_fps <= 120.0) else 0.0

    # Warm up camera sensor for live cameras (not needed for YouTube or file)
    if source_kind not in ("file", "youtube"):
        for _ in range(5):
            cap.read()
        
    window_name = "UniFace Live Camera"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # For YouTube and video files: setup timeline duration and interactive slider
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if source_kind in ("youtube", "file") else 0
    effective_fps = native_fps if (native_fps and native_fps > 0) else 25.0
    total_seconds = int(total_frames / effective_fps) if (total_frames > 0 and effective_fps > 0) else 0
    
    seek_target_sec = None
    is_programmatic_trackbar_update = False
    
    def on_trackbar_change(pos):
        nonlocal seek_target_sec, is_programmatic_trackbar_update
        if not is_programmatic_trackbar_update:
            seek_target_sec = pos

    has_trackbar = False
    trackbar_name = "Pozicija (s)"
    if total_seconds > 5:
        try:
            cv2.createTrackbar(trackbar_name, window_name, 0, total_seconds, on_trackbar_change)
            has_trackbar = True
        except Exception:
            pass

    # Start position offset if requested
    if start_sec > 0 and source_kind in ("youtube", "file"):
        target_msec = start_sec * 1000.0
        cap.set(cv2.CAP_PROP_POS_MSEC, target_msec)
        print(f"[INFO] Video pokrenut od {start_sec // 60:02d}:{start_sec % 60:02d}...")

    print("[INFO] Ucitavam biometrijski FaceIndex u memoriju...")
    face_idx = face_engine.get_face_index()
    stats = db.get_stats()
    total_persons = stats["total_persons"]
    print(f"[OK] Ucitano {total_persons} osoba i {stats['total_samples']} uzoraka.")
    
    print("[INFO] Zagrijavam AI modele za prepoznavanje...")
    face_engine.get_analyzer(device=device, with_attributes=False)
    print("[OK] AI modeli spremni za rad!")
    print(f"Pokrecem video prikaz ({source_label}). Za izlaz pritisnite tipku 'Q' ili 'ESC' u prozoru...")

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
    
    last_seen_times = {} # person_name -> epoch timestamp for cooldown filter
    
    def record_event_if_eligible(person_name, similarity, crop_bgr, full_frame=None):
        nonlocal status_notification, status_notification_time
        if not log_events:
            return
        now_ts = time.time()
        last_ts = last_seen_times.get(person_name, 0.0)
        if (now_ts - last_ts) >= cooldown_sec:
            last_seen_times[person_name] = now_ts
            crop_saved_path = ""
            snap_saved_path = ""
            t_stamp = time.strftime("%Y%m%d_%H%M%S")
            safe_pname = "".join(c for c in person_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
            
            # 1. Save face crop
            if crop_bgr is not None and getattr(crop_bgr, "size", 0) > 0:
                crop_filename = f"ev_crop_{safe_pname}_{t_stamp}.jpg"
                crop_saved_path = os.path.join(EVENTS_DIR, crop_filename)
                try:
                    imwrite_unicode(crop_saved_path, crop_bgr)
                except Exception:
                    crop_saved_path = ""

            # 2. Save full camera picture (Kadar sa kamere)
            if full_frame is not None and getattr(full_frame, "size", 0) > 0:
                snap_filename = f"ev_cam_{safe_pname}_{t_stamp}.jpg"
                snap_saved_path = os.path.join(EVENTS_DIR, snap_filename)
                try:
                    imwrite_unicode(snap_saved_path, full_frame)
                except Exception:
                    snap_saved_path = ""
            elif crop_saved_path:
                snap_saved_path = crop_saved_path

            try:
                db.log_detection_event(
                    person_name=person_name,
                    similarity=similarity,
                    source_label=source_label,
                    crop_path=crop_saved_path,
                    snapshot_path=snap_saved_path
                )
                sim_pct = similarity * 100
                status_notification = f"📋 Evidentiran prolazak: {person_name} ({sim_pct:.1f}%)"
                status_notification_time = time.time()
                print(f"[EVIDENCIJA] Zabilježen prolazak: {person_name} ({sim_pct:.1f}%) [{source_label}] (Slika kamere spremljena)")
            except Exception as log_err:
                print(f"[UPOZORENJE] Greška evidentiranja u dnevnik: {log_err}")

    try:
        while True:
            # Handle seek request from trackbar or keyboard shortcuts
            if seek_target_sec is not None and source_kind in ("youtube", "file"):
                target_sec = max(0, min(total_seconds, int(seek_target_sec))) if total_seconds > 0 else max(0, int(seek_target_sec))
                seek_target_sec = None
                cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
                status_notification = f"⏩ Premotano na: {target_sec // 60:02d}:{target_sec % 60:02d}"
                status_notification_time = time.time()
                ret_s, frame_s = cap.read()
                if ret_s:
                    frame = frame_s
                    if paused:
                        paused_frame = frame_s.copy()
                    # Trigger instant face recognition on seeked frame
                    try:
                        detected = face_engine.extract_faces_from_image(frame, device=device, with_attributes=False)
                        new_tracked = []
                        for f in detected:
                            bbox = f["bbox"]
                            emb = f.get("embedding")
                            if emb is None:
                                continue
                            match = face_idx.match(emb, threshold=threshold)
                            sim = float(match.get("similarity", 0.0))
                            sim_pct = sim * 100
                            p_name = match.get("person_name") or match.get("best_name", "Nepoznato")
                            if match.get("matched", False):
                                color = (16, 185, 129)
                                label = f"{p_name} ({sim_pct:.1f}%)"
                                status_type = "match"
                                record_event_if_eligible(p_name, sim, f.get("crop_bgr", None))
                            elif sim >= max(0.30, threshold - possible_threshold_delta):
                                color = (11, 158, 245)
                                label = f"Moguce: {p_name} ({sim_pct:.1f}%)"
                                status_type = "possible"
                            else:
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
                    except Exception as err:
                        log_debug(f"Greska brzog prepoznavanja pri premotavanju: {err}")

            if not paused:
                ret, frame = cap.read()
                if not ret:
                    if source_kind in ("file", "youtube"):
                        log_debug("Dosegnut kraj video zapisa.")
                        print("[INFO] Dosegnut kraj video zapisa.")
                        if 'display_frame' in locals() and display_frame is not None:
                            end_frame = display_frame.copy()
                            cv2.rectangle(end_frame, (0, 0), (end_frame.shape[1], 55), (15, 23, 42), -1)
                            cv2.putText(end_frame, "Videozapis je zavrsio. Pritisnite bilo koju tipku za izlaz...", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.6, (34, 197, 94), 1, cv2.LINE_AA)
                            cv2.imshow(window_name, end_frame)
                            cv2.waitKey(6000)
                        break
                    else:
                        print("⚠️ Gubitak signala s kamere.")
                        time.sleep(0.1)
                        continue
                frame_count += 1
            else:
                frame = paused_frame.copy()
                
            cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC) if source_kind in ("youtube", "file") else 0.0
            cur_sec = max(0.0, cur_msec / 1000.0)

            # Periodically update trackbar position to reflect current playback time
            if has_trackbar and (frame_count % 15 == 0) and not paused:
                is_programmatic_trackbar_update = True
                try:
                    cv2.setTrackbarPos(trackbar_name, window_name, int(cur_sec))
                except Exception:
                    pass
                is_programmatic_trackbar_update = False

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
                try:
                    # Analyze frame with RetinaFace + ArcFace
                    detected = face_engine.extract_faces_from_image(frame, device=device, with_attributes=False)
                    new_tracked = []
                    
                    for f in detected:
                        bbox = f["bbox"]
                        emb = f.get("embedding")
                        if emb is None:
                            continue
                        
                        # Sub-millisecond BLAS matching against database
                        match = face_idx.match(emb, threshold=threshold)
                        
                        sim = float(match.get("similarity", 0.0))
                        sim_pct = sim * 100
                        p_name = match.get("person_name") or match.get("best_name", "Nepoznato")
                        
                        if match.get("matched", False):
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
                            "emb": emb,
                            "raw_name": p_name,
                            "similarity": sim
                        })
                    current_faces_tracked = new_tracked
                except Exception as frame_err:
                    print(f"[UPOZORENJE] Greška obrade lica u kadru: {frame_err}")

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

            # Record detection event with full camera picture (with annotated bounding box)
            if log_events and current_faces_tracked and not paused:
                for face in current_faces_tracked:
                    if face.get("status") == "match":
                        record_event_if_eligible(
                            person_name=face.get("raw_name", "Osoba"),
                            similarity=face.get("similarity", 0.0),
                            crop_bgr=face.get("crop"),
                            full_frame=display_frame
                        )

            # Notification timeout (2.5 seconds)
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
                status_msg=active_msg,
                camera_label=source_label,
                cur_sec=cur_sec,
                total_sec=total_seconds,
                log_events=log_events
            )
            
            cv2.imshow(window_name, display_frame)

            # Ensure OpenCV window pops up in front of the console window on launch
            if frame_count <= 2:
                try:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
                    import ctypes
                    hwnd = ctypes.windll.user32.FindWindowW(None, window_name)
                    if hwnd:
                        ctypes.windll.user32.ShowWindow(hwnd, 9) # SW_RESTORE
                        ctypes.windll.user32.SetForegroundWindow(hwnd)
                        ctypes.windll.user32.BringWindowToTop(hwnd)
                except Exception:
                    pass
            elif frame_count == 3:
                try:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)
                except Exception:
                    pass

            # Keep real-time playback pacing for video files and YouTube streams
            if source_kind in ("file", "youtube") and target_frame_time > 0 and not paused:
                elapsed_render = time.perf_counter() - now
                sleep_remainder = target_frame_time - elapsed_render
                if sleep_remainder > 0:
                    time.sleep(sleep_remainder)

            # Keyboard controls (must be called immediately after imshow to pump window events)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')): # ESC or Q -> Exit
                log_debug(f"Petlja prekinuta tipkom na tipkovnici: key={key}")
                break
            elif key in (ord(' '), ord('k'), ord('K')): # Space or K -> Pause
                paused = not paused
                if paused:
                    paused_frame = frame.copy()
                    status_notification = "⏸️ Slika zamrznuta (Pauza). Pritisnite SPACE za nastavak."
                else:
                    status_notification = "▶️ Nastavak snimanja uzivo."
                status_notification_time = time.time()
            elif key in (ord('e'), ord('E')): # E -> Toggle Event Logging
                log_events = not log_events
                if log_events:
                    status_notification = f"📋 Evidencija prolazaka UKLJUČENA (Cooldown: {cooldown_sec}s)"
                    print(f"[INFO] 📋 Evidencija prolazaka UKLJUČENA (Cooldown: {cooldown_sec}s)")
                else:
                    status_notification = "⏸️ Evidencija prolazaka ISKLJUČENA"
                    print("[INFO] ⏸️ Evidencija prolazaka ISKLJUČENA")
                status_notification_time = time.time()
            elif key in (ord('a'), ord('A'), ord('j'), ord('J')): # Seek -10s
                if source_kind in ("youtube", "file"):
                    seek_target_sec = max(0, cur_sec - 10)
            elif key in (ord('d'), ord('D'), ord('l'), ord('L')): # Seek +10s
                if source_kind in ("youtube", "file"):
                    seek_target_sec = min(total_seconds, cur_sec + 10) if total_seconds > 0 else cur_sec + 10
            elif key in (ord(','), ): # Seek -30s
                if source_kind in ("youtube", "file"):
                    seek_target_sec = max(0, cur_sec - 30)
            elif key in (ord('.'), ): # Seek +30s
                if source_kind in ("youtube", "file"):
                    seek_target_sec = min(total_seconds, cur_sec + 30) if total_seconds > 0 else cur_sec + 30
            elif key in (ord('s'), ord('S')): # S -> Snapshot
                t_str = time.strftime("%Y%m%d_%H%M%S")
                snap_dir = config.get_snapshot_dir()
                snap_filename = f"live_snap_{t_str}.jpg"
                snap_path = os.path.join(snap_dir, snap_filename)
                target_img = display_frame.copy() if 'display_frame' in locals() and display_frame is not None else frame
                imwrite_unicode(snap_path, target_img)
                status_notification = f"📸 Kadar spremljen u: {snap_filename} (Tipka 'O' za mapu)"
                status_notification_time = time.time()
                print(f"[OK] Snimka kadra spremljena: {snap_path}")
            elif key in (ord('o'), ord('O')): # O -> Open folder in Windows Explorer
                config.open_folder_in_explorer()
                status_notification = "📂 Otvorena mapa sa snimkama u Exploreru."
                status_notification_time = time.time()
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
                
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        log_debug(f"IZNIMKA u petlji kamere: {err_msg}")
        print(f"\n❌ [GREŠKA] Neočekivana greška u radu live kamere: {e}")
        traceback.print_exc()
        time.sleep(3)
    finally:
        log_debug(f"Gasim kameru i zatvaram prozor (ukupno kadrova: {frame_count}).")
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera uspjesno oslobodena i ugasena.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UniFace Live Face Recognition")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or RTSP/HTTP URL")
    parser.add_argument("--camera", type=str, default=None, help="Legacy alias for camera source")
    parser.add_argument("--threshold", type=float, default=0.45, help="Recognition cosine similarity threshold (default: 0.45)")
    parser.add_argument("--skip", type=int, default=2, help="Process every N frames (default: 2)")
    parser.add_argument("--device", type=str, default="CPU", help="Inference device: CPU or CUDA (default: CPU)")
    parser.add_argument("--start", type=int, default=0, help="Start position in seconds for video/youtube (default: 0)")
    parser.add_argument("--log-events", action="store_true", default=False, help="Enable automatic detection event logging")
    parser.add_argument("--cooldown", type=int, default=30, help="Cooldown in seconds between re-logging same person (default: 30)")
    args = parser.parse_args()
    
    src = args.camera if args.camera is not None else args.source

    run_live_camera(
        camera_source=src,
        threshold=args.threshold,
        process_interval=args.skip,
        device=args.device,
        start_sec=args.start,
        log_events=args.log_events,
        cooldown_sec=args.cooldown
    )
