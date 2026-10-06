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
import hardware
import notifier
from image_utils import imwrite_unicode
from nvr_recorder import get_nvr_manager

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

def draw_corner_box(img, pt1, pt2, color, thickness=1, corner_len=14):
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

def draw_hud(frame, fps, num_faces, threshold, total_persons, paused=False, status_msg="", camera_label="USB Web Kamera", cur_sec=0.0, total_sec=0.0, log_events=False, record_nvr=False, only_matched=False, anti_spoof=True, hover_x=-1, hover_y=-1, is_dragging=False, drag_sec=None):
    """Renders sleek top and bottom HUD panels with live telemetry and interactive timeline."""
    h, w = frame.shape[:2]
    
    # 1. Top HUD bar
    top_bar = frame[0:55, 0:w].copy()
    overlay = np.zeros_like(top_bar)
    cv2.rectangle(overlay, (0, 0), (w, 55), (22, 27, 34), -1) # Dark slate
    cv2.addWeighted(overlay, 0.78, top_bar, 0.22, 0, top_bar)
    frame[0:55, 0:w] = top_bar
    cv2.line(frame, (0, 55), (w, 55), (38, 44, 54), 1) # Blue accent line
    
    # Top HUD text
    cv2.putText(frame, "ArgusFace Live Feed", (15, 26), cv2.FONT_HERSHEY_DUPLEX, 0.68, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Subtitle: camera label and badges
    sub_title = camera_label
    if log_events:
        sub_title += " | [📋 EVIDENCIJA]"
    if anti_spoof:
        sub_title += " | [🛡️ ANTI-SPOOF: ON]"
    else:
        sub_title += " | [⚠️ ANTI-SPOOF: OFF (F)]"
    cv2.putText(frame, sub_title, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (148, 163, 184), 1, cv2.LINE_AA)
    
    # Stats on top right
    stats_x = max(260, w - 460)
    status_indicator = "|| PAUZIRANO" if paused else f"{fps:.1f} FPS"
    fps_color = (0, 165, 255) if paused else (16, 185, 129) # Orange or green
    cv2.putText(frame, f"Brzina: {status_indicator}", (stats_x, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fps_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"Lica u kadru: {num_faces}", (stats_x, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (226, 232, 240), 1, cv2.LINE_AA)
    
    # NVR REC Badge
    if record_nvr:
        cv2.circle(frame, (stats_x - 30, 24), 6, (0, 0, 235), -1)
        cv2.putText(frame, "REC", (stats_x - 18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"Prag: {threshold:.2f}", (stats_x + 180, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 158, 11), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Baza: {total_persons} osoba", (stats_x + 180, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (203, 213, 225), 1, cv2.LINE_AA)

    # 2. Bottom HUD & Interactive Timeline
    bot_h = 46 if total_sec > 0 else 38
    bot_y = h - bot_h
    bot_bar = frame[bot_y:h, 0:w].copy()
    b_overlay = np.zeros_like(bot_bar)
    cv2.rectangle(b_overlay, (0, 0), (w, bot_h), (15, 23, 42), -1)
    cv2.addWeighted(b_overlay, 0.85, bot_bar, 0.15, 0, bot_bar)
    frame[bot_y:h, 0:w] = bot_bar
    cv2.line(frame, (0, bot_y), (w, bot_y), (30, 41, 59), 1)

    # If video timeline exists (YouTube or local video file)
    if total_sec > 0:
        bar_x1 = 12
        bar_x2 = w - 12
        bar_w = max(1, bar_x2 - bar_x1)
        
        is_hover_timeline = (bar_x1 <= hover_x <= bar_x2 and (bot_y - 12) <= hover_y <= h)
        
        # Track height & Y
        track_h = 8 if (is_hover_timeline or is_dragging) else 6
        track_y1 = bot_y + (5 if not (is_hover_timeline or is_dragging) else 4)
        track_y2 = track_y1 + track_h
        
        # Background bar
        cv2.rectangle(frame, (bar_x1, track_y1), (bar_x2, track_y2), (30, 41, 59), -1)
        cv2.rectangle(frame, (bar_x1, track_y1), (bar_x2, track_y2), (51, 65, 85), 1)
        
        # Hover ghost bar (semi-transparent guide up to hover cursor)
        if is_hover_timeline and hover_x > bar_x1:
            ghost_w = min(bar_x2, hover_x)
            cv2.rectangle(frame, (bar_x1, track_y1), (ghost_w, track_y2), (71, 85, 105), -1)
            
        # Progress fill
        active_sec = drag_sec if (is_dragging and drag_sec is not None) else cur_sec
        pct = max(0.0, min(1.0, float(active_sec) / float(total_sec)))
        fill_w = int(bar_w * pct)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x1, track_y1), (bar_x1 + fill_w, track_y2), (246, 130, 59), -1)
            
        # Scrubber Thumb (glowing circular handle)
        thumb_cx = bar_x1 + fill_w
        thumb_cy = (track_y1 + track_y2) // 2
        outer_r = 8 if (is_hover_timeline or is_dragging) else 6
        cv2.circle(frame, (thumb_cx, thumb_cy), outer_r, (246, 130, 59), -1, cv2.LINE_AA)
        cv2.circle(frame, (thumb_cx, thumb_cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
        
        # Tooltip badge on hover or dragging
        if is_hover_timeline or is_dragging:
            h_pct = max(0.0, min(1.0, float(hover_x - bar_x1) / float(bar_w)))
            target_tip_sec = drag_sec if (is_dragging and drag_sec is not None) else (h_pct * float(total_sec))
            tm, ts = int(target_tip_sec) // 60, int(target_tip_sec) % 60
            tip_str = f"Skok: {tm:02d}:{ts:02d}" if is_dragging else f"{tm:02d}:{ts:02d}"
            
            (tw, th), _ = cv2.getTextSize(tip_str, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
            bx1 = max(10, min(w - tw - 20, hover_x - (tw + 16) // 2))
            by1 = bot_y - 28
            bx2 = bx1 + tw + 16
            by2 = by1 + th + 10
            
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (15, 23, 42), -1)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (246, 130, 59), 1)
            cv2.putText(frame, tip_str, (bx1 + 8, by2 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(frame, (hover_x, track_y1 - 2), (hover_x, track_y2 + 2), (255, 255, 255), 1)

    # Bottom status / keyboard row
    row_y = h - 11
    
    # Right-hand video time string
    time_w = 0
    if total_sec > 0:
        c_min, c_s = int(cur_sec) // 60, int(cur_sec) % 60
        t_min, t_s = int(total_sec) // 60, int(total_sec) % 60
        time_str = f"{c_min:02d}:{c_s:02d} / {t_min:02d}:{t_s:02d}"
        (time_w, _), _ = cv2.getTextSize(time_str, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
        cv2.putText(frame, time_str, (w - time_w - 15, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.44, (226, 232, 240), 1, cv2.LINE_AA)

    # Left-hand status notification or controls shortcuts
    if status_msg:
        cv2.putText(frame, status_msg, (15, row_y), cv2.FONT_HERSHEY_DUPLEX, 0.50, (34, 197, 94), 1, cv2.LINE_AA)
    else:
        if total_sec > 0:
            controls_txt = "[SPACE] Pauza  [<-/->] +-5s  [A/D] +-10s  [,/.] Kadar  [0-9] %  [F] AntiSpoof  [S] Kadar  [Q] Izlaz"
            (ctw, _), _ = cv2.getTextSize(controls_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            if ctw > (w - time_w - 40):
                controls_txt = "[SPACE] Pauza  [<-/->] +-5s  [,/.] Kadar  [0-9] %  [Q] Izlaz"
        else:
            filter_str = "Samo Zelena" if only_matched else "Sva Lica"
            spoof_str = "AntiSpoof:ON" if anti_spoof else "AntiSpoof:OFF"
            controls_txt = f"[SPACE] Pauza  [F] {spoof_str}  [M] {filter_str}  [R] NVR  [E] Evidencija  [S] Kadar  [Q] Izlaz"
            
        cv2.putText(frame, controls_txt, (15, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (148, 163, 184), 1, cv2.LINE_AA)
_STREAM_DURATIONS = {}

def get_stream_duration(source_input):
    """Vraća trajanje videa u sekundama ako je poznato iz metapodataka."""
    return _STREAM_DURATIONS.get(str(source_input).strip(), 0.0)

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
                            dur = info.get('duration', 0.0)
                            if dur:
                                _STREAM_DURATIONS[source_str] = float(dur)
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

def run_live_camera(camera_source=0, threshold=0.45, process_interval=2, device="AUTO", start_sec=0, log_events=False, cooldown_sec=30, record_nvr=False, segment_duration_sec=300, max_storage_gb=20.0, only_matched=False, anti_spoof=None):
    """
    Main loop for live face recognition from webcam, IP/RTSP camera, YouTube, or video file.
    Includes active Anti-Spoofing / Presentation Attack Detection (MiniFASNet V2).
    """
    if anti_spoof is None:
        anti_spoof = config.get_anti_spoofing()

    source_str = str(camera_source).strip()
    log_debug(f"run_live_camera pokrenut: camera_source={source_str}, threshold={threshold}, interval={process_interval}, start_sec={start_sec}, log_events={log_events}, cooldown={cooldown_sec}, record_nvr={record_nvr}, only_matched={only_matched}, anti_spoof={anti_spoof}, device={device}")
    
    accel_type, accel_status = hardware.get_onnx_acceleration_status()
    print("===================================================")
    print("      ArgusFace Live Feed - Prepoznavanje Lica")
    print(f"      [{accel_status} ({accel_type})]")
    if anti_spoof:
        print("      [🛡️ Anti-Spoofing: AKTIVAN (Blokira fotografije i ekrane)]")
    else:
        print("      [⚠️ Anti-Spoofing: ISKLJUČEN (Dozvoljena identifikacija sa slika)]")
    if log_events:
        print(f"      [📋 Evidencija prolazaka: AKTIVNA | Cooldown: {cooldown_sec}s]")
    if record_nvr:
        print(f"      [🔴 NVR Snimanje: AKTIVNO | Segment: {segment_duration_sec}s]")
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
        
    window_name = "ArgusFace Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    # For YouTube and video files: setup timeline duration and interactive on-screen controls
    meta_dur = get_stream_duration(source_str) or get_stream_duration(camera_source)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) if source_kind in ("youtube", "file") else 0
    effective_fps = native_fps if (native_fps and native_fps > 0) else 25.0
    if meta_dur and meta_dur > 0:
        total_seconds = int(meta_dur)
    elif total_frames > 0 and effective_fps > 0:
        total_seconds = int(total_frames / effective_fps)
    else:
        total_seconds = 0
    
    seek_target_sec = None
    timeline_hover_x = -1
    timeline_hover_y = -1
    is_mouse_dragging = False
    mouse_drag_sec = None
    cur_frame_w = 1280
    cur_frame_h = 720

    def on_mouse_event(event, x, y, flags, param):
        nonlocal timeline_hover_x, timeline_hover_y, is_mouse_dragging, mouse_drag_sec
        nonlocal seek_target_sec, paused, paused_frame, status_notification, status_notification_time
        
        timeline_hover_x = x
        timeline_hover_y = y
        
        if total_seconds <= 0:
            return
            
        bar_x1 = 12
        bar_x2 = cur_frame_w - 12
        bar_w = max(1, bar_x2 - bar_x1)
        bar_y_min = cur_frame_h - 52
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if bar_y_min <= y <= cur_frame_h:
                is_mouse_dragging = True
                norm_pct = max(0.0, min(1.0, float(x - bar_x1) / float(bar_w)))
                mouse_drag_sec = norm_pct * float(total_seconds)
        elif event == cv2.EVENT_MOUSEMOVE:
            if is_mouse_dragging:
                norm_pct = max(0.0, min(1.0, float(x - bar_x1) / float(bar_w)))
                mouse_drag_sec = norm_pct * float(total_seconds)
        elif event == cv2.EVENT_LBUTTONUP:
            if is_mouse_dragging:
                is_mouse_dragging = False
                norm_pct = max(0.0, min(1.0, float(x - bar_x1) / float(bar_w)))
                seek_target_sec = norm_pct * float(total_seconds)
                mouse_drag_sec = None
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if y < bar_y_min and y > 60:
                paused = not paused
                if paused:
                    paused_frame = frame.copy() if 'frame' in locals() and frame is not None else None
                    status_notification = "⏸️ Slika zamrznuta (Pauza). Pritisnite SPACE za nastavak."
                else:
                    status_notification = "▶️ Nastavak reprodukcije."
                status_notification_time = time.time()

    cv2.setMouseCallback(window_name, on_mouse_event)

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
    if anti_spoof:
        print("[INFO] Ucitavam MiniFASNet anti-spoofing model (detekcija zivosti)...")
        face_engine.get_spoofer(device=device)
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

    # NVR Manager initialization
    nvr = get_nvr_manager()
    nvr.max_storage_gb = max_storage_gb
    cam_nvr_id = 1
    if record_nvr:
        nvr.configure_channel(cam_nvr_id, source_label, segment_duration_sec=segment_duration_sec, fps=20.0)
    
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

            vid_path, vid_offset = nvr.get_bookmark(cam_nvr_id) if record_nvr else ("", 0.0)

            try:
                db.log_detection_event(
                    person_name=person_name,
                    similarity=similarity,
                    source_label=source_label,
                    crop_path=crop_saved_path,
                    snapshot_path=snap_saved_path,
                    video_path=vid_path,
                    video_offset_sec=vid_offset
                )
                sim_pct = similarity * 100
                bmark_str = f" [Video: {os.path.basename(vid_path)} @ {vid_offset:.1f}s]" if vid_path else ""
                status_notification = f"📋 Evidentiran prolazak: {person_name} ({sim_pct:.1f}%)"
                status_notification_time = time.time()
                print(f"[EVIDENCIJA] Zabilježen prolazak: {person_name} ({sim_pct:.1f}%) [{source_label}]{bmark_str}")
            except Exception as log_err:
                print(f"[UPOZORENJE] Greška evidentiranja u dnevnik: {log_err}")

    def process_faces_for_frame(frame_to_process):
        try:
            detected = face_engine.extract_faces_from_image(frame_to_process, device=device, with_attributes=False)
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

                is_real = True
                if anti_spoof:
                    is_real, _ = face_engine.check_liveness(frame_to_process, bbox, device=device)

                if anti_spoof and not is_real:
                    color = (0, 0, 255)
                    if match.get("matched", False):
                        label = f"⚠ LAZIRANO: {p_name} ({sim_pct:.0f}%)"
                    else:
                        label = "⚠ LAZIRANO: Ekran/Slika"
                    status_type = "spoof"
                    notifier.trigger_alert_async("spoof", p_name if match.get("matched", False) else "Nepoznato", sim, source_label, frame_to_process, f.get("crop_bgr"))
                elif match.get("matched", False):
                    p_role = db.get_person_role(p_name)
                    if p_role == "blacklist":
                        color = (0, 0, 255)
                        label = f"🚨 CRNA LISTA: {p_name} ({sim_pct:.1f}%)"
                        status_type = "match"
                        notifier.trigger_alert_async("blacklist", p_name, sim, source_label, frame_to_process, f.get("crop_bgr"))
                    elif p_role == "vip":
                        color = (245, 185, 11)
                        label = f"⭐ VIP: {p_name} ({sim_pct:.1f}%)"
                        status_type = "match"
                        notifier.trigger_alert_async("vip", p_name, sim, source_label, frame_to_process, f.get("crop_bgr"))
                    else:
                        color = (16, 185, 129)
                        label = f"{p_name} ({sim_pct:.1f}%)"
                        status_type = "match"
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
                    "emb": emb,
                    "raw_name": p_name,
                    "similarity": sim
                })
            return new_tracked
        except Exception as frame_err:
            log_debug(f"Greška analize lica: {frame_err}")
            return []

    try:
        while True:
            # Handle seek request from interactive timeline or keyboard shortcuts
            if seek_target_sec is not None and source_kind in ("youtube", "file"):
                target_sec = max(0.0, min(float(total_seconds), float(seek_target_sec))) if total_seconds > 0 else max(0.0, float(seek_target_sec))
                seek_target_sec = None
                
                if source_kind == "file" and effective_fps > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(target_sec * effective_fps))
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC, target_sec * 1000.0)
                    
                sm, ss = int(target_sec) // 60, int(target_sec) % 60
                status_notification = f"⏩ Premotano na: {sm:02d}:{ss:02d}"
                status_notification_time = time.time()
                ret_s, frame_s = cap.read()
                if ret_s:
                    frame = frame_s
                    if paused:
                        paused_frame = frame_s.copy()
                    current_faces_tracked = process_faces_for_frame(frame)

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
                frame = paused_frame.copy() if paused_frame is not None else frame

            # Update current resolution for mouse mapping
            cur_frame_h, cur_frame_w = frame.shape[:2]

            cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC) if source_kind in ("youtube", "file") else 0.0
            cur_sec = max(0.0, cur_msec / 1000.0)
            if cur_sec <= 0.0 and source_kind in ("youtube", "file"):
                cur_f = cap.get(cv2.CAP_PROP_POS_FRAMES)
                if cur_f > 0 and effective_fps > 0:
                    cur_sec = cur_f / effective_fps

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
                current_faces_tracked = process_faces_for_frame(frame)

            # Render tracked faces onto frame
            display_frame = frame.copy()
            for face in current_faces_tracked:
                if only_matched and face.get("status") != "match":
                    continue
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

            # Draw HUD with interactive timeline
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
                log_events=log_events,
                record_nvr=record_nvr,
                only_matched=only_matched,
                anti_spoof=anti_spoof,
                hover_x=timeline_hover_x,
                hover_y=timeline_hover_y,
                is_dragging=is_mouse_dragging,
                drag_sec=mouse_drag_sec
            )

            # Write frame to NVR if active
            if record_nvr and not paused and display_frame is not None:
                nvr.write_frame(cam_nvr_id, display_frame)
            
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

            # Keyboard controls (cv2.waitKeyEx pumps Windows GUI events and returns extended codes)
            key_raw = cv2.waitKeyEx(25 if paused else 1)
            if key_raw != -1:
                key = key_raw & 0xFF
                
                # ESC or Q -> Exit
                if key in (27, ord('q'), ord('Q')):
                    log_debug(f"Petlja prekinuta tipkom na tipkovnici: key={key_raw}")
                    break
                    
                # SPACE or K -> Pause / Play
                elif key in (ord(' '), ord('k'), ord('K')):
                    paused = not paused
                    if paused:
                        paused_frame = frame.copy()
                        status_notification = "⏸️ Slika zamrznuta (Pauza). Pritisnite SPACE za nastavak."
                    else:
                        status_notification = "▶️ Nastavak reprodukcije."
                    status_notification_time = time.time()
                    
                # Frame-by-frame forward: [.] or [>] or Right Arrow while paused
                elif (key in (ord('.'), ord('>')) and paused) or (key_raw in (2555904, 65363) and paused):
                    if source_kind in ("youtube", "file"):
                        ret_step, frame_step = cap.read()
                        if ret_step:
                            frame = frame_step
                            paused_frame = frame_step.copy()
                            frame_count += 1
                            current_faces_tracked = process_faces_for_frame(frame)
                            cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                            cur_sec = max(0.0, cur_msec / 1000.0) if cur_msec > 0 else (frame_count / effective_fps)
                            c_min, c_s = int(cur_sec) // 60, int(cur_sec) % 60
                            status_notification = f"▶| Kadar +1 ({c_min:02d}:{c_s:02d})"
                            status_notification_time = time.time()
                            
                # Frame-by-frame backward: [,] or [<] or Left Arrow while paused
                elif (key in (ord(','), ord('<')) and paused) or (key_raw in (2424832, 65361) and paused):
                    if source_kind in ("youtube", "file"):
                        cur_f = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                        target_f = max(0, cur_f - 2)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
                        ret_step, frame_step = cap.read()
                        if ret_step:
                            frame = frame_step
                            paused_frame = frame_step.copy()
                            current_faces_tracked = process_faces_for_frame(frame)
                            cur_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                            cur_sec = max(0.0, cur_msec / 1000.0) if cur_msec > 0 else (target_f / effective_fps)
                            c_min, c_s = int(cur_sec) // 60, int(cur_sec) % 60
                            status_notification = f"|◀ Kadar -1 ({c_min:02d}:{c_s:02d})"
                            status_notification_time = time.time()
                            
                # Seek -5s: Left Arrow (while playing) or [,] (while playing)
                elif (key_raw in (2424832, 65361) and not paused) or (key in (ord(','), ord('<')) and not paused):
                    if source_kind in ("youtube", "file"):
                        seek_target_sec = max(0.0, cur_sec - 5.0)
                        
                # Seek +5s: Right Arrow (while playing) or [.] (while playing)
                elif (key_raw in (2555904, 65363) and not paused) or (key in (ord('.'), ord('>')) and not paused):
                    if source_kind in ("youtube", "file"):
                        seek_target_sec = min(float(total_seconds), cur_sec + 5.0) if total_seconds > 0 else cur_sec + 5.0
                        
                # Seek -10s: [A] or [J]
                elif key in (ord('a'), ord('A'), ord('j'), ord('J')):
                    if source_kind in ("youtube", "file"):
                        seek_target_sec = max(0.0, cur_sec - 10.0)
                        
                # Seek +10s: [D] or [L]
                elif key in (ord('d'), ord('D'), ord('l'), ord('L')):
                    if source_kind in ("youtube", "file"):
                        seek_target_sec = min(float(total_seconds), cur_sec + 10.0) if total_seconds > 0 else cur_sec + 10.0
                        
                # Jump to percentage: [0] - [9]
                elif ord('0') <= key <= ord('9'):
                    if source_kind in ("youtube", "file") and total_seconds > 0:
                        pct_val = (key - ord('0')) / 10.0
                        seek_target_sec = pct_val * float(total_seconds)
                        status_notification = f"⏩ Skok na {int(pct_val * 100)}% videa"
                        status_notification_time = time.time()
                        
                # R -> NVR Record toggle
                elif key in (ord('r'), ord('R')):
                    record_nvr = not record_nvr
                    if record_nvr:
                        nvr.configure_channel(cam_nvr_id, source_label, segment_duration_sec=segment_duration_sec, fps=20.0)
                        status_notification = "🔴 NVR Snimanje: AKTIVNO"
                    else:
                        nvr.stop_all()
                        status_notification = "NVR Snimanje: ISKLJUCENO"
                    status_notification_time = time.time()
                    
                # U -> Refresh DB cache
                elif key in (ord('u'), ord('U')):
                    db.invalidate_cache()
                    face_idx = face_engine.get_face_index()
                    stats = db.get_stats()
                    total_persons = stats["total_persons"]
                    status_notification = f"Baza osvjezena! Osoba: {total_persons}"
                    status_notification_time = time.time()
                    print(f"[OK] {status_notification}")
                    
                # E -> Toggle Event Logging
                elif key in (ord('e'), ord('E')):
                    log_events = not log_events
                    if log_events:
                        status_notification = f"📋 Evidencija prolazaka UKLJUČENA (Cooldown: {cooldown_sec}s)"
                        print(f"[INFO] 📋 Evidencija prolazaka UKLJUČENA (Cooldown: {cooldown_sec}s)")
                    else:
                        status_notification = "⏸️ Evidencija prolazaka ISKLJUČENA"
                        print("[INFO] ⏸️ Evidencija prolazaka ISKLJUČENA")
                    status_notification_time = time.time()
                    
                # M -> Toggle only matched (green) vs all faces
                elif key in (ord('m'), ord('M')):
                    only_matched = not only_matched
                    status_notification = "Filter: SAMO PREPOZNATA LICA (Zelena)" if only_matched else "Filter: SVA LICA (Ukljucujuci nepoznata)"
                    status_notification_time = time.time()
                    print(f"[INFO] {status_notification}")
                    
                # F -> Toggle Anti-Spoofing / Presentation Attack Check
                elif key in (ord('f'), ord('F')):
                    anti_spoof = not anti_spoof
                    config.set_anti_spoofing(anti_spoof)
                    if anti_spoof:
                        status_notification = "🛡️ Zastita od laziranja (Anti-Spoof): UKLJUCENA"
                        print("[INFO] 🛡️ Zaštita od lažiranja (Anti-Spoof): UKLJUČENA (Blokira fotografije i ekrane)")
                    else:
                        status_notification = "⚠️ Zastita od laziranja: ISKLJUCENA (Dozvoljena identifikacija sa slika)"
                        print("[INFO] ⚠️ Zaštita od lažiranja: ISKLJUČENA (Dozvoljena identifikacija sa slika/mobitela)")
                    status_notification_time = time.time()
                    
                # S -> Snapshot
                elif key in (ord('s'), ord('S')):
                    t_str = time.strftime("%Y%m%d_%H%M%S")
                    snap_dir = config.get_snapshot_dir()
                    snap_filename = f"live_snap_{t_str}.jpg"
                    snap_path = os.path.join(snap_dir, snap_filename)
                    target_img = display_frame.copy() if 'display_frame' in locals() and display_frame is not None else frame
                    imwrite_unicode(snap_path, target_img)
                    status_notification = f"📸 Kadar spremljen u: {snap_filename} (Tipka 'O' za mapu)"
                    status_notification_time = time.time()
                    print(f"[OK] Snimka kadra spremljena: {snap_path}")
                    
                # O -> Open folder in Windows Explorer
                elif key in (ord('o'), ord('O')):
                    config.open_folder_in_explorer()
                    status_notification = "📂 Otvorena mapa sa snimkama u Exploreru."
                    status_notification_time = time.time()
                    
                # Up Arrow or [+] -> Increase threshold
                elif key_raw in (2490368, 65362) or key in (ord('+'), ord('=')):
                    threshold = min(0.95, round(threshold + 0.02, 2))
                    status_notification = f"Prag povecan na: {threshold:.2f} (stroze)"
                    status_notification_time = time.time()
                    
                # Down Arrow or [-] -> Decrease threshold
                elif key_raw in (2621440, 65364) or key in (ord('-'), ord('_')):
                    threshold = max(0.20, round(threshold - 0.02, 2))
                    status_notification = f"Prag smanjen na: {threshold:.2f} (blaze)"
                    status_notification_time = time.time()
                
    except Exception as e:
        import traceback
        err_msg = traceback.format_exc()
        log_debug(f"IZNIMKA u petlji kamere: {err_msg}")
        print(f"\n❌ [GREŠKA] Neočekivana greška u radu live kamere: {e}")
        traceback.print_exc()
        time.sleep(3)
    finally:
        log_debug(f"Gasim kameru i zatvaram prozor (ukupno kadrova: {frame_count}).")
        try:
            nvr.stop_all()
        except Exception:
            pass
        cap.release()
        cv2.destroyAllWindows()
        print("Kamera uspjesno oslobodena i ugasena.")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argusface Live Face Recognition")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or RTSP/HTTP URL")
    parser.add_argument("--camera", type=str, default=None, help="Legacy alias for camera source")
    parser.add_argument("--threshold", type=float, default=0.45, help="Recognition cosine similarity threshold (default: 0.45)")
    parser.add_argument("--skip", type=int, default=2, help="Process every N frames (default: 2)")
    parser.add_argument("--device", type=str, default="AUTO", help="Inference device: AUTO, DIRECTML, CUDA, OPENVINO, or CPU (default: AUTO)")
    parser.add_argument("--start", type=int, default=0, help="Start position in seconds for video/youtube (default: 0)")
    parser.add_argument("--log-events", action="store_true", default=False, help="Enable automatic detection event logging")
    parser.add_argument("--cooldown", type=int, default=30, help="Cooldown in seconds between re-logging same person (default: 30)")
    parser.add_argument("--record-nvr", action="store_true", default=False, help="Enable continuous NVR MP4 segment recording")
    parser.add_argument("--segment-min", type=int, default=5, help="Duration of each MP4 video segment in minutes (default: 5)")
    parser.add_argument("--max-gb", type=float, default=20.0, help="Maximum disk storage quota in GB for FIFO cleanup (default: 20)")
    parser.add_argument("--only-matched", action="store_true", default=False, help="Display bounding boxes only for recognized faces (green)")
    parser.add_argument("--anti-spoof", dest="anti_spoof", action="store_true", default=None, help="Enable presentation attack / photo liveness check (default: True)")
    parser.add_argument("--no-anti-spoof", dest="anti_spoof", action="store_false", help="Disable anti-spoofing to allow identifying persons from phone photos")
    args = parser.parse_args()
    
    src = args.camera if args.camera is not None else args.source

    run_live_camera(
        camera_source=src,
        threshold=args.threshold,
        process_interval=args.skip,
        device=args.device,
        start_sec=args.start,
        log_events=args.log_events,
        cooldown_sec=args.cooldown,
        record_nvr=args.record_nvr,
        segment_duration_sec=args.segment_min * 60,
        max_storage_gb=args.max_gb,
        only_matched=args.only_matched,
        anti_spoof=args.anti_spoof
    )
