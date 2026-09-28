import os
import sys
import time
import threading
import cv2
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")
DEFAULT_RECORDINGS_DIR = os.path.join(DATA_DIR, "recordings")
os.makedirs(DEFAULT_RECORDINGS_DIR, exist_ok=True)

def cleanup_old_recordings(recordings_dir: str, max_storage_gb: float = 20.0):
    """
    FIFO policy: If total size of video recordings exceeds max_storage_gb,
    automatically deletes the oldest video files until under 85% of quota.
    """
    if not os.path.isdir(recordings_dir) or max_storage_gb <= 0:
        return
        
    max_bytes = max_storage_gb * 1024 * 1024 * 1024
    target_bytes = max_bytes * 0.85

    video_files = []
    total_bytes = 0

    for root, _, files in os.walk(recordings_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv")):
                fpath = os.path.join(root, f)
                try:
                    fsize = os.path.getsize(fpath)
                    fmtime = os.path.getmtime(fpath)
                    video_files.append((fmtime, fsize, fpath))
                    total_bytes += fsize
                except Exception:
                    pass

    if total_bytes > max_bytes:
        # Sort oldest first
        video_files.sort(key=lambda x: x[0])
        freed = 0
        for mtime, size, fpath in video_files:
            try:
                os.remove(fpath)
                total_bytes -= size
                freed += size
                print(f"[NVR CLEANUP] Obrisana stara snimka ({size // (1024*1024)} MB): {os.path.basename(fpath)}")
                if total_bytes <= target_bytes:
                    break
            except Exception as e:
                print(f"[NVR CLEANUP] Greška pri brisanju: {e}")

def cleanup_recordings_by_age(recordings_dir: str = DEFAULT_RECORDINGS_DIR, retention_days: int = 30) -> dict:
    """
    GDPR retention cleanup: Deletes all NVR video recordings older than retention_days.
    Returns: {"deleted_videos": int, "freed_bytes": int}
    """
    if not os.path.isdir(recordings_dir) or retention_days <= 0:
        return {"deleted_videos": 0, "freed_bytes": 0}
        
    cutoff_time = time.time() - (retention_days * 86400)
    deleted_count = 0
    freed_bytes = 0
    
    for root, _, files in os.walk(recordings_dir):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mkv")):
                fpath = os.path.join(root, f)
                try:
                    fmtime = os.path.getmtime(fpath)
                    if fmtime < cutoff_time:
                        fsize = os.path.getsize(fpath)
                        os.remove(fpath)
                        deleted_count += 1
                        freed_bytes += fsize
                        print(f"[GDPR RETENTION] Obrisana stara NVR snimka: {os.path.basename(fpath)}")
                except Exception as e:
                    print(f"[GDPR RETENTION] Greška pri brisanju snimke {fpath}: {e}")
                    
    return {
        "deleted_videos": deleted_count,
        "freed_bytes": freed_bytes
    }


class NVRChannelRecorder:
    """
    Records a continuous stream into segmented MP4 video files with millisecond bookmarks.
    """
    def __init__(
        self,
        cam_id: int,
        cam_name: str,
        recordings_dir: str = DEFAULT_RECORDINGS_DIR,
        segment_duration_sec: int = 300, # 5 minutes default
        max_storage_gb: float = 20.0,
        fps: float = 20.0
    ):
        self.cam_id = cam_id
        self.cam_name = str(cam_name).strip() or f"Kamera {cam_id}"
        self.recordings_dir = recordings_dir
        self.segment_duration_sec = max(30, int(segment_duration_sec))
        self.max_storage_gb = max_storage_gb
        self.fps = fps if fps > 0 else 20.0

        self.writer = None
        self.current_video_path = ""
        self.segment_start_time = 0.0
        self.segment_frames = 0
        self.lock = threading.Lock()
        self.is_active = False

        self.fourcc = cv2.VideoWriter_fourcc(*'avc1')

    def start(self):
        with self.lock:
            self.is_active = True
            self.segment_start_time = 0.0
            self.segment_frames = 0
            self.current_video_path = ""

    def stop(self):
        with self.lock:
            self.is_active = False
            if self.writer is not None:
                try:
                    self.writer.release()
                except Exception:
                    pass
                self.writer = None
            self.current_video_path = ""

    def _start_new_segment(self, frame_w: int, frame_h: int):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            self.writer = None

        date_str = time.strftime("%Y-%m-%d")
        day_dir = os.path.join(self.recordings_dir, date_str)
        os.makedirs(day_dir, exist_ok=True)

        t_stamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c for c in self.cam_name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_')
        filename = f"rec_cam{self.cam_id}_{safe_name}_{t_stamp}.mp4"
        self.current_video_path = os.path.join(day_dir, filename)

        # Prefer standard H.264 (avc1) for 100% native HTML5 web browser playback
        writer = None
        for tag in ['avc1', 'H264', 'mp4v']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*tag)
                w = cv2.VideoWriter(self.current_video_path, fourcc, self.fps, (frame_w, frame_h))
                if w.isOpened():
                    writer = w
                    break
                else:
                    w.release()
            except Exception:
                pass
        self.writer = writer
        self.segment_start_time = time.time()
        self.segment_frames = 0
        print(f"[NVR RECORDER] Započet novi segment: {filename} ({frame_w}x{frame_h} @ {self.fps} FPS, H.264)")

        # Periodically trigger background cleanup
        threading.Thread(target=cleanup_old_recordings, args=(self.recordings_dir, self.max_storage_gb), daemon=True).start()

    def write_frame(self, frame: np.ndarray):
        """Writes a video frame and handles segment rotation."""
        if not self.is_active or frame is None or frame.size == 0:
            return

        with self.lock:
            h, w = frame.shape[:2]
            now = time.time()

            # Need new segment if writer is None or time exceeded
            if self.writer is None or (now - self.segment_start_time) >= self.segment_duration_sec:
                self._start_new_segment(w, h)

            if self.writer is not None and self.writer.isOpened():
                self.writer.write(frame)
                self.segment_frames += 1

    def get_current_bookmark(self) -> tuple:
        """
        Returns (video_path, offset_sec) for the active recording.
        Used by the face recognition engine to record the exact timestamp.
        """
        with self.lock:
            if not self.is_active or not self.current_video_path or self.segment_start_time <= 0:
                return "", 0.0
            offset = max(0.0, time.time() - self.segment_start_time)
            return self.current_video_path, round(offset, 2)


class NVRManager:
    """
    Central manager for multiple NVR recording channels.
    """
    def __init__(self, recordings_dir: str = DEFAULT_RECORDINGS_DIR, max_storage_gb: float = 20.0):
        self.recordings_dir = recordings_dir
        self.max_storage_gb = max_storage_gb
        self.channels = {} # { cam_id: NVRChannelRecorder }
        self.lock = threading.Lock()

    def configure_channel(self, cam_id: int, cam_name: str, segment_duration_sec: int = 300, fps: float = 20.0):
        with self.lock:
            if cam_id in self.channels:
                self.channels[cam_id].stop()
            rec = NVRChannelRecorder(
                cam_id=cam_id,
                cam_name=cam_name,
                recordings_dir=self.recordings_dir,
                segment_duration_sec=segment_duration_sec,
                max_storage_gb=self.max_storage_gb,
                fps=fps
            )
            rec.start()
            self.channels[cam_id] = rec
            return rec

    def write_frame(self, cam_id: int, frame: np.ndarray):
        rec = self.channels.get(cam_id)
        if rec:
            rec.write_frame(frame)

    def get_bookmark(self, cam_id: int) -> tuple:
        rec = self.channels.get(cam_id)
        if rec:
            return rec.get_current_bookmark()
        return "", 0.0

    def stop_all(self):
        with self.lock:
            for rec in self.channels.values():
                rec.stop()
            self.channels.clear()

    def list_recordings(self, date_filter: str = "") -> list:
        """Returns sorted metadata list of all recorded MP4 video segments."""
        results = []
        if not os.path.isdir(self.recordings_dir):
            return results

        for root, _, files in os.walk(self.recordings_dir):
            day_folder = os.path.basename(root)
            if date_filter and date_filter not in day_folder:
                continue

            for f in files:
                if f.lower().endswith(".mp4"):
                    fpath = os.path.join(root, f)
                    try:
                        stat = os.stat(fpath)
                        size_mb = round(stat.st_size / (1024 * 1024), 2)
                        mtime_str = time.strftime("%d.%m.%Y. %H:%M:%S", time.localtime(stat.st_mtime))
                        
                        # Parse camera name from filename
                        cam_tag = "Kamera"
                        if "rec_cam" in f:
                            parts = f.replace(".mp4", "").split("_")
                            if len(parts) >= 3:
                                cam_tag = f"Cam #{parts[1].replace('cam', '')}"
                                
                        results.append({
                            "filename": f,
                            "path": fpath,
                            "day": day_folder,
                            "camera": cam_tag,
                            "size_mb": size_mb,
                            "mtime": stat.st_mtime,
                            "time_str": mtime_str
                        })
                    except Exception:
                        pass

        # Sort newest first
        results.sort(key=lambda x: x["mtime"], reverse=True)
        return results

# Global default instance
_default_nvr_manager = None
def get_nvr_manager() -> NVRManager:
    global _default_nvr_manager
    if _default_nvr_manager is None:
        _default_nvr_manager = NVRManager()
    return _default_nvr_manager
