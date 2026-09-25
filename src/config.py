import os
import json
import shutil
import subprocess

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
DEFAULT_SNAPSHOT_DIR = os.path.join(DATA_DIR, "snapshots")
SETTINGS_FILE = os.path.join(DATA_DIR, "settings.json")

def load_settings() -> dict:
    """Loads application settings from data/settings.json or returns defaults."""
    defaults = {
        "snapshot_dir": DEFAULT_SNAPSHOT_DIR,
        "save_mode": "annotated", # 'annotated' (with HUD & boxes) or 'both' or 'clean'
        "threshold": 0.45
    }
    if not os.path.isfile(SETTINGS_FILE):
        return defaults
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                defaults.update(data)
    except Exception as e:
        print(f"[UPOZORENJE] Greška pri čitanju settings.json: {e}")
    return defaults

def save_settings(settings_dict: dict) -> bool:
    """Saves settings dict to data/settings.json."""
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[GREŠKA] Ne mogu spremiti settings.json: {e}")
        return False

def get_snapshot_dir() -> str:
    """
    Returns the configured directory for saving camera/video snapshots.
    Ensures directory exists and migrates legacy snapshots from data/uploads.
    """
    cfg = load_settings()
    snap_dir = cfg.get("snapshot_dir") or DEFAULT_SNAPSHOT_DIR
    snap_dir = os.path.abspath(snap_dir)
    try:
        os.makedirs(snap_dir, exist_ok=True)
    except Exception:
        snap_dir = DEFAULT_SNAPSHOT_DIR
        os.makedirs(snap_dir, exist_ok=True)
        
    # Migrate any legacy live_snap_*.jpg from data/uploads to snapshots dir
    uploads_dir = os.path.join(DATA_DIR, "uploads")
    if os.path.isdir(uploads_dir):
        for fname in os.listdir(uploads_dir):
            if fname.startswith("live_snap_") and fname.lower().endswith((".jpg", ".png")):
                src_path = os.path.join(uploads_dir, fname)
                dst_path = os.path.join(snap_dir, fname)
                if not os.path.exists(dst_path):
                    try:
                        shutil.move(src_path, dst_path)
                    except Exception:
                        pass
                        
    return snap_dir

def set_snapshot_dir(new_path: str) -> tuple[bool, str]:
    """Validates and updates the snapshot directory in settings."""
    if not new_path or not str(new_path).strip():
        return False, "Putanja mape ne može biti prazna."
    clean_path = os.path.abspath(str(new_path).strip().strip('"\''))
    try:
        os.makedirs(clean_path, exist_ok=True)
        # Test write permissions
        test_file = os.path.join(clean_path, ".test_write")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
    except Exception as e:
        return False, f"Nije moguće stvoriti ili pisati u mapu: {e}"

    cfg = load_settings()
    cfg["snapshot_dir"] = clean_path
    if save_settings(cfg):
        return True, f"Mapa za snimke uspješno postavljena na:\n`{clean_path}`"
    return False, "Greška pri spremanju postavki u datoteku."

def get_saved_snapshots() -> list[dict]:
    """
    Returns list of all saved snapshots in the snapshot directory, newest first.
    Each item: { 'path', 'filename', 'time_str', 'size_kb' }
    """
    snap_dir = get_snapshot_dir()
    results = []
    if not os.path.isdir(snap_dir):
        return results
        
    for fname in os.listdir(snap_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            fpath = os.path.join(snap_dir, fname)
            try:
                stat = os.stat(fpath)
                import time
                t_str = time.strftime("%d.%m.%Y. %H:%M:%S", time.localtime(stat.st_mtime))
                size_kb = round(stat.st_size / 1024, 1)
                results.append({
                    "path": fpath,
                    "filename": fname,
                    "time_str": t_str,
                    "mtime": stat.st_mtime,
                    "size_kb": size_kb
                })
            except Exception:
                continue
                
    results.sort(key=lambda x: x["mtime"], reverse=True)
    return results

def open_folder_in_explorer(target_folder: str = None) -> tuple[bool, str]:
    """Opens a folder directly in Windows File Explorer."""
    folder = target_folder or get_snapshot_dir()
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    try:
        if hasattr(os, "startfile"):
            os.startfile(folder)
            return True, f"Otvorena mapa: {folder}"
        else:
            subprocess.Popen(["explorer", folder])
            return True, f"Otvorena mapa: {folder}"
    except Exception as e:
        return False, f"Greška pri otvaranju Explorera: {e}"
