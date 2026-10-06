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
        "threshold": 0.45,
        "gdpr_retention_days": 30,
        "anti_spoofing": True,
        "telegram_enabled": False,
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "telegram_chat_id_security": "",
        "telegram_chat_id_vip": "",
        "telegram_notify_blacklist": True,
        "telegram_notify_vip": True,
        "telegram_notify_spoof": True,
        "telegram_notify_unknown": False,
        "telegram_cooldown_min": 5,
        "email_enabled": False,
        "email_smtp_host": "smtp.gmail.com",
        "email_smtp_port": 587,
        "email_use_tls": True,
        "email_use_ssl": False,
        "email_sender": "",
        "email_password": "",
        "email_recipients": "",
        "email_notify_blacklist": True,
        "email_notify_spoof": True,
        "email_notify_vip": False,
        "email_cooldown_min": 10
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

def get_retention_days() -> int:
    cfg = load_settings()
    return int(cfg.get("gdpr_retention_days", 30))

def set_retention_days(days: int) -> bool:
    cfg = load_settings()
    cfg["gdpr_retention_days"] = int(days)
    return save_settings(cfg)

def execute_gdpr_retention(retention_days: int = None) -> dict:
    """
    Executes unified GDPR retention cleanup:
    1. Purges database detection_events and corresponding JPEG crops/snapshots older than retention_days.
    2. Purges NVR recordings older than retention_days.
    3. Cleans up any unreferenced/orphan snapshot files older than retention_days.
    Returns: {
        "retention_days": int,
        "deleted_events": int,
        "deleted_snapshots": int,
        "deleted_videos": int,
        "total_freed_bytes": int,
        "freed_mb": float,
        "status_message": str
    }
    """
    import time
    if retention_days is None:
        retention_days = get_retention_days()
        
    if retention_days <= 0:
        return {
            "retention_days": 0,
            "deleted_events": 0,
            "deleted_snapshots": 0,
            "deleted_videos": 0,
            "total_freed_bytes": 0,
            "freed_mb": 0.0,
            "status_message": "ℹ️ Politika zadržavanja je postavljena na trajno čuvanje (automatsko brisanje isključeno)."
        }
        
    try:
        from . import db
        from . import nvr_recorder
    except ImportError:
        import db
        import nvr_recorder
    
    # 1. Purge DB events and their files
    db_res = db.purge_old_detection_events(retention_days)
    
    # 2. Purge NVR videos
    recordings_dir = getattr(nvr_recorder, "DEFAULT_RECORDINGS_DIR", os.path.join(DATA_DIR, "recordings"))
    nvr_res = nvr_recorder.cleanup_recordings_by_age(recordings_dir, retention_days)
    
    # 3. Clean any orphaned snapshot files in snapshots_dir older than retention_days
    snap_dir = get_snapshot_dir()
    cutoff_time = time.time() - (retention_days * 86400)
    orphan_snaps = 0
    orphan_bytes = 0
    if os.path.isdir(snap_dir):
        for fname in os.listdir(snap_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(snap_dir, fname)
                try:
                    if os.path.getmtime(fpath) < cutoff_time:
                        sz = os.path.getsize(fpath)
                        os.remove(fpath)
                        orphan_snaps += 1
                        orphan_bytes += sz
                except Exception:
                    pass
                    
    total_freed = db_res.get("freed_bytes", 0) + nvr_res.get("freed_bytes", 0) + orphan_bytes
    total_snapshots = db_res.get("deleted_files", 0) + orphan_snaps
    freed_mb = round(total_freed / (1024 * 1024), 2)
    
    return {
        "retention_days": retention_days,
        "deleted_events": db_res.get("deleted_events", 0),
        "deleted_snapshots": total_snapshots,
        "deleted_videos": nvr_res.get("deleted_videos", 0),
        "total_freed_bytes": total_freed,
        "freed_mb": freed_mb,
        "status_message": f"✅ **GDPR rotacija uspješno izvršena** (starije od {retention_days} dana):\n* Obrisano događaja u evidenciji: **{db_res.get('deleted_events', 0)}**\n* Obrisano slika kadrova: **{total_snapshots}**\n* Obrisano NVR video snimaka: **{nvr_res.get('deleted_videos', 0)}**\n* Oslobođeno prostora na disku: **{freed_mb} MB**"
    }

def get_anti_spoofing() -> bool:
    """Returns True if anti-spoofing (liveness detection) is enabled (default: True)."""
    return bool(load_settings().get("anti_spoofing", True))

def set_anti_spoofing(enabled: bool) -> bool:
    """Saves anti-spoofing toggle to settings."""
    cfg = load_settings()
    cfg["anti_spoofing"] = bool(enabled)
    return save_settings(cfg)

def get_telegram_config() -> dict:
    """Returns Telegram notification configuration."""
    cfg = load_settings()
    return {
        "enabled": bool(cfg.get("telegram_enabled", False)),
        "bot_token": str(cfg.get("telegram_bot_token", "")).strip(),
        "chat_id": str(cfg.get("telegram_chat_id", "")).strip(),
        "chat_id_security": str(cfg.get("telegram_chat_id_security", "")).strip(),
        "chat_id_vip": str(cfg.get("telegram_chat_id_vip", "")).strip(),
        "notify_blacklist": bool(cfg.get("telegram_notify_blacklist", True)),
        "notify_vip": bool(cfg.get("telegram_notify_vip", True)),
        "notify_spoof": bool(cfg.get("telegram_notify_spoof", True)),
        "notify_unknown": bool(cfg.get("telegram_notify_unknown", False)),
        "cooldown_min": int(cfg.get("telegram_cooldown_min", 5))
    }

def save_telegram_config(
    enabled: bool,
    bot_token: str,
    chat_id: str,
    notify_blacklist: bool = True,
    notify_vip: bool = True,
    notify_spoof: bool = True,
    notify_unknown: bool = False,
    cooldown_min: int = 5,
    chat_id_security: str = "",
    chat_id_vip: str = ""
) -> bool:
    """Saves Telegram notification configuration."""
    cfg = load_settings()
    cfg["telegram_enabled"] = bool(enabled)
    cfg["telegram_bot_token"] = str(bot_token).strip()
    cfg["telegram_chat_id"] = str(chat_id).strip()
    cfg["telegram_chat_id_security"] = str(chat_id_security).strip()
    cfg["telegram_chat_id_vip"] = str(chat_id_vip).strip()
    cfg["telegram_notify_blacklist"] = bool(notify_blacklist)
    cfg["telegram_notify_vip"] = bool(notify_vip)
    cfg["telegram_notify_spoof"] = bool(notify_spoof)
    cfg["telegram_notify_unknown"] = bool(notify_unknown)
    cfg["telegram_cooldown_min"] = max(1, int(cooldown_min))
    return save_settings(cfg)

def get_email_config() -> dict:
    """Returns E-mail notification configuration."""
    cfg = load_settings()
    return {
        "enabled": bool(cfg.get("email_enabled", False)),
        "smtp_host": str(cfg.get("email_smtp_host", "smtp.gmail.com")).strip(),
        "smtp_port": int(cfg.get("email_smtp_port", 587)),
        "use_tls": bool(cfg.get("email_use_tls", True)),
        "use_ssl": bool(cfg.get("email_use_ssl", False)),
        "sender": str(cfg.get("email_sender", "")).strip(),
        "password": str(cfg.get("email_password", "")).strip(),
        "recipients": str(cfg.get("email_recipients", "")).strip(),
        "notify_blacklist": bool(cfg.get("email_notify_blacklist", True)),
        "notify_spoof": bool(cfg.get("email_notify_spoof", True)),
        "notify_vip": bool(cfg.get("email_notify_vip", False)),
        "cooldown_min": int(cfg.get("email_cooldown_min", 10))
    }

def save_email_config(
    enabled: bool,
    smtp_host: str,
    smtp_port: int,
    use_tls: bool,
    use_ssl: bool,
    sender: str,
    password: str,
    recipients: str,
    notify_blacklist: bool = True,
    notify_spoof: bool = True,
    notify_vip: bool = False,
    cooldown_min: int = 10
) -> bool:
    """Saves E-mail notification configuration."""
    cfg = load_settings()
    cfg["email_enabled"] = bool(enabled)
    cfg["email_smtp_host"] = str(smtp_host).strip()
    cfg["email_smtp_port"] = int(smtp_port)
    cfg["email_use_tls"] = bool(use_tls)
    cfg["email_use_ssl"] = bool(use_ssl)
    cfg["email_sender"] = str(sender).strip()
    cfg["email_password"] = str(password).strip()
    cfg["email_recipients"] = str(recipients).strip()
    cfg["email_notify_blacklist"] = bool(notify_blacklist)
    cfg["email_notify_spoof"] = bool(notify_spoof)
    cfg["email_notify_vip"] = bool(notify_vip)
    cfg["email_cooldown_min"] = max(1, int(cooldown_min))
    return save_settings(cfg)


