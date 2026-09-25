import os
import platform
import subprocess
import ctypes

def get_system_ram_gb() -> float | None:
    """Detects total physical system RAM in GB using Windows API."""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return round(stat.ullTotalPhys / (1024**3), 1)
    except Exception:
        pass
    return None

def get_gpus() -> list[str]:
    """Detects physical GPUs, filtering out virtual/remote displays."""
    try:
        cmd = "(Get-CimInstance Win32_VideoController).Name"
        out = subprocess.check_output(["powershell", "-Command", cmd], timeout=4).decode().strip()
        lines = [line.strip() for line in out.splitlines() if line.strip()]
        real_gpus = [g for g in lines if "virtual" not in g.lower()]
        return real_gpus if real_gpus else lines
    except Exception:
        return ["Generički video kontroler"]

def get_onnx_acceleration_status() -> tuple[str, str]:
    """Checks ONNX Runtime available execution providers."""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "NVIDIA CUDA / TensorRT", "⚡ Hardversko ubrzanje (GPU) aktivno"
        elif "ROCMExecutionProvider" in providers:
            return "AMD ROCm", "⚡ Hardversko ubrzanje (GPU) aktivno"
        else:
            return "CPU (OpenMP / AVX2)", "⚙️ CPU izvođenje"
    except Exception as e:
        return "CPU", f"Automatski ({e})"

def get_disk_usage_info(data_dir: str) -> dict:
    """Calculates disk sizes for database, crops, and uploads."""
    info = {"db_size_kb": 0, "crops_size_mb": 0.0, "uploads_size_mb": 0.0, "total_mb": 0.0}
    try:
        db_path = os.path.join(data_dir, "database.db")
        if os.path.exists(db_path):
            info["db_size_kb"] = round(os.path.getsize(db_path) / 1024, 1)

        crops_dir = os.path.join(data_dir, "crops")
        if os.path.exists(crops_dir):
            c_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(crops_dir) for f in fs)
            info["crops_size_mb"] = round(c_bytes / (1024 * 1024), 2)

        uploads_dir = os.path.join(data_dir, "uploads")
        if os.path.exists(uploads_dir):
            u_bytes = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(uploads_dir) for f in fs)
            info["uploads_size_mb"] = round(u_bytes / (1024 * 1024), 2)

        info["total_mb"] = round((info["db_size_kb"] / 1024) + info["crops_size_mb"] + info["uploads_size_mb"], 2)
    except Exception:
        pass
    return info

def get_system_report_markdown(data_dir: str) -> str:
    """Generates dynamically detected hardware and system report in Markdown."""
    ram = get_system_ram_gb()
    ram_str = f"{ram} GB" if ram else "Nedostupno"
    gpus = get_gpus()
    gpu_str = ", ".join(gpus) if gpus else "Nedostupno"
    cpu_cores = os.cpu_count() or "-"
    os_name = f"{platform.system()} {platform.release()} ({platform.machine()})"
    accel_type, accel_status = get_onnx_acceleration_status()
    disk = get_disk_usage_info(data_dir)

    md = f"""### 💻 Stvarna hardverska konfiguracija računala (Automatski detektirano)
* **Grafička kartica (GPU):** {gpu_str}
* **Radna memorija (RAM):** {ram_str}
* **Procesor (CPU):** {cpu_cores} logičkih jezgri ({platform.processor() or 'x86_64'})
* **Operativni sustav:** {os_name}
* **ONNX Ubrzanje:** {accel_type} — *{accel_status}*

---

### 💾 Zauzeće diska i lokalna pohrana
* **SQLite baza podataka:** `{disk['db_size_kb']} KB` (`data/database.db`)
* **Izrezana lica (Crops):** `{disk['crops_size_mb']} MB` (`data/crops/`)
* **Izvornici (Uploads):** `{disk['uploads_size_mb']} MB` (`data/uploads/`)
* **Ukupna pohrana:** `{disk['total_mb']} MB`
"""
    return md
