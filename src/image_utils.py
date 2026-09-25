import os
import cv2
import numpy as np
from PIL import Image

def imread_unicode(file_input):
    """
    Safely reads an image from file path (supporting Unicode / Croatian characters on Windows),
    or handles PIL Image / numpy array / Gradio FileData / dict.
    Returns BGR numpy array.
    """
    if file_input is None:
        return None, "Nema ulazne slike"

    # If it is already a numpy array (e.g. from gr.Image)
    if isinstance(file_input, np.ndarray):
        if len(file_input.shape) == 2:
            return cv2.cvtColor(file_input, cv2.COLOR_GRAY2BGR), None
        elif len(file_input.shape) == 3:
            if file_input.shape[2] == 4:
                return cv2.cvtColor(file_input, cv2.COLOR_RGBA2BGR), None
            elif file_input.shape[2] == 3:
                # Note: gr.Image passes RGB array
                return cv2.cvtColor(file_input, cv2.COLOR_RGB2BGR), None
        return file_input, None

    # If it is a PIL Image
    if isinstance(file_input, Image.Image):
        arr = np.array(file_input.convert("RGB"))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), None

    # If it is a Gradio FileData or dict
    path = None
    if hasattr(file_input, "path") and getattr(file_input, "path", None):
        path = file_input.path
    elif isinstance(file_input, dict) and "path" in file_input:
        path = file_input["path"]
    elif hasattr(file_input, "name") and getattr(file_input, "name", None):
        path = file_input.name
    elif isinstance(file_input, str):
        path = file_input

    if not path or not os.path.exists(path):
        return None, f"Putanja datoteke ne postoji: {path}"

    try:
        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL in case of webp or special formats
            pil_img = Image.open(path).convert("RGB")
            img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return img, None
    except Exception as e:
        return None, f"Greška pri učitavanju slike ({os.path.basename(path)}): {e}"

def imwrite_unicode(path: str, img_bgr: np.ndarray) -> bool:
    """
    Safely writes an image to disk supporting Unicode paths on Windows.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".jpg"
        success, buffer = cv2.imencode(ext, img_bgr)
        if success:
            with open(path, "wb") as f:
                f.write(buffer)
            return True
        return False
    except Exception:
        return False

def compute_image_hash(img_bgr: np.ndarray) -> str:
    """
    Computes a deterministic SHA-256 hash based on image dimensions and pixel data.
    Fast and robust against duplicate image files.
    """
    import hashlib
    h = hashlib.sha256()
    h.update(str(img_bgr.shape).encode("utf-8"))
    h.update(img_bgr.tobytes())
    return h.hexdigest()

def save_image_dedup(img_bgr: np.ndarray, target_dir: str, prefix: str = "orig") -> str:
    """
    Saves an image into target_dir using content-based addressing (SHA-256).
    If an identical image is already stored, returns the existing file path to prevent disk waste.
    """
    os.makedirs(target_dir, exist_ok=True)
    img_hash = compute_image_hash(img_bgr)[:16]
    filename = f"{prefix}_{img_hash}.jpg"
    target_path = os.path.join(target_dir, filename)
    if not os.path.exists(target_path):
        imwrite_unicode(target_path, img_bgr)
    return target_path
