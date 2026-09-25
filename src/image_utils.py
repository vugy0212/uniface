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
