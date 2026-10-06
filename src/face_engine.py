import os
import cv2
import numpy as np

try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# Use local models directory if present (for self-contained / portable installs)
_local_models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
if os.path.isdir(_local_models_dir):
    os.environ["UNIFACE_CACHE_DIR"] = _local_models_dir

from uniface import FaceAnalyzer, RetinaFace, EdgeFace
from uniface.recognition.edgeface import EdgeFaceWeights
from uniface.spoofing import MiniFASNet

_analyzer_instances = {}
_spoofer_instances = {}


def get_execution_providers(device: str = "AUTO") -> list[str]:
    """
    Resolves prioritized ONNX Runtime execution providers based on requested target.
    Supports:
      - 'AUTO': Probes CUDA -> DirectML (DML) -> OpenVINO -> CoreML -> ROCm -> CPU
      - 'DIRECTML' / 'DML': Forces DirectX 12 hardware acceleration (Intel/AMD/NVIDIA/NPU)
      - 'CUDA': Forces NVIDIA CUDA
      - 'OPENVINO': Forces Intel OpenVINO
      - 'CPU': Forces universal CPU execution
    """
    dev = (device or "AUTO").upper().strip()
    if dev == "CPU":
        return ["CPUExecutionProvider"]

    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except Exception:
        return ["CPUExecutionProvider"]

    if dev in ("DIRECTML", "DML"):
        if "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if dev == "CUDA":
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    if dev == "OPENVINO":
        if "OpenVINOExecutionProvider" in available:
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    # AUTO: prioritize highest-performance available providers
    providers = []
    for candidate in [
        "CUDAExecutionProvider",
        "DmlExecutionProvider",
        "OpenVINOExecutionProvider",
        "CoreMLExecutionProvider",
        "ROCMExecutionProvider"
    ]:
        if candidate in available:
            providers.append(candidate)

    providers.append("CPUExecutionProvider")
    return providers

def get_analyzer(device="AUTO", with_attributes=False):
    """
    Returns FaceAnalyzer using commercial-ready EdgeFace BASE recognizer (BSD-3-Clause)
    and RetinaFace detector (MIT), with automatic hardware acceleration negotiation.
    """
    global _analyzer_instances
    key = ((device or "AUTO").upper().strip(), bool(with_attributes))
    if key not in _analyzer_instances:
        providers = get_execution_providers(device)
        try:
            detector = RetinaFace(confidence_threshold=0.45, providers=providers)
            recognizer = EdgeFace(model_name=EdgeFaceWeights.BASE, providers=providers)
            _analyzer_instances[key] = FaceAnalyzer(detector=detector, recognizer=recognizer, predictors=[])
        except Exception:
            detector = RetinaFace(confidence_threshold=0.45, providers=["CPUExecutionProvider"])
            recognizer = EdgeFace(model_name=EdgeFaceWeights.BASE, providers=["CPUExecutionProvider"])
            _analyzer_instances[key] = FaceAnalyzer(detector=detector, recognizer=recognizer, predictors=[])
    return _analyzer_instances[key]

def get_spoofer(device="AUTO"):
    """
    Returns cached MiniFASNet V2 anti-spoofing engine (BSD/MIT ready, 1.2 MB)
    with prioritized hardware acceleration.
    """
    global _spoofer_instances
    key = (device or "AUTO").upper().strip()
    if key not in _spoofer_instances:
        providers = get_execution_providers(device)
        try:
            _spoofer_instances[key] = MiniFASNet(providers=providers)
        except Exception:
            _spoofer_instances[key] = MiniFASNet(providers=["CPUExecutionProvider"])
    return _spoofer_instances[key]

def check_liveness(image_bgr: np.ndarray, bbox, device="AUTO") -> tuple[bool, float]:
    """
    Analyzes presentation attack / spoofing on a detected face (photo, phone/tablet screen, paper).
    Returns:
        (is_real: bool, confidence: float)
    """
    if image_bgr is None or bbox is None or len(bbox) < 4:
        return True, 0.5
    try:
        spoofer = get_spoofer(device)
        res = spoofer.predict(image_bgr, bbox)
        return bool(res.is_real), float(res.confidence)
    except Exception as e:
        return True, 0.5

def crop_face(image: np.ndarray, bbox, margin_ratio=0.25):
    if image is None or image.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    h, w = image.shape[:2]
    try:
        x1, y1, x2, y2 = map(int, bbox)
    except Exception:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
        
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    
    x1 = max(0, int(x1 - bw * margin_ratio))
    y1 = max(0, int(y1 - bh * margin_ratio))
    x2 = min(w, int(x2 + bw * margin_ratio))
    y2 = min(h, int(y2 + bh * margin_ratio))
    
    if x2 <= x1 or y2 <= y1:
        x1, y1, x2, y2 = 0, 0, min(w, 100), min(h, 100)
        
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    return crop

def extract_faces_from_image(image_bgr: np.ndarray, device="AUTO", with_attributes=False, check_spoofing=False):
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return []
    analyzer = get_analyzer(device, with_attributes=with_attributes)
    faces = analyzer.analyze(image_bgr)
    
    extracted = []
    for idx, face in enumerate(faces):
        crop = crop_face(image_bgr, face.bbox)
        emb = face.embedding
        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
        
        is_real = True
        liveness_conf = 1.0
        if check_spoofing:
            is_real, liveness_conf = check_liveness(image_bgr, face.bbox, device=device)
                
        extracted.append({
            "index": idx,
            "bbox": face.bbox.astype(int).tolist(),
            "landmarks": face.landmarks.tolist() if face.landmarks is not None else None,
            "confidence": float(face.confidence) if face.confidence is not None else 1.0,
            "embedding": emb,
            "crop_bgr": crop,
            "is_real": is_real,
            "liveness_conf": liveness_conf,
            "age": getattr(face, "age", None),
            "gender": getattr(face, "gender", None),
            "race": getattr(face, "race", None)
        })
    return extracted

def build_person_profiles(all_samples: list[dict]) -> dict:
    """
    Groups individual samples by person, calculates the normalized centroid embedding
    representing the multi-angle synthesized facial profile, and assigns quality ratings.
    """
    profiles = {}
    for s in all_samples:
        pid = s["person_id"]
        if pid not in profiles:
            profiles[pid] = {
                "person_id": pid,
                "person_name": s["person_name"],
                "samples": [],
                "crop_path": s["crop_path"],
                "sample_ids": []
            }
        emb = s["embedding"]
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        profiles[pid]["samples"].append(emb)
        profiles[pid]["sample_ids"].append(s["sample_id"])
        
    for pid, p in profiles.items():
        # Compute normalized centroid
        mean_vec = np.mean(p["samples"], axis=0)
        c_norm = np.linalg.norm(mean_vec)
        p["centroid"] = mean_vec / c_norm if c_norm > 0 else mean_vec
        p["count"] = len(p["samples"])
        
        # Profile quality & coverage ratings
        if p["count"] == 1:
            p["quality_badge"] = "Osnovno (1 slika)"
            p["quality_icon"] = "🔴"
            p["quality_pct"] = 35
            p["recommendation"] = "Preporuka: Dodajte još 1-2 slike pod blagim kutom ili s osmijehom."
        elif p["count"] == 2:
            p["quality_badge"] = "Dobro (2 slike)"
            p["quality_icon"] = "🟡"
            p["quality_pct"] = 70
            p["recommendation"] = "Dobra točnost. Za 100% pokrivenost možete dodati još 1 profilnu sliku."
        else:
            p["quality_badge"] = f"Izvrsno ({p['count']} slika)"
            p["quality_icon"] = "🟢"
            p["quality_pct"] = 100
            p["recommendation"] = "Optimalna točnost. Pokriveni su višestruki kutevi i crte lica."
            
    return profiles

class FaceIndex:
    """
    High-performance scalable vectorized face search index.
    Supports dual-tier acceleration:
      1. FAISS Accelerated Index (IndexFlatIP for exact sub-millisecond search;
         IndexHNSWFlat for logarithmic O(log N) graph search on large datasets >= 1,000 vectors)
      2. Optimized NumPy BLAS Flat Matrix Multiplication fallback.
    """
    def __init__(self, all_samples: list[dict]):
        self.profiles = build_person_profiles(all_samples)
        self.person_ids = list(self.profiles.keys())
        self.total_samples = len(all_samples)
        self.num_persons = len(self.person_ids)
        self.backend = "NumPy BLAS"

        if self.total_samples == 0 or self.num_persons == 0:
            self.samples_matrix = np.empty((0, 512), dtype=np.float32)
            self.centroids_matrix = np.empty((0, 512), dtype=np.float32)
            self.sample_to_person = []
            self.person_sample_slices = {}
            self.faiss_samples_index = None
            self.faiss_centroids_index = None
            return

        samples_list = []
        self.sample_to_person = []
        self.person_sample_slices = {}
        curr = 0
        for pid in self.person_ids:
            p_samples = self.profiles[pid]["samples"]
            samples_list.extend(p_samples)
            n_p = len(p_samples)
            self.person_sample_slices[pid] = (curr, curr + n_p)
            self.sample_to_person.extend([pid] * n_p)
            curr += n_p

        self.samples_matrix = np.ascontiguousarray(np.stack(samples_list), dtype=np.float32)
        s_norms = np.linalg.norm(self.samples_matrix, axis=1, keepdims=True)
        s_norms[s_norms == 0] = 1.0
        self.samples_matrix = self.samples_matrix / s_norms

        centroids_list = [self.profiles[pid]["centroid"] for pid in self.person_ids]
        self.centroids_matrix = np.ascontiguousarray(np.stack(centroids_list), dtype=np.float32)
        c_norms = np.linalg.norm(self.centroids_matrix, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1.0
        self.centroids_matrix = self.centroids_matrix / c_norms

        # Initialize FAISS Index if available
        self.faiss_samples_index = None
        self.faiss_centroids_index = None
        if HAS_FAISS:
            try:
                dim = 512
                # If >= 1,000 vectors, use HNSW graph index; otherwise exact FlatIP
                if self.total_samples >= 1000:
                    self.faiss_samples_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                    self.backend = f"FAISS HNSW v{faiss.__version__} (Graph Index)"
                else:
                    self.faiss_samples_index = faiss.IndexFlatIP(dim)
                    self.backend = f"FAISS FlatIP v{faiss.__version__} (AVX2 SIMD)"

                self.faiss_samples_index.add(self.samples_matrix)

                if self.num_persons >= 1000:
                    self.faiss_centroids_index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
                else:
                    self.faiss_centroids_index = faiss.IndexFlatIP(dim)

                self.faiss_centroids_index.add(self.centroids_matrix)
            except Exception:
                self.faiss_samples_index = None
                self.faiss_centroids_index = None
                self.backend = "NumPy BLAS (Fallback)"

    def match(self, query_embedding: np.ndarray, threshold: float = 0.50) -> dict:
        if self.total_samples == 0 or query_embedding is None:
            return {
                "matched": False,
                "best_name": "Baza je prazna",
                "person_name": "Baza je prazna",
                "similarity": 0.0,
                "person_id": None,
                "status": "Nema baze",
                "quality_icon": "⚪",
                "quality_badge": "Nema uzoraka",
                "sample_count": 0,
                "margin": 0.0,
                "crop_path": None,
                "index_backend": self.backend
            }

        q = np.ascontiguousarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Scalable candidate selection for medium/large databases (> 100 profiles)
        if self.faiss_samples_index is not None and self.num_persons > 100:
            q_batch = q[np.newaxis, :]
            k_s = min(60, self.total_samples)
            D_s, I_s = self.faiss_samples_index.search(q_batch, k_s)

            k_c = min(40, self.num_persons)
            D_c, I_c = self.faiss_centroids_index.search(q_batch, k_c)

            candidate_pids = set()
            for idx in I_s[0]:
                if 0 <= idx < len(self.sample_to_person):
                    candidate_pids.add(self.sample_to_person[idx])
            for idx in I_c[0]:
                if 0 <= idx < len(self.person_ids):
                    candidate_pids.add(self.person_ids[idx])

            candidate_scores = []
            for pid in candidate_pids:
                p = self.profiles[pid]
                start_i, end_i = self.person_sample_slices[pid]
                s_sims = self.samples_matrix[start_i:end_i] @ q
                max_sim = float(np.max(s_sims)) if len(s_sims) > 0 else 0.0
                centroid_sim = float(p["centroid"] @ q)

                if p["count"] > 1:
                    effective_sim = max(max_sim * 0.96, centroid_sim, 0.45 * max_sim + 0.55 * centroid_sim)
                else:
                    effective_sim = max_sim

                candidate_scores.append({
                    "person_id": pid,
                    "person_name": p["person_name"],
                    "similarity": max(0.0, effective_sim),
                    "max_sample_sim": max(0.0, max_sim),
                    "centroid_sim": max(0.0, centroid_sim),
                    "count": p["count"],
                    "quality_badge": p["quality_badge"],
                    "quality_icon": p["quality_icon"],
                    "crop_path": p["crop_path"]
                })
        else:
            # Full evaluation across all registered persons (NumPy BLAS / small DB)
            all_sample_sims = self.samples_matrix @ q
            all_centroid_sims = self.centroids_matrix @ q

            candidate_scores = []
            for i, pid in enumerate(self.person_ids):
                p = self.profiles[pid]
                start_i, end_i = self.person_sample_slices[pid]
                s_sims = all_sample_sims[start_i:end_i]
                max_sim = float(np.max(s_sims)) if len(s_sims) > 0 else 0.0
                centroid_sim = float(all_centroid_sims[i])

                if p["count"] > 1:
                    effective_sim = max(max_sim * 0.96, centroid_sim, 0.45 * max_sim + 0.55 * centroid_sim)
                else:
                    effective_sim = max_sim

                candidate_scores.append({
                    "person_id": pid,
                    "person_name": p["person_name"],
                    "similarity": max(0.0, effective_sim),
                    "max_sample_sim": max(0.0, max_sim),
                    "centroid_sim": max(0.0, centroid_sim),
                    "count": p["count"],
                    "quality_badge": p["quality_badge"],
                    "quality_icon": p["quality_icon"],
                    "crop_path": p["crop_path"]
                })

        if not candidate_scores:
            return {
                "matched": False,
                "best_name": "Nepoznat",
                "person_name": "Nepoznat",
                "person_id": None,
                "similarity": 0.0,
                "status": "Nepoznat",
                "quality_icon": "⚪",
                "quality_badge": "Nema poklapanja",
                "sample_count": 0,
                "margin": 0.0,
                "crop_path": None,
                "index_backend": self.backend
            }

        candidate_scores.sort(key=lambda x: x["similarity"], reverse=True)
        best = candidate_scores[0]
        second_sim = candidate_scores[1]["similarity"] if len(candidate_scores) > 1 else 0.0
        margin = best["similarity"] - second_sim

        matched = (best["similarity"] >= threshold)
        possible_threshold = max(0.30, round(threshold - 0.08, 2))
        if matched:
            status = "Prepoznat"
        elif best["similarity"] >= possible_threshold:
            status = "Moguće poklapanje"
        else:
            status = "Nepoznat"

        return {
            "matched": matched,
            "best_name": best["person_name"],
            "person_name": best["person_name"],
            "person_id": best["person_id"],
            "similarity": best["similarity"],
            "status": status,
            "quality_icon": best["quality_icon"],
            "quality_badge": best["quality_badge"],
            "sample_count": best["count"],
            "margin": margin,
            "crop_path": best["crop_path"],
            "index_backend": self.backend
        }

    def match_sample_hybrid(self, query_embedding: np.ndarray, threshold: float = 0.50) -> dict:
        """Alias for match method for backward and live camera compatibility."""
        return self.match(query_embedding, threshold=threshold)

    def get_info(self) -> dict:
        """Returns metadata about the active vector index backend."""
        return {
            "backend": self.backend,
            "has_faiss": HAS_FAISS,
            "total_samples": self.total_samples,
            "num_persons": self.num_persons,
            "dimension": 512
        }

_face_index = None

def invalidate_face_index():
    global _face_index
    _face_index = None

def get_face_index(all_samples=None, force_refresh=False) -> FaceIndex:
    global _face_index
    if _face_index is None or force_refresh:
        if all_samples is None:
            try:
                import db
            except ImportError:
                from src import db
            all_samples = db.get_cached_embeddings()
        _face_index = FaceIndex(all_samples)
    return _face_index

# Auto-register callback with db to invalidate cache when DB changes
try:
    try:
        import db
    except ImportError:
        from src import db
    db.register_cache_invalidation_callback(invalidate_face_index)
except Exception:
    pass

def match_face(query_embedding: np.ndarray, profiles=None, threshold=0.50):
    if isinstance(profiles, FaceIndex):
        return profiles.match(query_embedding, threshold)
    if profiles is None:
        return get_face_index().match(query_embedding, threshold)
    if isinstance(profiles, dict):
        idx = FaceIndex([])
        idx.profiles = profiles
        idx.person_ids = list(profiles.keys())
        idx.total_samples = sum(len(p.get("samples", [])) for p in profiles.values())
        idx.num_persons = len(idx.person_ids)
        if idx.total_samples > 0:
            samples_list = []
            idx.person_sample_slices = {}
            curr = 0
            for pid in idx.person_ids:
                p_samples = profiles[pid]["samples"]
                samples_list.extend(p_samples)
                n_p = len(p_samples)
                idx.person_sample_slices[pid] = (curr, curr + n_p)
                curr += n_p
            idx.samples_matrix = np.ascontiguousarray(np.stack(samples_list), dtype=np.float32)
            s_norms = np.linalg.norm(idx.samples_matrix, axis=1, keepdims=True)
            s_norms[s_norms == 0] = 1.0
            idx.samples_matrix = idx.samples_matrix / s_norms
            centroids_list = [profiles[pid]["centroid"] for pid in idx.person_ids]
            idx.centroids_matrix = np.ascontiguousarray(np.stack(centroids_list), dtype=np.float32)
            c_norms = np.linalg.norm(idx.centroids_matrix, axis=1, keepdims=True)
            c_norms[c_norms == 0] = 1.0
            idx.centroids_matrix = idx.centroids_matrix / c_norms
            return idx.match(query_embedding, threshold)
    return get_face_index().match(query_embedding, threshold)

def process_and_annotate(image_bgr: np.ndarray, all_samples: list[dict] = None, threshold=0.50,
                         draw_landmarks=False, blur_unknown=False, blur_all=False, device="AUTO", face_index=None,
                         cached_faces=None):
    if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.size == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8), []
    annotated = image_bgr.copy()
    if cached_faces is not None and len(cached_faces) > 0:
        faces_data = cached_faces
    else:
        faces_data = extract_faces_from_image(image_bgr, device)
    
    if face_index is None:
        if all_samples is not None:
            face_index = FaceIndex(all_samples)
        else:
            face_index = get_face_index()
    
    # Sort faces from left to right for clean visual indexing [#1], [#2]
    faces_data.sort(key=lambda f: f["bbox"][0])
    for i, f in enumerate(faces_data):
        f["display_index"] = i + 1
        
    results = []
    img_h, img_w = annotated.shape[:2]
    
    for face in faces_data:
        x1, y1, x2, y2 = face["bbox"]
        f_num = face["display_index"]
        
        match_info = face_index.match(face["embedding"], threshold)
        is_known = match_info["matched"]
        status = match_info["status"]
        best_name = match_info["best_name"]
        sim_pct = match_info["similarity"] * 100
        margin_pct = match_info["margin"] * 100
        q_icon = match_info["quality_icon"]
        
        if is_known:
            box_color = (30, 200, 75)      # Green
            label = f"[#{f_num}] {best_name} ({sim_pct:.1f}%)"
            badge = "✅ Prepoznat"
        elif status == "Moguće poklapanje":
            box_color = (30, 160, 255)     # Orange
            label = f"[#{f_num}] {best_name}? ({sim_pct:.1f}%)"
            badge = f"⚠️ Moguće ({best_name})"
        else:
            box_color = (40, 50, 230)      # Red
            label = f"[#{f_num}] Nepoznato ({sim_pct:.1f}%)"
            badge = "❌ Nepoznat"
            
        # 1. Apply high-grade privacy blur if requested
        should_blur = bool(blur_all) or (bool(blur_unknown) and not is_known)
        if should_blur:
            fx1, fy1 = max(0, x1), max(0, y1)
            fx2, fy2 = min(img_w, x2), min(img_h, y2)
            sub = annotated[fy1:fy2, fx1:fx2]
            if sub.size > 0:
                bw, bh = fx2 - fx1, fy2 - fy1
                factor = max(8, min(bw, bh) // 8)
                small = cv2.resize(sub, (max(1, bw // factor), max(1, bh // factor)), interpolation=cv2.INTER_LINEAR)
                mosaic = cv2.resize(small, (bw, bh), interpolation=cv2.INTER_NEAREST)
                ksize = max(15, (bw // 6) * 2 + 1)
                blurred = cv2.GaussianBlur(mosaic, (ksize, ksize), 0)
                annotated[fy1:fy2, fx1:fx2] = blurred

        # 2. Draw clean border box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 1)
        
        # 3. Position label: If near top of image (y1 < 28), draw INSIDE box to prevent cut-off and overlapping
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        if y1 > th + 12:
            lbl_y1 = y1 - th - 8
            lbl_y2 = y1
            text_baseline = y1 - 4
        else:
            lbl_y1 = y1
            lbl_y2 = y1 + th + 8
            text_baseline = y1 + th + 4
            
        lbl_x2 = min(img_w, x1 + tw + 8)
        cv2.rectangle(annotated, (x1, lbl_y1), (lbl_x2, lbl_y2), box_color, -1)
        cv2.putText(annotated, label, (x1 + 4, text_baseline), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
        # 4. Draw landmarks if requested (visible biometric geometry wireframe + scaled dots)
        if draw_landmarks and face.get("landmarks") is not None and len(face["landmarks"]) > 0:
            pts = face["landmarks"]
            face_w = max(1, x2 - x1)
            pt_r = max(4, int(round(face_w * 0.016)))
            line_w = max(1, int(round(face_w * 0.006)))
            
            pts_int = [(int(p[0]), int(p[1])) for p in pts]
            if len(pts_int) >= 5:
                le, re, nose, lm, rm = pts_int[:5]
                mesh_col = (255, 200, 0) # High-tech biometric cyan in BGR
                cv2.line(annotated, le, re, mesh_col, line_w, cv2.LINE_AA)
                cv2.line(annotated, le, nose, mesh_col, line_w, cv2.LINE_AA)
                cv2.line(annotated, re, nose, mesh_col, line_w, cv2.LINE_AA)
                cv2.line(annotated, nose, lm, mesh_col, line_w, cv2.LINE_AA)
                cv2.line(annotated, nose, rm, mesh_col, line_w, cv2.LINE_AA)
                cv2.line(annotated, lm, rm, mesh_col, line_w, cv2.LINE_AA)
                
            for px, py in pts_int:
                cv2.circle(annotated, (px, py), pt_r + 2, (15, 23, 42), -1, cv2.LINE_AA)
                cv2.circle(annotated, (px, py), pt_r, (0, 255, 255), -1, cv2.LINE_AA)
                
        results.append({
            "index": f_num,
            "badge": badge,
            "best_name": best_name,
            "status": status,
            "similarity": f"{sim_pct:.1f}%",
            "threshold": f"{threshold*100:.0f}%",
            "margin": f"+{margin_pct:.1f}%" if margin_pct > 0 else "-",
            "profile_quality": f"{q_icon} {match_info.get('sample_count', 1)} sl.",
            "age": str(int(face["age"])) if face["age"] is not None else "-",
            "gender": str(face["gender"]) if face["gender"] is not None else "-",
            "race": str(face["race"]) if face["race"] is not None else "-",
            "bbox": str(face["bbox"]),
            "crop_bgr": face["crop_bgr"],
            "embedding": face["embedding"],
            "landmarks": face.get("landmarks"),
            "raw_face": face
        })
        
    return annotated, results
