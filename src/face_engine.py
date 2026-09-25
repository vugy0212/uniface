import os
import cv2
import numpy as np
from uniface import FaceAnalyzer, RetinaFace, ArcFace, FairFace

_analyzer = None

def get_analyzer(device="CPU"):
    global _analyzer
    if _analyzer is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.upper() == "CUDA" else ["CPUExecutionProvider"]
        try:
            detector = RetinaFace(confidence_threshold=0.45, providers=providers)
            recognizer = ArcFace(providers=providers)
            predictor = FairFace(providers=providers)
            _analyzer = FaceAnalyzer(detector=detector, recognizer=recognizer, predictors=[predictor])
        except Exception:
            detector = RetinaFace(confidence_threshold=0.45, providers=["CPUExecutionProvider"])
            recognizer = ArcFace(providers=["CPUExecutionProvider"])
            predictor = FairFace(providers=["CPUExecutionProvider"])
            _analyzer = FaceAnalyzer(detector=detector, recognizer=recognizer, predictors=[predictor])
    return _analyzer

def crop_face(image: np.ndarray, bbox, margin_ratio=0.25):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    bw = x2 - x1
    bh = y2 - y1
    
    x1 = max(0, int(x1 - bw * margin_ratio))
    y1 = max(0, int(y1 - bh * margin_ratio))
    x2 = min(w, int(x2 + bw * margin_ratio))
    y2 = min(h, int(y2 + bh * margin_ratio))
    
    return image[y1:y2, x1:x2].copy()

def extract_faces_from_image(image_bgr: np.ndarray, device="CPU"):
    analyzer = get_analyzer(device)
    faces = analyzer.analyze(image_bgr)
    
    extracted = []
    for idx, face in enumerate(faces):
        crop = crop_face(image_bgr, face.bbox)
        emb = face.embedding
        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
                
        extracted.append({
            "index": idx,
            "bbox": face.bbox.astype(int).tolist(),
            "landmarks": face.landmarks.tolist() if face.landmarks is not None else None,
            "confidence": float(face.confidence) if face.confidence is not None else 1.0,
            "embedding": emb,
            "crop_bgr": crop,
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

def match_face(query_embedding: np.ndarray, profiles: dict, threshold=0.50):
    if not profiles or query_embedding is None:
        return {
            "matched": False,
            "best_name": "Baza je prazna",
            "similarity": 0.0,
            "person_id": None,
            "status": "Nema baze",
            "quality_icon": "⚪",
            "quality_badge": "Nema uzoraka",
            "margin": 0.0,
            "crop_path": None
        }
    
    q_norm = np.linalg.norm(query_embedding)
    if q_norm > 0:
        query_embedding = query_embedding / q_norm
        
    candidate_scores = []
    
    for pid, p in profiles.items():
        # 1. Similarity to each individual sample
        sample_sims = [float(np.dot(query_embedding, s)) for s in p["samples"]]
        max_sim = max(sample_sims) if sample_sims else 0.0
        
        # 2. Similarity to centroid (smoothed multi-angle synthesized model)
        centroid_sim = float(np.dot(query_embedding, p["centroid"]))
        
        # 3. Hybrid metric:
        # If multiple samples exist, we blend the best individual angle and the centroid.
        # This increases true match confidence and filters out impostors.
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
        
    candidate_scores.sort(key=lambda x: x["similarity"], reverse=True)
    best = candidate_scores[0]
    second_sim = candidate_scores[1]["similarity"] if len(candidate_scores) > 1 else 0.0
    margin = best["similarity"] - second_sim
    
    matched = (best["similarity"] >= threshold)
    if matched:
        status = "Prepoznat"
    elif best["similarity"] >= 0.38:
        status = "Moguće poklapanje"
    else:
        status = "Nepoznat"
        
    return {
        "matched": matched,
        "best_name": best["person_name"],
        "person_id": best["person_id"],
        "similarity": best["similarity"],
        "status": status,
        "quality_icon": best["quality_icon"],
        "quality_badge": best["quality_badge"],
        "sample_count": best["count"],
        "margin": margin,
        "crop_path": best["crop_path"]
    }

def process_and_annotate(image_bgr: np.ndarray, all_samples: list[dict], threshold=0.50,
                         draw_landmarks=True, blur_unknown=False, device="CPU"):
    annotated = image_bgr.copy()
    faces_data = extract_faces_from_image(image_bgr, device)
    
    # Pre-build person profiles (multi-sample centroid models)
    profiles = build_person_profiles(all_samples)
    
    # Sort faces from left to right for clean visual indexing [#1], [#2]
    faces_data.sort(key=lambda f: f["bbox"][0])
    for i, f in enumerate(faces_data):
        f["display_index"] = i + 1
        
    results = []
    img_h, img_w = annotated.shape[:2]
    
    for face in faces_data:
        x1, y1, x2, y2 = face["bbox"]
        f_num = face["display_index"]
        
        match_info = match_face(face["embedding"], profiles, threshold)
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
            
        if blur_unknown and not is_known:
            fx1, fy1 = max(0, x1), max(0, y1)
            fx2, fy2 = min(img_w, x2), min(img_h, y2)
            sub = annotated[fy1:fy2, fx1:fx2]
            if sub.size > 0:
                ksize = max(15, (fx2 - fx1) // 3 * 2 + 1)
                blurred = cv2.GaussianBlur(sub, (ksize, ksize), 30)
                annotated[fy1:fy2, fx1:fx2] = blurred

        # Draw clean border box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)
        
        # Position label: If near top of image (y1 < 28), draw INSIDE box to prevent cut-off and overlapping
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
        
        # Draw landmarks if requested
        if draw_landmarks and face["landmarks"]:
            for pt in face["landmarks"]:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(annotated, (px, py), 2, (0, 255, 255), -1)
                
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
            "embedding": face["embedding"]
        })
        
    return annotated, results
