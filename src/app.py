import os
import sys
import uuid
import re
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from typing import Optional, List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import face_engine
import hardware
import backup
import config
import notifier
import photo_sorter
import queue
import threading
import shutil
from image_utils import imread_unicode, imwrite_unicode, save_image_dedup

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
CROPS_DIR = os.path.join(DATA_DIR, "crops")
SNAPSHOTS_DIR = config.get_snapshot_dir()
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)

def _get_single_state(state):
    if not isinstance(state, dict):
        return {"faces": [], "bgr": None, "selected_idx": 1, "saved_indices": set()}
    if "faces" not in state:
        state["faces"] = []
    if "bgr" not in state:
        state["bgr"] = None
    if "selected_idx" not in state:
        state["selected_idx"] = 1
    if "saved_indices" not in state or not isinstance(state["saved_indices"], set):
        state["saved_indices"] = set()
    return state

def get_person_dropdown_choices(filter_query=""):
    persons = db.get_all_persons()
    query = (filter_query or "").strip().lower()
    if query:
        persons = [
            p for p in persons
            if query in p["name"].lower() or query == str(p["id"]) or query in (p["notes"] or "").lower()
        ]
    return [f"{p['id']}: {p['name']} ({p['sample_count']} slika)" for p in persons]

def update_both_person_dropdowns():
    choices = get_person_dropdown_choices()
    return gr.update(choices=choices), gr.update(choices=choices)

def clean_filename_to_name(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    base = re.sub(r'[\-_]+', ' ', base)
    base = re.sub(r'\d+', '', base).strip()
    return base.title() if base else "Nova Osoba"

def get_profile_badge(count: int):
    if count <= 1:
        return "🔴 Osnovno (1 slika)"
    elif count == 2:
        return "🟡 Dobro (2 slike)"
    else:
        return f"🟢 Izvrsno ({count} slika)"

def format_role_badge(role_str: str) -> str:
    r = (role_str or "standard").lower()
    if r == "vip":
        return "⭐ VIP"
    elif r == "blacklist":
        return "🚨 Crna lista"
    return "Standard"

def refresh_database_view(search_query=""):
    all_persons = db.get_all_persons()
    stats = db.get_stats()
    query = (search_query or "").strip().lower()
    
    if query:
        filtered = [
            p for p in all_persons
            if query in p["name"].lower() or query == str(p["id"]) or query in (p["notes"] or "").lower() or query in (p.get("role", "") or "").lower()
        ]
        stats_text = f"🔍 Pronađeno: **{len(filtered)}** od ukupno **{stats['total_persons']}** osoba (Filter: *'{search_query}'*)"
    else:
        filtered = all_persons
        stats_text = f"Ukupno osoba u bazi: **{stats['total_persons']}** | Ukupno slika/uzoraka lica: **{stats['total_samples']}**"
        
    rows = []
    for p in filtered:
        rows.append([
            p["id"],
            p["name"],
            format_role_badge(p.get("role", "standard")),
            p["sample_count"],
            get_profile_badge(p["sample_count"]),
            p["notes"] or "-",
            p["created_at"]
        ])
        
    return rows, stats_text

# ---------------- REAL-TIME DETECTION CARDS GENERATOR ----------------
def generate_detection_cards_html(results, show_all_faces=False):
    if not results:
        return """
        <div class="detection-panel-inner">
            <div class="panel-header">
                <div class="panel-title">
                    <span class="pulse-icon"></span> Real-time Detekcija
                </div>
                <span class="panel-badge">Spremno</span>
            </div>
            <div class="detection-empty-state">
                <div class="radar-scan-box">
                    <div class="radar-beam"></div>
                </div>
                <div class="empty-title">Čekanje na unos</div>
                <div class="empty-sub">Učitajte fotografiju ili pokrenite live kameru za biometrijsku analizu lica u stvarnom vremenu.</div>
            </div>
        </div>
        """
    import base64
    
    num_total = len(results)
    num_recognized = sum(1 for r in results if r.get("status") == "Prepoznat")
    
    if not show_all_faces:
        display_results = [r for r in results if r.get("status") == "Prepoznat"]
    else:
        display_results = results

    if not display_results:
        badge_text = f"0 / {num_total} lica"
        return f"""
        <div class="detection-panel-inner">
            <div class="panel-header">
                <div class="panel-title">
                    <span class="pulse-icon active"></span> Real-time Detekcija
                </div>
                <span class="panel-badge active" style="color: #94a3b8; border-color: rgba(148, 163, 184, 0.4);">{badge_text}</span>
            </div>
            <div class="detection-empty-state">
                <div class="radar-scan-box">
                    <div class="radar-beam"></div>
                </div>
                <div class="empty-title" style="color: #f59e0b;">Nema prepoznatih lica</div>
                <div class="empty-sub">
                    Pronađeno je <b>{num_total}</b> lica u kadru, ali nijedno ne prelazi zadani prag.<br><br>
                    Uključite kvačicu <i>"Prikaži i nepoznata lica"</i> iznad za prikaz svih lica s postotkom sličnosti ili snizite prag.
                </div>
            </div>
        </div>
        """

    cards = []
    for r in display_results:
        crop_bgr = r.get("crop_bgr")
        img_src = ""
        if crop_bgr is not None and isinstance(crop_bgr, np.ndarray) and crop_bgr.size > 0:
            try:
                success, buffer = cv2.imencode('.jpg', crop_bgr)
                if success:
                    img_b64 = base64.b64encode(buffer).decode('utf-8')
                    img_src = f"data:image/jpeg;base64,{img_b64}"
            except Exception:
                img_src = ""
        
        sim_val = r.get("similarity", 0)
        pct = 0.0
        if isinstance(sim_val, (int, float)):
            pct = float(sim_val) * 100
            sim_str = f"{pct:.1f}% Match"
        else:
            sim_str = f"{sim_val} Match"
            
        status = r.get("status", "Nepoznat")
        name = r.get("best_name", "Nepoznata osoba")
        
        # Determine person role badge
        p_role = "standard"
        if status in ("Prepoznat", "Moguće poklapanje") and name not in ("Nepoznata osoba", "Nepoznato"):
            try:
                p_role = db.get_person_role(name)
            except Exception:
                p_role = "standard"
                
        if p_role == "blacklist":
            badge_cls = "match-blacklist is-blacklist"
            status_text = "Crna lista"
            role_badge = '<span class="role-pill role-blacklist">🚨 CRNA LISTA</span>'
        elif p_role == "vip":
            badge_cls = "match-vip is-vip"
            status_text = "VIP"
            role_badge = '<span class="role-pill role-vip">⭐ VIP</span>'
        elif status == "Prepoznat":
            badge_cls = "match-success"
            status_text = "Prepoznato"
            role_badge = '<span class="role-pill role-verified">✓ Verificirano</span>'
        elif status == "Moguće poklapanje":
            badge_cls = "match-warning"
            status_text = "Moguće"
            role_badge = '<span class="role-pill role-possible">? Provjera</span>'
        else:
            badge_cls = "match-unknown"
            status_text = "Nepoznato"
            role_badge = ''
            
        age = r.get("age", "-")
        gender = r.get("gender", "-")
        meta_parts = []
        if age and str(age) != "-":
            meta_parts.append(f'<span class="meta-item"><i class="meta-label">Dob:</i> <b>~{age}g</b></span>')
        if gender and str(gender) != "-":
            meta_parts.append(f'<span class="meta-item"><i class="meta-label">Spol:</i> <b>{gender}</b></span>')
        meta_html = ' <span class="meta-sep">•</span> '.join(meta_parts) if meta_parts else '<span class="meta-item">Biometrijski profil</span>'
        
        cards.append(f"""
        <div class="cyber-detection-card {badge_cls}">
            <div class="card-avatar-wrap">
                <img src="{img_src}" class="card-avatar" alt="{name}" />
                <span class="card-status-dot"></span>
            </div>
            <div class="card-details">
                <div class="card-name-row">
                    <span class="card-name" title="{name}">{name}</span>
                    {role_badge}
                </div>
                <div class="card-meta">
                    {meta_html}
                </div>
                <div class="card-similarity-badge">
                    <span class="sim-pill">{sim_str}</span>
                </div>
                <div class="card-conf-track" title="Sličnost: {pct:.1f}%">
                    <div class="card-conf-fill" style="width: {min(100.0, max(6.0, pct)):.1f}%;"></div>
                </div>
            </div>
        </div>
        """)
        
    cards_html = "".join(cards)
    if not show_all_faces:
        badge_text = f"{num_recognized} prepoznato" if num_recognized == num_total else f"{num_recognized} / {num_total} lica"
    else:
        badge_text = f"{num_total} lica"
        
    return f"""
    <div class="detection-panel-inner">
        <div class="panel-header">
            <div class="panel-title">
                <span class="pulse-icon active"></span> Real-time Detekcija
            </div>
            <span class="panel-badge active">{badge_text}</span>
        </div>
        <div class="cyber-cards-scroll">
            {cards_html}
        </div>
    </div>
    """

# ---------------- PREPOZNAVANJE ----------------
def recognize_faces(image, threshold, draw_landmarks, blur_unknown, blur_all=False, show_all_faces=False, cached_faces=None):
    if image is None:
        return None, [], [], "⚠️ Molimo učitajte sliku za analizu.", gr.update(choices=[], value=None), [], generate_detection_cards_html([], show_all_faces=show_all_faces)
    
    try:
        img_bgr, err = imread_unicode(image)
        if img_bgr is None or not isinstance(img_bgr, np.ndarray) or img_bgr.size == 0:
            return None, [], [], f"❌ Greška pri obradi slike: {err or 'Neispravan format slike'}", gr.update(choices=[], value=None), [], generate_detection_cards_html([], show_all_faces=show_all_faces)
            
        annotated_bgr, results = face_engine.process_and_annotate(
            img_bgr,
            threshold=float(threshold),
            draw_landmarks=bool(draw_landmarks),
            blur_unknown=bool(blur_unknown),
            blur_all=bool(blur_all),
            cached_faces=cached_faces
        )
        
        if annotated_bgr is not None and isinstance(annotated_bgr, np.ndarray) and annotated_bgr.size > 0:
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            annotated_rgb = None
        
        table_data = []
        crops_gallery = []
        candidate_choices = []
        
        for r in results:
            table_data.append([
                f"Lice #{r['index']}",
                r["badge"],
                r["best_name"],
                r["similarity"],
                r["threshold"],
                r["margin"],
                r["profile_quality"],
                r["age"],
                r["gender"]
            ])
            crop_bgr = r.get("crop_bgr")
            if crop_bgr is not None and isinstance(crop_bgr, np.ndarray) and crop_bgr.size > 0:
                try:
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    crop_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
            else:
                crop_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
            
            if r["status"] == "Prepoznat":
                caption = f"[#{r['index']}] {r['best_name']} ({r['similarity']})"
            elif r["status"] == "Moguće poklapanje":
                caption = f"[#{r['index']}] {r['best_name']}? ({r['similarity']} - ispod praga)"
            else:
                caption = f"[#{r['index']}] Nepoznato (max {r['similarity']})"
                
            crops_gallery.append((crop_rgb, caption))
            candidate_choices.append(f"[#{r['index']}] {r['best_name']} ({r['similarity']})")
                
        num_recognized = sum(1 for r in results if r["status"] == "Prepoznat")
        num_possible = sum(1 for r in results if r["status"] == "Moguće poklapanje")
        num_unknown = sum(1 for r in results if r["status"] == "Nepoznat")
        
        summary = f"🔍 Pronađeno lica: **{len(results)}** | ✅ Prepoznato: **{num_recognized}** | ⚠️ Moguće (ispod praga): **{num_possible}** | ❌ Nepoznato: **{num_unknown}**"
        dropdown_update = gr.update(choices=candidate_choices, value=candidate_choices[0] if candidate_choices else None)
        cards_html = generate_detection_cards_html(results, show_all_faces=show_all_faces)
        return annotated_rgb, crops_gallery, table_data, summary, dropdown_update, results, cards_html
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, [], [], f"⚠️ Greška pri analizi slike: {str(e)}", gr.update(choices=[], value=None), [], generate_detection_cards_html([], show_all_faces=show_all_faces)

def on_recognition_gallery_click(evt: gr.SelectData, rec_faces):
    faces = rec_faces or []
    if evt.index is not None and 0 <= evt.index < len(faces):
        r = faces[evt.index]
        face_choice = f"[#{r['index']}] {r['best_name']} ({r['similarity']})"
        default_name = r["best_name"] if r["best_name"] != "Nepoznato" else ""
        return gr.update(value=face_choice), gr.update(value=default_name)
    return gr.update(), gr.update()

def quick_add_face_to_db(selected_face_str, new_name, rec_faces):
    if not selected_face_str:
        return "⚠️ Niste odabrali lice sa slike.", gr.update(), gr.update(), gr.update(), gr.update()
    if not new_name or not new_name.strip():
        return "⚠️ Morate unijeti ime osobe.", gr.update(), gr.update(), gr.update(), gr.update()
    
    try:
        f_num = int(selected_face_str.split(']')[0].replace('[#', '').strip())
        target_face = None
        for f in (rec_faces or []):
            if f["index"] == f_num:
                target_face = f
                break
        if not target_face:
            return "❌ Odabrano lice više nije dostupno.", gr.update(), gr.update(), gr.update(), gr.update()
    except Exception as e:
        return f"❌ Pogrešan format odabira ({e})", gr.update(), gr.update(), gr.update(), gr.update()
        
    person_name = new_name.strip()
    person_id = db.get_or_create_person(person_name)
    
    crop_filename = f"crop_{person_id}_{uuid.uuid4().hex[:8]}.jpg"
    crop_path = os.path.join(CROPS_DIR, crop_filename)
    imwrite_unicode(crop_path, target_face["crop_bgr"])
    
    db.add_face_sample(person_id, crop_path, crop_path, target_face["embedding"])
    
    msg = f"✅ Lice [#{f_num}] uspješno dodano osobi **{person_name}** u bazu podataka!"
    choices = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    samples = db.get_person_samples(person_id)
    choice_val = f"{person_id}: {person_name} ({len(samples)} slika)"
    return msg, gr.update(choices=choices, value=choice_val), gr.update(choices=choices, value=choice_val), table_view, stats_view

# ---------------- POJEDINAČNI UNOS / OZNAČAVANJE LICA NA GRUPNOJ SLICI ----------------
def find_face_at_coords(x, y, faces):
    if not faces:
        return None
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return f
    best_f = None
    min_dist = float("inf")
    for f in faces:
        x1, y1, x2, y2 = f["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        dist = (cx - x)**2 + (cy - y)**2
        if dist < min_dist:
            min_dist = dist
            best_f = f
    return best_f

def render_annotated_group_image(img_bgr, faces, selected_index=1, saved_indices=None):
    if saved_indices is None:
        saved_indices = set()
    annotated = img_bgr.copy()
    for f in faces:
        idx = f["display_index"]
        x1, y1, x2, y2 = f["bbox"]
        is_selected = (idx == selected_index)
        is_saved = (idx in saved_indices)
        
        if is_selected:
            color = (30, 235, 70)   # Vibrant Green
            thickness = 4
            lbl = f"ODABRANO: Lice #{idx}"
        elif is_saved:
            color = (240, 160, 40)  # Blue/Cyan
            thickness = 2
            lbl = f"Lice #{idx} (Spremljeno)"
        else:
            color = (0, 215, 255)   # Gold/Yellow
            thickness = 2
            lbl = f"Lice #{idx}"
            
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        
        font_scale = 0.65 if is_selected else 0.55
        font_thick = 2
        (tw, th), bl = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        tag_y1 = max(0, y1 - th - 10)
        tag_y2 = y1
        tag_x2 = min(annotated.shape[1], x1 + tw + 12)
        cv2.rectangle(annotated, (x1, tag_y1), (tag_x2, tag_y2), color, -1)
        
        text_color = (0, 0, 0)
        cv2.putText(annotated, lbl, (x1 + 6, tag_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)
        
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

def build_face_choices_and_gallery(faces, selected_index=1, saved_indices=None):
    if saved_indices is None:
        saved_indices = set()
    choices = []
    gallery = []
    for f in faces:
        idx = f["display_index"]
        status = "✅ Spremljeno" if idx in saved_indices else "🎯 Nije spremljeno"
        choice_text = f"Lice #{idx} ({status})"
        choices.append(choice_text)
        
        crop_rgb = cv2.cvtColor(f["crop_bgr"], cv2.COLOR_BGR2RGB)
        caption = f"Lice #{idx} | {status}"
        if f.get("age") is not None:
            caption += f" (~{int(f['age'])}g)"
        gallery.append((crop_rgb, caption))
        
    selected_choice = None
    for c in choices:
        if c.startswith(f"Lice #{selected_index}"):
            selected_choice = c
            break
    if not selected_choice and choices:
        selected_choice = choices[0]
        
    return choices, gallery, selected_choice

def set_active_face(target_index, state):
    state = _get_single_state(state)
    faces = state["faces"]
    bgr = state["bgr"]
    saved = state["saved_indices"]
    if not faces or bgr is None:
        return None, None, "Nema učitanih lica.", gr.update(), state
        
    valid_face = None
    for f in faces:
        if f["display_index"] == target_index:
            valid_face = f
            break
            
    if not valid_face:
        valid_face = faces[0]
        target_index = valid_face["display_index"]
        
    state["selected_idx"] = target_index
    
    annotated_rgb = render_annotated_group_image(
        bgr, faces,
        selected_index=target_index,
        saved_indices=saved
    )
    
    crop_rgb = cv2.cvtColor(valid_face["crop_bgr"], cv2.COLOR_BGR2RGB)
    
    status_str = " (već spremljeno u bazu)" if target_index in saved else ""
    info = f"🎯 Trenutno odabrano: **Lice #{target_index}**{status_str}."
    if valid_face.get("age") is not None:
        info += f" | Procjena dobi: ~{int(valid_face['age'])} god, Spol: {valid_face['gender']}"
        
    choices, _, sel_choice = build_face_choices_and_gallery(
        faces,
        selected_index=target_index,
        saved_indices=saved
    )
    
    return annotated_rgb, crop_rgb, info, gr.update(value=sel_choice), state

def on_single_image_uploaded(image, current_name, state):
    new_state = {"faces": [], "bgr": None, "selected_idx": 1, "saved_indices": set()}
    c_name = str(current_name or "").strip()
    default_btn = f"💾 Spremi dodatno lice za: {c_name}" if c_name else "💾 Spremi odabrano lice u bazu"
    
    if image is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(choices=[], value=None, visible=False),
            None, "Učitajte fotografiju za automatsku detekciju lica.",
            gr.update(value=default_btn),
            new_state
        )
        
    img_bgr, err = imread_unicode(image)
    if img_bgr is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(choices=[], value=None, visible=False),
            None, f"❌ Greška pri čitanju slike: {err}",
            gr.update(value=default_btn),
            new_state
        )
        
    faces = face_engine.extract_faces_from_image(img_bgr)
    if not faces:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(choices=[], value=None, visible=False),
            None, "⚠️ Na slici NIJE pronađeno lice. Pokušajte s jasnijom slikom.",
            gr.update(value=default_btn),
            new_state
        )
        
    faces.sort(key=lambda f: f["bbox"][0])
    for i, f in enumerate(faces):
        f["display_index"] = i + 1
        
    new_state["bgr"] = img_bgr
    new_state["faces"] = faces
    new_state["selected_idx"] = 1
    new_state["saved_indices"] = set()
    
    annotated_rgb = render_annotated_group_image(img_bgr, faces, selected_index=1, saved_indices=new_state["saved_indices"])
    choices, gallery, sel_choice = build_face_choices_and_gallery(faces, selected_index=1, saved_indices=new_state["saved_indices"])
    
    f1 = faces[0]
    crop1_rgb = cv2.cvtColor(f1["crop_bgr"], cv2.COLOR_BGR2RGB)
    if c_name:
        info1 = f"👥 Pronađeno **{len(faces)}** lica na slici! 🎯 Spremno za spremanje uzorka za osobu: **{c_name}**."
        btn_text = f"💾 Spremi dodatno lice za: {c_name}"
    else:
        info1 = f"👥 Pronađeno **{len(faces)}** lica na slici! Trenutno je odabrano: **Lice #1**."
        btn_text = "💾 Spremi odabrano lice u bazu"
        
    if f1.get("age") is not None:
        info1 += f" | Procjena dobi: ~{int(f1['age'])} god, Spol: {f1['gender']}"
        
    return (
        gr.update(value=annotated_rgb, visible=True),
        gr.update(value=gallery, visible=True),
        gr.update(choices=choices, value=sel_choice, visible=True),
        crop1_rgb,
        info1,
        gr.update(value=btn_text),
        new_state
    )

def on_radio_face_change(choice_str, state):
    state = _get_single_state(state)
    if not choice_str:
        return gr.update(), gr.update(), gr.update(), gr.update(), state
    try:
        idx = int(str(choice_str).split("#")[1].split(" ")[0].strip())
        return set_active_face(idx, state)
    except Exception:
        return gr.update(), gr.update(), gr.update(), gr.update(), state

def on_crop_gallery_select(evt: gr.SelectData, state):
    state = _get_single_state(state)
    faces = state["faces"]
    if evt.index is not None and 0 <= evt.index < len(faces):
        idx = evt.index + 1
        return set_active_face(idx, state)
    return gr.update(), gr.update(), gr.update(), gr.update(), state

def on_single_image_click(evt: gr.SelectData, state):
    state = _get_single_state(state)
    faces = state["faces"]
    if not faces or not evt.index:
        return gr.update(), gr.update(), gr.update(), gr.update(), state
    try:
        x, y = evt.index[0], evt.index[1]
        f = find_face_at_coords(x, y, faces)
        if f:
            return set_active_face(f["display_index"], state)
    except Exception:
        pass
    return gr.update(), gr.update(), gr.update(), gr.update(), state

def save_single_person(name, notes, face_choice_str, role, state):
    state = _get_single_state(state)
    faces = state["faces"]
    bgr = state["bgr"]
    saved = state["saved_indices"]
    
    if not name or not name.strip():
        return (
            "⚠️ Ime osobe je obavezno!",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            state,
            gr.update(), gr.update()
        )
    if not faces:
        return (
            "⚠️ Niste učitali sliku ili na slici nema lica!",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            state,
            gr.update(), gr.update()
        )
        
    target_idx = state.get("selected_idx", 1)
    if face_choice_str:
        try:
            target_idx = int(str(face_choice_str).split("#")[1].split(" ")[0].strip())
        except Exception:
            target_idx = state.get("selected_idx", 1)
            
    target_face = None
    for f in faces:
        if f["display_index"] == target_idx:
            target_face = f
            break
    if not target_face:
        target_face = faces[0]
        target_idx = target_face["display_index"]
        
    person_name = name.strip()
    person_role = (role or "standard").strip().lower()
    person_id = db.get_or_create_person(person_name, notes or "", role=person_role)
    try:
        db.update_person(person_id, person_name, notes or "", role=person_role)
    except Exception:
        pass
    
    orig_img_to_save = bgr if bgr is not None else target_face["crop_bgr"]
    orig_path = save_image_dedup(orig_img_to_save, UPLOADS_DIR, prefix="orig")
        
    crop_filename = f"crop_{person_id}_{uuid.uuid4().hex[:8]}.jpg"
    crop_path = os.path.join(CROPS_DIR, crop_filename)
    imwrite_unicode(crop_path, target_face["crop_bgr"])
    
    db.add_face_sample(person_id, orig_path, crop_path, target_face["embedding"], target_face["confidence"])
    saved.add(target_idx)
    state["saved_indices"] = saved
    
    samples = db.get_person_samples(person_id)
    choices_db = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    
    next_idx = None
    for f in faces:
        if f["display_index"] not in saved:
            next_idx = f["display_index"]
            break
            
    choice_str = f"{person_id}: {person_name} ({len(samples)} slika)"
    
    if next_idx is not None:
        state["selected_idx"] = next_idx
        msg = f"🎉 **Lice #{target_idx}** uspješno spremljeno za osobu **{person_name}** ({format_role_badge(person_role)})! Sada upišite ime za sljedeću osobu sa slike (**Lice #{next_idx}**)."
        name_out = ""
        notes_out = ""
        btn_text = "💾 Spremi odabrano lice u bazu"
        picker_val = None
        avatar_html = render_person_avatar_html(None)
        edit_name = ""
        edit_notes = ""
        role_out = "standard"
        edit_role = "standard"
    else:
        # Sve osobe sa slike su spremljene ili je pojedinačni portret (najčešći slučaj)
        # Osoba ostaje trajno selektirana radi fluidnog unosa dodatnih slika!
        state["selected_idx"] = 1
        msg = (
            f"🎉 **Lice #{target_idx}** uspješno spremljeno za osobu **{person_name}** ({format_role_badge(person_role)}, ukupno {len(samples)} slika)!\n"
            f"✨ **{person_name}** ostaje odabran(a) – samo učitajte sljedeću sliku ili snimite kamerom i kliknite **'Spremi dodatno lice'** za novi uzorak."
        )
        name_out = person_name
        notes_out = notes or ""
        btn_text = f"💾 Spremi dodatno lice za: {person_name}"
        picker_val = choice_str
        avatar_html = render_person_avatar_html(person_id)
        edit_name = person_name
        edit_notes = notes or ""
        role_out = person_role
        edit_role = person_role
        
    annotated_rgb = render_annotated_group_image(
        bgr, faces,
        selected_index=state["selected_idx"],
        saved_indices=saved
    )
    
    new_choices, new_gallery, sel_choice = build_face_choices_and_gallery(
        faces,
        selected_index=state["selected_idx"],
        saved_indices=saved
    )
    
    active_face = None
    for f in faces:
        if f["display_index"] == state["selected_idx"]:
            active_face = f
            break
    if not active_face:
        active_face = target_face
        
    crop_rgb = cv2.cvtColor(active_face["crop_bgr"], cv2.COLOR_BGR2RGB)
    status_str = " (već spremljeno)" if state["selected_idx"] in saved else ""
    info = f"🎯 Trenutno odabrano: **Lice #{state['selected_idx']}**{status_str}."
    if active_face.get("age") is not None:
        info += f" | Procjena dobi: ~{int(active_face['age'])} god, Spol: {active_face['gender']}"
        
    return (
        msg,
        gr.update(choices=choices_db, value=choice_str),
        gr.update(choices=choices_db, value=picker_val),
        table_view,
        stats_view,
        annotated_rgb,
        crop_rgb,
        info,
        gr.update(choices=new_choices, value=sel_choice),
        name_out,
        notes_out,
        btn_text,
        avatar_html,
        edit_name,
        edit_notes,
        new_gallery,
        state,
        edit_role,
        role_out
    )

def clear_single_form():
    empty_state = {"faces": [], "bgr": None, "selected_idx": 1, "saved_indices": set()}
    return (
        "", "", None,
        gr.update(value=None, visible=False),
        gr.update(value=[], visible=False),
        gr.update(choices=[], value=None, visible=False),
        None, "Učitajte fotografiju za automatsku detekciju lica.",
        "💾 Spremi odabrano lice u bazu", "Formular očišćen za novu osobu.",
        gr.update(value=None),
        empty_state,
        render_person_avatar_html(None),
        "standard"
    )

# ---------------- MASOVNI (BATCH) UNOS ----------------
def batch_enroll_files(naming_mode, single_name, files):
    if not files:
        return "⚠️ Niste odabrali datoteke za unos.", gr.update(), gr.update(), gr.update(), gr.update()
        
    success_list = []
    fail_list = []
    
    for f in files:
        img_bgr, err = imread_unicode(f)
        fname = getattr(f, "orig_name", None) or getattr(f, "name", None) or (f if isinstance(f, str) else "slika")
        base_display_name = os.path.basename(str(fname))
        
        if img_bgr is None:
            fail_list.append(f"{base_display_name} (neispravna slika: {err})")
            continue
            
        faces = face_engine.extract_faces_from_image(img_bgr, with_attributes=False)
        if not faces:
            fail_list.append(f"{base_display_name} (lice nije detektirano)")
            continue
            
        faces.sort(key=lambda x: (x["bbox"][2]-x["bbox"][0]) * (x["bbox"][3]-x["bbox"][1]), reverse=True)
        main_face = faces[0]
        
        if naming_mode == "Ime iz naziva datoteke":
            person_name = clean_filename_to_name(base_display_name)
        else:
            if not single_name or not single_name.strip():
                return "⚠️ Morate upisati ime osobe za sve slike.", gr.update(), gr.update(), gr.update(), gr.update()
            person_name = single_name.strip()
            
        person_id = db.get_or_create_person(person_name)
        
        orig_path = save_image_dedup(img_bgr, UPLOADS_DIR, prefix="orig")
        
        crop_filename = f"crop_{person_id}_{uuid.uuid4().hex[:8]}.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        imwrite_unicode(crop_path, main_face["crop_bgr"])
        
        db.add_face_sample(person_id, orig_path, crop_path, main_face["embedding"], main_face["confidence"])
        success_list.append(f"**{person_name}** ({base_display_name})")
        
    report = f"### 📊 Rezultat obrade:\n* ✅ **Uspješno spremljeno:** {len(success_list)}\n"
    if success_list:
        report += "* " + ", ".join(success_list[:15]) + ("..." if len(success_list) > 15 else "") + "\n"
    if fail_list:
        report += f"* ⚠️ **Preskočeno ({len(fail_list)}):**\n  - " + "\n  - ".join(fail_list)
        
    choices = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    return report, gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), table_view, stats_view

# ---------------- PREGLED I BRISANJE ----------------
def parse_person_id(val):
    if not val:
        return None
    val_str = str(val).strip()
    if ":" in val_str:
        val_str = val_str.split(":")[0].strip()
    try:
        return int(val_str)
    except Exception:
        return None

def view_person_details(selected_person_str):
    person_id = parse_person_id(selected_person_str)
    if person_id is None:
        return [], "Kliknite na osobu u tablici ili je pretražite iznad.", gr.update(choices=[], value=None)
        
    person = db.get_person(person_id)
    if not person:
        return [], "Osoba nije pronađena u bazi.", gr.update(choices=[], value=None)
        
    samples = db.get_person_samples(person_id)
    gallery = []
    sample_choices = []
    for s in samples:
        if os.path.exists(s["crop_path"]):
            gallery.append((s["crop_path"], f"ID: {s['id']} ({s['created_at']})"))
            sample_choices.append(f"Slika #{s['id']}")
            
    count = len(samples)
    badge = get_profile_badge(count)
    p_role = person.get("role", "standard")
    role_display = format_role_badge(p_role)
    if count == 1:
        rec = "💡 **Savjet za točnost:** Osoba ima samo 1 sliku. Dodajte sliku pod blagim kutom (polu-profil) ili s osmijehom kako bi prepoznavanje bilo otporno na kretanje i različito osvjetljenje."
        centroid_status = "Korišten je pojedinačni vektor."
    elif count == 2:
        rec = "💡 **Savjet za točnost:** Dobar profil (2 slike). Za maksimalnu pouzdanost pod teškim uvjetima (sunce, sjene) možete dodati još 1 profilnu sliku."
        centroid_status = "✅ **Aktiviran je sintetizirani centroid** (kombinira 2 biometrijska kuta za veću otpornost)."
    else:
        rec = "🎯 **Vrhunski profil!** Osoba ima 3 ili više slika. Pokriveni su višestruki kutevi i crte lica."
        centroid_status = f"✅ **Aktiviran je visoko-precizni centroid** (objedinjuje {count} vektora u optimalni biometrijski model)."

    info_text = f"""
    ### 👤 {person['name']}
    * **Sigurnosni status / Uloga:** {role_display}
    * **Kvaliteta profila:** {badge}
    * **Biometrijski model:** {centroid_status}
    * **Bilješke:** {person['notes'] or 'Nema bilješki'}
    * **Registrirano slika:** {count}
    
    {rec}
    """
    return gallery, info_text, gr.update(choices=sample_choices, value=sample_choices[0] if sample_choices else None)

def render_person_avatar_html(person_id):
    empty_html = """
    <div class="cyber-person-avatar-thumb">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/>
            <circle cx="12" cy="7" r="4"/>
        </svg>
    </div>
    """
    if not person_id:
        return empty_html
        
    samples = db.get_person_samples(person_id)
    if samples and os.path.exists(samples[0]["crop_path"]):
        p = samples[0]["crop_path"]
        try:
            import base64
            with open(p, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            return f"""
            <div class="cyber-person-avatar-thumb">
                <img src="data:image/jpeg;base64,{b64}" alt="Avatar" />
            </div>
            """
        except Exception:
            pass
    return empty_html

def on_table_select(table_data, is_multi_mode, current_batch, evt: gr.SelectData):
    if evt.index is None:
        return (
            gr.update(), [], "", gr.update(),
            gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "",
            gr.update(), render_person_avatar_html(None),
            gr.update(), gr.update(), "",
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update()
        )
        
    row_idx = evt.index[0]
    person_id = None
    try:
        if hasattr(table_data, "iloc"):
            person_id = int(table_data.iloc[row_idx]["ID"])
        elif isinstance(table_data, list) and row_idx < len(table_data):
            person_id = int(table_data[row_idx][0])
    except Exception:
        pass
        
    if person_id is None:
        all_p = db.get_all_persons()
        if row_idx < len(all_p):
            person_id = all_p[row_idx]["id"]
            
    if person_id is None:
        return (
            gr.update(), [], "", gr.update(),
            gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "",
            gr.update(), render_person_avatar_html(None),
            gr.update(), gr.update(), "",
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update()
        )
        
    p = db.get_person(person_id)
    if not p:
        return (
            gr.update(), [], "", gr.update(),
            gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "",
            gr.update(), render_person_avatar_html(None),
            gr.update(), gr.update(), "",
            gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update()
        )
        
    choice = f"{p['id']}: {p['name']} ({p['sample_count']} slika)"

    if is_multi_mode:
        batch_list = list(current_batch or [])
        if choice in batch_list:
            batch_list.remove(choice)
            status_txt = f"➖ Uklonjeno s popisa za brisanje: **{p['name']}** (Preostalo označeno: **{len(batch_list)}** osoba)"
        else:
            batch_list.append(choice)
            status_txt = f"➕ Označeno za brisanje: **{p['name']}** (Ukupno označeno: **{len(batch_list)}** osoba)"
            
        btn_label = f"🗑️ Obriši označene osobe ({len(batch_list)})"
        return (
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            batch_list, btn_label, status_txt,
            gr.update(), gr.update()
        )

    gallery, info, sample_drop = view_person_details(choice)
    avatar_html = render_person_avatar_html(p["id"])
    
    name_val = p["name"]
    notes_val = p["notes"] or ""
    role_val = p.get("role", "standard")
    btn_text = f"💾 Spremi dodatno lice za: {p['name']}"
    status_msg = f"📌 Odabrano za unos novih slika: **{p['name']}** ({p['sample_count']} slika, {format_role_badge(role_val)}). Učitajte sliku i kliknite Spremi."
    
    return (
        choice, gallery, info, sample_drop,
        name_val, notes_val, btn_text, status_msg,
        gr.update(value=choice),
        avatar_html,
        name_val, notes_val, "",
        gr.update(), gr.update(), gr.update(),
        role_val, role_val
    )

def on_existing_person_picked(selected_choice):
    person_id = parse_person_id(selected_choice)
    if person_id is None:
        return (
            gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "",
            [], "", gr.update(), gr.update(),
            render_person_avatar_html(None),
            gr.update(), gr.update(), "",
            "standard", "standard"
        )
        
    p = db.get_person(person_id)
    if not p:
        return (
            gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "",
            [], "", gr.update(), gr.update(),
            render_person_avatar_html(None),
            gr.update(), gr.update(), "",
            "standard", "standard"
        )
        
    name_val = p["name"]
    notes_val = p["notes"] or ""
    role_val = p.get("role", "standard")
    btn_text = f"💾 Spremi dodatno lice za: {p['name']}"
    status_msg = f"📌 Odabrano za unos novih slika: **{p['name']}** ({p['sample_count']} slika, {format_role_badge(role_val)}). Učitajte sliku i kliknite Spremi."
    choice_str = f"{p['id']}: {p['name']} ({p['sample_count']} slika)"
    gallery, info, sample_drop = view_person_details(choice_str)
    avatar_html = render_person_avatar_html(p["id"])
    
    return (
        name_val, notes_val, btn_text, status_msg,
        gallery, info, sample_drop, gr.update(value=choice_str),
        avatar_html,
        name_val, notes_val, "",
        role_val, role_val
    )

def on_manage_person_change(selected_person_str):
    person_id = parse_person_id(selected_person_str)
    if person_id is None:
        return [], "Kliknite na osobu u tablici ili je pretražite iznad.", gr.update(choices=[], value=None), "", "", "", "standard"
        
    p = db.get_person(person_id)
    if not p:
        return [], "Osoba nije pronađena u bazi.", gr.update(choices=[], value=None), "", "", "", "standard"
        
    gallery, info, sample_drop = view_person_details(selected_person_str)
    return gallery, info, sample_drop, p["name"], p["notes"] or "", "", p.get("role", "standard")

def on_table_search_changed(search_query):
    rows, stats = refresh_database_view(search_query)
    choices = get_person_dropdown_choices(search_query)
    return rows, stats, gr.update(choices=choices)

def on_table_search_clear():
    rows, stats = refresh_database_view("")
    choices = get_person_dropdown_choices("")
    return "", rows, stats, gr.update(choices=choices, value=None)

def delete_selected_person(selected_person_str):
    person_id = parse_person_id(selected_person_str)
    if person_id is None:
        return "⚠️ Niste odabrali osobu.", gr.update(), gr.update(), gr.update(), gr.update(), [], "", "", "", ""
        
    p = db.get_person(person_id)
    if p:
        db.delete_person(person_id)
        msg = f"🗑️ Osoba **{p['name']}** i sve njezine fotografije su obrisane iz baze."
    else:
        msg = "Osoba ne postoji."
        
    choices = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    return msg, gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), table_view, stats_view, [], "", "", "", ""

def update_person_handler(selected_person_str, new_name, new_notes, new_role="standard"):
    person_id = parse_person_id(selected_person_str)
    if person_id is None:
        return (
            "⚠️ Nije odabrana valjana osoba za uređivanje. Kliknite na redak u tablici ili odaberite osobu iz padajućeg izbornika.",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    new_name = (new_name or "").strip()
    if not new_name:
        return (
            "⚠️ Ime i prezime osobe ne smije biti prazno.",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    role_clean = (new_role or "standard").strip().lower()
    try:
        db.update_person(person_id, new_name, (new_notes or "").strip(), role=role_clean)
    except ValueError as ve:
        return (
            f"⚠️ {str(ve)}",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    except Exception as e:
        return (
            f"❌ Greška pri spremanju izmjena: {e}",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    updated_person = db.get_person(person_id)
    sample_count = updated_person["sample_count"] if updated_person else 0
    new_choice_str = f"{person_id}: {new_name} ({sample_count} slika)"
    
    table_view, stats_view = refresh_database_view()
    choices = get_person_dropdown_choices()
    _, info_text, _ = view_person_details(new_choice_str)
    avatar_html = render_person_avatar_html(person_id)
    btn_text = f"💾 Spremi odabrano lice za: {new_name}"
    
    success_msg = f"✅ **Uspješno spremljeno:** Podaci za osobu **{new_name}** ({format_role_badge(role_clean)}) su ažurirani!"
    
    return (
        success_msg,
        table_view,
        stats_view,
        gr.update(choices=choices, value=new_choice_str),
        gr.update(choices=choices, value=new_choice_str),
        new_name,
        (new_notes or "").strip(),
        info_text,
        avatar_html,
        btn_text,
        role_clean
    )

def delete_selected_sample(selected_sample_str, selected_person_str):
    if not selected_sample_str:
        return "Niste odabrali sliku za brisanje.", [], ""
    try:
        sample_id = int(str(selected_sample_str).replace("Slika #", "").strip())
        db.delete_sample(sample_id)
        msg = f"Uzorak #{sample_id} je obrisan."
    except Exception as e:
        msg = f"Greška pri brisanju: {e}"
        
    gallery, info, choice_upd = view_person_details(selected_person_str)
    return msg, gallery, info

# ---------------- BACKUP & HARDWARE HANDLERS ----------------
def handle_export_backup():
    try:
        zip_path = backup.export_database_zip(DATA_DIR)
        filename = os.path.basename(zip_path)
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        size_str = f"{size_mb / 1024:.2f} GB" if size_mb >= 1024 else f"{size_mb:.1f} MB"
        abs_path = os.path.abspath(zip_path)
        msg = (
            f"✅ **Sigurnosna kopija uspješno generirana!**\n\n"
            f"* **Datoteka:** `{filename}` ({size_str})\n"
            f"* **Lokalna putanja na disku:** `{abs_path}`\n\n"
            f"💡 *Savjet:* Datoteka je već sigurno spremljena na Vašem računalu! Kliknite na gumb **'📂 Otvori mapu sa sigurnosnim kopijama'** ispod kako biste je odmah otvorili u Windows Exploreru i premjestili ili kopirali."
        )
        return zip_path, msg
    except Exception as e:
        return None, f"❌ **Greška pri izvozu:** {e}"

def handle_open_backup_folder():
    backup_dir = os.path.abspath(os.path.join(DATA_DIR, "backups"))
    os.makedirs(backup_dir, exist_ok=True)
    if os.name == "nt":
        os.startfile(backup_dir)
        return f"📂 **Otvorena mapa sigurnosnih kopija u Windows Exploreru:** `{backup_dir}`"
    return f"📁 **Lokacija mapa sigurnosnih kopija:** `{backup_dir}`"

def handle_import_backup(file_obj):
    if not file_obj:
        return "⚠️ Niste odabrali ZIP datoteku za uvoz.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    success, msg = backup.import_database_zip(file_obj, DATA_DIR)
    table_view, stats_view = refresh_database_view()
    choices = get_person_dropdown_choices()
    sys_report = hardware.get_system_report_markdown(DATA_DIR)
    return msg, table_view, stats_view, gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), sys_report

def handle_refresh_sysinfo():
    return hardware.get_hardware_acceleration_badge_html(), hardware.get_system_report_markdown(DATA_DIR)

# ---------------- DANGER ZONE HANDLERS ----------------
def handle_open_wipe_modal():
    stats = db.get_stats()
    num_p = stats.get("total_persons", 0)
    num_s = stats.get("total_samples", 0)
    stats_md = (
        f"📊 **Trenutno evidentirano u bazi:** **{num_p}** registriranih osoba i **{num_s}** biometrijskih uzoraka lica.\n\n"
        f"⚠️ Potvrdom ove radnje **svi navedeni profili i biometrijski vektori bit će trajno uklonjeni**."
    )
    return gr.update(visible=True), stats_md, False, ""

def handle_close_wipe_modal():
    return gr.update(visible=False), False, ""

def handle_execute_wipe(confirmed: bool):
    if not confirmed:
        return (
            gr.update(visible=True),
            "⚠️ **Morate označiti potvrdni okvir** kako biste omogućili brisanje cjelokupne baze!",
            "",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )
    
    try:
        backup_zip = backup.export_database_zip(DATA_DIR)
        bak_name = os.path.basename(backup_zip)
    except Exception as e:
        bak_name = f"Neuspjeh backupa ({e})"
        
    deleted_p, deleted_s = db.clear_all_persons(delete_crops=True)
    face_engine.get_face_index(force_refresh=True)
    
    choices = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    sys_report = hardware.get_system_report_markdown(DATA_DIR)
    empty_avatar = render_person_avatar_html(None)
    
    success_msg = (
        f"✅ **Cjelokupna baza osoba je uspješno obrisana i resetirana!**\n\n"
        f"* Obrisano profila osoba: **{deleted_p}**\n"
        f"* Obrisano biometrijskih uzoraka i izreza lica: **{deleted_s}**\n"
        f"* 🛡️ Sigurnosna kopija prije brisanja automatski je spremljena: `{bak_name}` u mapi `data/backups/`."
    )
    
    return (
        gr.update(visible=False),
        "",
        success_msg,
        table_view,
        stats_view,
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        [],
        "",
        "",
        "",
        empty_avatar,
        sys_report
    )

# ---------------- BATCH DELETE (VIŠESTRUKI ODABIR I BRISANJE) ----------------
def handle_open_batch_delete_modal(selected_items):
    if not selected_items:
        return gr.update(visible=False), "", "⚠️ Niste označili niti jednu osobu za brisanje!", gr.update()
    
    count = len(selected_items)
    names = []
    for item in selected_items:
        name_part = item.split(":")[1].split("(")[0].strip() if ":" in item else item
        names.append(f"• **{name_part}**")
        
    if len(names) > 25:
        names_preview = "\n".join(names[:25]) + f"\n\n*... i još {len(names) - 25} drugih osoba.*"
    else:
        names_preview = "\n".join(names)
        
    summary_md = (
        f"📋 **Označeno za brisanje:** Ukupno **{count}** osoba:\n\n"
        f"{names_preview}\n\n"
        f"⚠️ Potvrdom će sve označene osobe i njihove fotografije biti trajno uklonjene iz sustava."
    )
    btn_label = f"🔥 Potvrdi i obriši ({count} osoba)"
    return gr.update(visible=True), summary_md, "", gr.update(value=btn_label)

def handle_close_batch_delete_modal():
    return gr.update(visible=False)

def handle_execute_batch_delete(selected_items):
    if not selected_items:
        return (
            gr.update(visible=False),
            "⚠️ Nema označenih osoba.",
            [],
            "🗑️ Obriši označene osobe (0)",
            gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(), gr.update(),
            gr.update(), gr.update()
        )
        
    pids = []
    for item in selected_items:
        pid = parse_person_id(item)
        if pid is not None:
            pids.append(pid)
            
    deleted_count = 0
    for pid in pids:
        try:
            db.delete_person(pid)
            deleted_count += 1
        except Exception:
            pass
            
    face_engine.get_face_index(force_refresh=True)
    
    choices = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    sys_report = hardware.get_system_report_markdown(DATA_DIR)
    empty_avatar = render_person_avatar_html(None)
    
    success_msg = f"🗑️ **Uspješno obrisano {deleted_count} osoba** i svi njihovi biometrijski uzorci!"
    
    return (
        gr.update(visible=False),
        success_msg,
        gr.update(choices=choices, value=[]),
        "🗑️ Obriši označene osobe (0)",
        table_view,
        stats_view,
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        [],
        "",
        "",
        "",
        empty_avatar,
        sys_report
    )

def handle_select_all_filtered(search_query, current_selected):
    all_persons = db.get_all_persons()
    q = (search_query or "").strip().lower()
    if q:
        filtered = [
            p for p in all_persons
            if q in p["name"].lower() or q == str(p["id"]) or q in (p["notes"] or "").lower()
        ]
    else:
        filtered = all_persons
        
    selected_set = set(current_selected or [])
    for p in filtered:
        item = f"{p['id']}: {p['name']} ({p['sample_count']} slika)"
        selected_set.add(item)
        
    res = list(selected_set)
    btn_label = f"🗑️ Obriši označene osobe ({len(res)})"
    status = f"☑️ Označeno **{len(filtered)}** osoba iz rezultata pretrage (Ukupno označeno: **{len(res)}**)."
    return res, btn_label, status

def handle_clear_batch_selection():
    return [], "🗑️ Obriši označene osobe (0)", "Odabir je poništen."

def on_batch_selection_change(selected_items):
    n = len(selected_items or [])
    return f"🗑️ Obriši označene osobe ({n})"

def _parse_retention_days(label_str: str) -> int:
    if not label_str or "Trajno" in label_str:
        return 0
    import re
    m = re.search(r"(\d+)", label_str)
    return int(m.group(1)) if m else 30

def handle_retention_period_change(val: str) -> str:
    days = _parse_retention_days(val)
    config.set_retention_days(days)
    if days == 0:
        return "ℹ️ Politika zadržavanja je postavljena na **trajno čuvanje**. Automatsko brisanje starih podataka je isključeno."
    return f"⚙️ Politika zadržavanja ažurirana: podaci stariji od **{days} dana** bit će automatski uklonjeni."

def handle_run_retention_cleanup(val: str) -> tuple:
    days = _parse_retention_days(val)
    if days == 0:
        return "⚠️ Odaberite period zadržavanja (15, 30, 60 ili 90 dana) za pokretanje čišćenja.", hardware.get_system_report_markdown(DATA_DIR)
    res = config.execute_gdpr_retention(days)
    sys_report = hardware.get_system_report_markdown(DATA_DIR)
    return res["status_message"], sys_report

# ---------------- TELEGRAM NOTIFICATION HANDLERS ----------------
def handle_save_telegram_config(enabled, token, chat_id, notify_bl, notify_vip, notify_spoof, notify_unk, cooldown, chat_id_sec="", chat_id_vip=""):
    try:
        config.save_telegram_config(
            enabled=enabled,
            bot_token=token,
            chat_id=chat_id,
            notify_blacklist=notify_bl,
            notify_vip=notify_vip,
            notify_spoof=notify_spoof,
            notify_unknown=notify_unk,
            cooldown_min=cooldown,
            chat_id_security=chat_id_sec,
            chat_id_vip=chat_id_vip
        )
        status_txt = "✅ **Telegram postavke su uspješno spremljene!**"
        if enabled:
            if not str(token or "").strip() or not str(chat_id or "").strip():
                status_txt += "\n⚠️ *Napomena:* Obavijesti su omogućene, ali Bot Token ili Chat ID nisu uneseni. Unesite ih i pošaljite testnu poruku."
            else:
                status_txt += "\n🚀 Obavijesti su aktivne! Preporučujemo da kliknete **'📨 Pošalji testnu poruku'** za provjeru veze."
        else:
            status_txt += "\nℹ️ Telegram modul je trenutno isključen."
        return status_txt
    except Exception as e:
        return f"❌ Greška pri spremanju Telegram postavki: {e}"

def handle_test_telegram(token, chat_id):
    try:
        token_str = str(token or "").strip()
        chat_str = str(chat_id or "").strip()
        if not token_str or not chat_str:
            return "⚠️ **Upozorenje:** Molimo unesite valjani Telegram Bot Token i Chat ID prije pokretanja testa."
        ok, msg = notifier.test_telegram_connection(bot_token=token_str, chat_id=chat_str)
        if ok:
            return "✅ **Testna poruka je uspješno poslana!** Provjerite svoj Telegram račun ili grupu."
        return f"⚠️ **Slanje nije uspjelo:** {msg}"
    except Exception as e:
        return f"❌ Greška pri testiranju Telegrama: {e}"

# ---------------- EMAIL NOTIFICATION HANDLERS ----------------
def handle_save_email_config(enabled, host, port, enc_mode, sender, password, recipients, notify_bl, notify_spoof, notify_vip, cooldown):
    try:
        use_tls = "STARTTLS" in str(enc_mode)
        use_ssl = "SSL" in str(enc_mode)
        config.save_email_config(
            enabled=enabled,
            smtp_host=str(host or "").strip(),
            smtp_port=int(port or 587),
            use_tls=use_tls,
            use_ssl=use_ssl,
            sender=str(sender or "").strip(),
            password=str(password or "").strip(),
            recipients=str(recipients or "").strip(),
            notify_blacklist=bool(notify_bl),
            notify_spoof=bool(notify_spoof),
            notify_vip=bool(notify_vip),
            cooldown_min=int(cooldown or 10)
        )
        msg = "✅ **E-mail postavke su uspješno spremljene!**"
        if enabled:
            if not str(host or "").strip() or not str(sender or "").strip() or not str(recipients or "").strip():
                msg += "\n⚠️ *Napomena:* Obavijesti su uključene, ali SMTP poslužitelj, pošiljatelj ili primatelji nisu u potpunosti uneseni."
            else:
                msg += "\n📧 Sustav je spreman. Kliknite **'📨 Pošalji testni E-mail'** za provjeru ispravnosti veze."
        else:
            msg += "\nℹ️ E-mail modul je trenutno isključen."
        return msg
    except Exception as e:
        return f"❌ Greška pri spremanju E-mail postavki: {e}"

def handle_test_email(host, port, enc_mode, sender, password, recipients):
    try:
        use_tls = "STARTTLS" in str(enc_mode)
        use_ssl = "SSL" in str(enc_mode)
        h = str(host or "").strip()
        s = str(sender or "").strip()
        r = str(recipients or "").strip()
        if not h or not s or not r:
            return "⚠️ **Upozorenje:** Unesite SMTP poslužitelj, pošiljatelja i barem jednog primatelja prije testiranja."
        ok, msg = notifier.test_email_connection(
            smtp_host=h,
            smtp_port=int(port or 587),
            use_tls=use_tls,
            use_ssl=use_ssl,
            sender=s,
            password=str(password or "").strip(),
            recipients=r
        )
        if ok:
            return f"✅ **Testni E-mail je uspješno poslan!** Provjerite ulaznu poštu ({r})."
        return f"⚠️ **Slanje nije uspjelo:** {msg}"
    except Exception as e:
        return f"❌ Greška pri slanju testnog e-maila: {e}"

def handle_launch_live(source_type="USB Web Kamera", usb_idx="0", rtsp_url="", youtube_url="", video_file=None, start_sec=0, log_events=False, cooldown_sec=30, record_nvr=False, segment_min=5, anti_spoof=True):
    live_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_cam.py")
    if not os.path.exists(live_script):
        return "⚠️ Skripta `live_cam.py` nije pronađena."
    try:
        import subprocess
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            
        candidates = [
            os.path.join(APP_DIR, ".venv", "Scripts", "pythonw.exe"),
            os.path.join(os.path.dirname(APP_DIR), ".venv", "Scripts", "pythonw.exe"),
            os.path.join(APP_DIR, "python", "pythonw.exe"),
            os.path.join(sys.prefix, "Scripts", "pythonw.exe"),
            os.path.join(APP_DIR, ".venv", "Scripts", "python.exe"),
            os.path.join(os.path.dirname(APP_DIR), ".venv", "Scripts", "python.exe"),
            os.path.join(APP_DIR, "python", "python.exe"),
            sys.executable
        ]
        py_exe = sys.executable
        for cand in candidates:
            if os.path.isfile(cand):
                py_exe = cand
                break
                
        env = os.environ.copy()
        venv_scripts = os.path.dirname(py_exe)
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = os.path.dirname(venv_scripts)

        if source_type == "USB Web Kamera":
            source_arg = str(usb_idx).strip() or "0"
            target_name = f"USB kamera (indeks {source_arg})"
        elif source_type == "IP / RTSP Kamera za nadzor":
            source_arg = str(rtsp_url).strip()
            target_name = f"IP Nadzor ({source_arg})"
        elif source_type == "YouTube / Web Video":
            source_arg = str(youtube_url).strip()
            target_name = f"YouTube Video ({source_arg})"
        elif source_type == "Lokalna Video Datoteka":
            if video_file is None:
                return "⚠️ Molimo odaberite video datoteku s računala."
            source_arg = getattr(video_file, "name", str(video_file))
            target_name = f"Datoteka: {os.path.basename(source_arg)}"
        else:
            source_arg = "0"
            target_name = "Kamera"

        if not source_arg:
            return "⚠️ Molimo unesite valjanu adresu ili odaberite izvor."

        cmd = [py_exe, live_script, "--source", source_arg]
        if start_sec and int(start_sec) > 0 and source_type in ("YouTube / Web Video", "Lokalna Video Datoteka"):
            cmd.extend(["--start", str(int(start_sec))])
            target_name += f" (od {int(start_sec) // 60:02d}:{int(start_sec) % 60:02d})"

        if log_events:
            cmd.extend(["--log-events", "--cooldown", str(int(cooldown_sec))])
            target_name += f" [📋 Dnevnik: {int(cooldown_sec)}s]"

        if record_nvr:
            cmd.extend(["--record-nvr", "--segment-min", str(int(segment_min))])
            target_name += f" [🔴 NVR: {int(segment_min)}m]"

        if anti_spoof:
            cmd.append("--anti-spoof")
            target_name += " [🛡️ Anti-Spoof: ON]"
        else:
            cmd.append("--no-anti-spoof")
            target_name += " [⚠️ Anti-Spoof: OFF]"

        subprocess.Popen(cmd, cwd=APP_DIR, env=env, creationflags=creationflags)
        
        return f"🎥 **Live prepoznavanje [{target_name}] je uspješno pokrenuto u novom prozoru!**\n*(Pritisnite tipku `F` za Anti-Spoof, `R` za NVR, `S` za kadar, `O` za mapu, `Q` za izlaz)*"
    except Exception as e:
        return f"❌ Greška pri pokretanju: {e}"

def handle_launch_multicam(
    c1_on, c1_name, c1_src,
    c2_on, c2_name, c2_src,
    c3_on, c3_name, c3_src,
    c4_on, c4_name, c4_src,
    threshold=0.45,
    log_events=False,
    cooldown_sec=30,
    record_nvr=False,
    segment_min=5
):
    try:
        import subprocess
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        
        candidates = [
            os.path.join(APP_DIR, ".venv", "Scripts", "pythonw.exe"),
            os.path.join(os.path.dirname(APP_DIR), ".venv", "Scripts", "pythonw.exe"),
            os.path.join(APP_DIR, "python", "pythonw.exe"),
            os.path.join(sys.prefix, "Scripts", "pythonw.exe"),
            os.path.join(APP_DIR, ".venv", "Scripts", "python.exe"),
            os.path.join(os.path.dirname(APP_DIR), ".venv", "Scripts", "python.exe"),
            os.path.join(APP_DIR, "python", "python.exe"),
            sys.executable
        ]
        py_exe = sys.executable
        for cand in candidates:
            if os.path.isfile(cand):
                py_exe = cand
                break
                
        env = os.environ.copy()
        venv_scripts = os.path.dirname(py_exe)
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
        env["VIRTUAL_ENV"] = os.path.dirname(venv_scripts)

        script_path = os.path.join(APP_DIR, "run_multicam.py")
        if not os.path.isfile(script_path):
            script_path = os.path.join(os.path.dirname(APP_DIR), "run_multicam.py")

        cmd = [py_exe, script_path, "--threshold", str(float(threshold))]
        
        if c1_on and str(c1_src).strip():
            cmd.extend(["--cam1", str(c1_src).strip(), "--name1", str(c1_name).strip() or "Kamera 1"])
        else:
            cmd.append("--off1")
            
        if c2_on and str(c2_src).strip():
            cmd.extend(["--cam2", str(c2_src).strip(), "--name2", str(c2_name).strip() or "Kamera 2"])
        else:
            cmd.append("--off2")

        if c3_on and str(c3_src).strip():
            cmd.extend(["--cam3", str(c3_src).strip(), "--name3", str(c3_name).strip() or "Kamera 3"])
        else:
            cmd.append("--off3")

        if c4_on and str(c4_src).strip():
            cmd.extend(["--cam4", str(c4_src).strip(), "--name4", str(c4_name).strip() or "Kamera 4"])
        else:
            cmd.append("--off4")

        if log_events:
            cmd.extend(["--log-events", "--cooldown", str(int(cooldown_sec))])

        if record_nvr:
            cmd.extend(["--record-nvr", "--segment-min", str(int(segment_min))])

        active_count = sum([
            bool(c1_on and str(c1_src).strip()),
            bool(c2_on and str(c2_src).strip()),
            bool(c3_on and str(c3_src).strip()),
            bool(c4_on and str(c4_src).strip())
        ])
        if active_count == 0:
            return "⚠️ Morate omogućiti barem jednu kameru za 2×2 mrežu."

        subprocess.Popen(cmd, cwd=APP_DIR, env=env, creationflags=creationflags)
        nvr_tag = f" uz 24/7 NVR snimanje ({int(segment_min)}m segmenti)" if record_nvr else ""
        return f"🎛️ **Multi-Camera 2×2 mreža ({active_count} kamere) je uspješno pokrenuta u novom prozoru{nvr_tag}!**\n*(Pritisnite tipke `1`-`4` za Solo prikaz, `0` ili `ESC` za mrežu, `R` za NVR snimanje, `E` za evidenciju, `S` za kadar, `Q` za izlaz)*"
    except Exception as e:
        return f"❌ Greška pri pokretanju 2×2 mreže: {e}"

# ---------------- SNAPSHOTS & ARCHIVE HELPERS ----------------
def get_snapshots_ui_data():
    snaps = config.get_saved_snapshots()
    gallery_items = []
    table_rows = []
    choices = []
    for s in snaps:
        gallery_items.append((s["path"], f"{s['filename']} ({s['size_kb']} KB)"))
        table_rows.append([s["filename"], s["time_str"], f"{s['size_kb']} KB"])
        choices.append(s["filename"])
        
    total_mb = sum(s["size_kb"] for s in snaps) / 1024.0
    snap_dir = config.get_snapshot_dir()
    info_md = f"📁 **Mapa za spremanje snimaka:** `{snap_dir}` &nbsp;|&nbsp; 📸 **Ukupno snimki:** `{len(snaps)}` &nbsp;|&nbsp; 💾 **Zauzeće:** `{total_mb:.2f} MB`"
    
    first_choice = choices[0] if choices else None
    first_preview = snaps[0]["path"] if snaps else None
    first_desc = ""
    if snaps:
        s0 = snaps[0]
        first_desc = f"📸 **Datoteka:** `{s0['filename']}`\n\n🕒 **Vrijeme snimanja:** {s0['time_str']} &nbsp;|&nbsp; 💾 **Veličina:** {s0['size_kb']} KB\n\n📍 **Puna putanja:** `{s0['path']}`"
        
    return gallery_items, table_rows, info_md, gr.update(choices=choices, value=first_choice), first_preview, first_desc

def on_snapshot_gallery_select(evt: gr.SelectData):
    snaps = config.get_saved_snapshots()
    idx = evt.index
    if 0 <= idx < len(snaps):
        s = snaps[idx]
        desc = f"📸 **Datoteka:** `{s['filename']}`\n\n🕒 **Vrijeme snimanja:** {s['time_str']} &nbsp;|&nbsp; 💾 **Veličina:** {s['size_kb']} KB\n\n📍 **Puna putanja:** `{s['path']}`"
        return s["path"], desc, gr.update(value=s["filename"])
    return None, "", gr.update()

def on_snapshot_dropdown_change(filename):
    if not filename:
        return None, ""
    snap_dir = config.get_snapshot_dir()
    path = os.path.join(snap_dir, filename)
    if os.path.isfile(path):
        import time
        stat = os.stat(path)
        t_str = time.strftime("%d.%m.%Y. %H:%M:%S", time.localtime(stat.st_mtime))
        size_kb = round(stat.st_size / 1024, 1)
        desc = f"📸 **Datoteka:** `{filename}`\n\n🕒 **Vrijeme snimanja:** {t_str} &nbsp;|&nbsp; 💾 **Veličina:** {size_kb} KB\n\n📍 **Puna putanja:** `{path}`"
        return path, desc
    return None, "Datoteka nije pronađena."

def on_delete_snapshot_click(filename):
    if not filename:
        g, t, info, dd, prev, desc = get_snapshots_ui_data()
        return "⚠️ Odaberite snimku za brisanje.", g, t, info, dd, prev, desc
    snap_dir = config.get_snapshot_dir()
    path = os.path.join(snap_dir, filename)
    if os.path.isfile(path):
        try:
            os.remove(path)
            msg = f"🗑️ Snimka `{filename}` je uspješno obrisana."
        except Exception as e:
            msg = f"❌ Greška pri brisanju: {e}"
    else:
        msg = "⚠️ Datoteka više ne postoji na disku."
    g, t, info, dd, prev, desc = get_snapshots_ui_data()
    return msg, g, t, info, dd, prev, desc

def on_save_snapshot_dir_click(new_dir):
    ok, msg = config.set_snapshot_dir(new_dir)
    g, t, info, dd, prev, desc = get_snapshots_ui_data()
    return msg, info, g, t, dd, prev, desc

def on_reset_snapshot_dir_click():
    ok, msg = config.set_snapshot_dir(config.DEFAULT_SNAPSHOT_DIR)
    g, t, info, dd, prev, desc = get_snapshots_ui_data()
    return f"Vraćeno na zadanu mapu: `{config.DEFAULT_SNAPSHOT_DIR}`", config.DEFAULT_SNAPSHOT_DIR, info, g, t, dd, prev, desc

def on_send_snapshot_to_recognition(filename, threshold, landmarks, blur_mode, show_all_faces=False):
    if not filename:
        return None, None, [], None, "⚠️ Nema odabrane snimke za analizu.", gr.update(choices=[]), [], generate_detection_cards_html([], show_all_faces=show_all_faces), "⚠️ Nema odabrane snimke."
    snap_dir = config.get_snapshot_dir()
    path = os.path.join(snap_dir, filename)
    if not os.path.isfile(path):
        return None, None, [], None, "⚠️ Datoteka nije pronađena.", gr.update(choices=[]), [], generate_detection_cards_html([], show_all_faces=show_all_faces), "⚠️ Datoteka nije pronađena."
    
    blur_unknown = blur_mode == "🔒 Zamuti nepoznata lica"
    blur_all     = blur_mode == "🛡️ Zamuti sva lica (GDPR)"
    pil_img = Image.open(path).convert("RGB")
    annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state, cards_html = recognize_faces(
        pil_img, threshold, landmarks, blur_unknown, blur_all=blur_all, show_all_faces=show_all_faces
    )
    msg = f"✅ Kadar `{filename}` je prebačen u Tab 1 i analiziran!"
    return pil_img, annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state, cards_html, msg

# ---------------- DETECTION EVENT LOG HELPERS ----------------
def get_events_ui_data(search_query=""):
    events = db.get_detection_events(limit=300, name_filter=search_query)
    stats = db.get_detection_stats()
    
    table_rows = []
    
    for ev in events:
        sim_pct = f"{ev['similarity']*100:.1f}%"
        table_rows.append([
            ev["id"],
            ev["local_time"],
            ev["person_name"],
            sim_pct,
            ev["source_label"]
        ])
            
    stats_md = (
        f"📊 **Ukupno prolazaka:** `{stats['total_events']}` &nbsp;|&nbsp; "
        f"👥 **Jedinstvenih osoba:** `{stats['unique_persons']}` &nbsp;|&nbsp; "
        f"⏱️ **Zadnji zabilježeni prolazak:** `{stats['latest_event']}`"
    )
    first_cam = None
    first_crop = None
    first_info = "💡 *Kliknite na redak u tablici za pregled kadra kamere i detalja.*"
    first_video = None
    first_id = None
    if events:
        first_ev = events[0]
        first_id = first_ev["id"]
        c_p = first_ev.get("snapshot_path", "")
        cr_p = first_ev.get("crop_path", "")
        first_cam = c_p if (c_p and os.path.exists(c_p)) else (cr_p if (cr_p and os.path.exists(cr_p)) else None)
        first_crop = cr_p if (cr_p and os.path.exists(cr_p)) else None
        cam_desc = "Kadar s kamere (nadzor)" if (c_p and os.path.exists(c_p)) else "Izrezano lice"
        
        v_p = first_ev.get("video_path", "")
        v_off = first_ev.get("video_offset_sec", 0.0)
        v_tag = ""
        if v_p and os.path.exists(v_p):
            first_video = v_p
            v_tag = f"\n* **📹 NVR Video snimka:** `{os.path.basename(v_p)}` (Detekcija na: **`{v_off:.1f}s`**)"

        first_info = (
            f"### 📋 Detalji odabranog prolaska #{first_id}\n"
            f"* **Prepoznata osoba:** **`{first_ev['person_name']}`**\n"
            f"* **Vrijeme prolaska:** `{first_ev['local_time']}`\n"
            f"* **Pouzdanost / Sličnost:** **`{first_ev['similarity']*100:.1f}%`**\n"
            f"* **Izvor / Kamera:** `{first_ev['source_label']}`\n"
            f"* **Prikaz slike:** {cam_desc}"
            f"{v_tag}"
        )
        
    btn_del_update = gr.update(value=f"🗑️ Obriši ovaj prolazak (#{first_id})" if first_id else "🗑️ Obriši odabrani prolazak", visible=bool(first_id))
    return stats_md, table_rows, first_cam, first_crop, first_info, gr.update(value=first_video, visible=bool(first_video)), first_id, btn_del_update

def on_events_search(search_query=""):
    return get_events_ui_data(search_query)

def on_event_select(evt: gr.SelectData, search_query=""):
    events = db.get_detection_events(limit=300, name_filter=search_query)
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if 0 <= row_idx < len(events):
        ev = events[row_idx]
        ev_id = ev["id"]
        cam_p = ev.get("snapshot_path", "")
        crop_p = ev.get("crop_path", "")
        main_img = cam_p if (cam_p and os.path.exists(cam_p)) else (crop_p if (crop_p and os.path.exists(crop_p)) else None)
        crop_img = crop_p if (crop_p and os.path.exists(crop_p)) else None
        sim_pct = f"{ev['similarity']*100:.1f}%"
        cam_desc = "✅ Puni kadar kamere" if (cam_p and os.path.exists(cam_p)) else "ℹ️ Prikaz izrezanog lica"
        
        v_p = ev.get("video_path", "")
        v_off = ev.get("video_offset_sec", 0.0)
        v_tag = ""
        has_video = bool(v_p and os.path.exists(v_p))
        if has_video:
            v_tag = f"\n* **📹 NVR Video snimka:** `{os.path.basename(v_p)}` (Detekcija na: **`{v_off:.1f}s`**)"

        info = (
            f"### 📋 Detalji prolaska #{ev_id}\n"
            f"* **Prepoznata osoba:** **`{ev['person_name']}`**\n"
            f"* **Vrijeme prolaska:** `{ev['local_time']}`\n"
            f"* **Pouzdanost / Sličnost:** **`{sim_pct}`**\n"
            f"* **Izvor / Kamera:** `{ev['source_label']}`\n"
            f"* **Prikaz slike:** {cam_desc}"
            f"{v_tag}"
        )
        return main_img, crop_img, info, gr.update(value=v_p if has_video else None, visible=has_video), ev_id, gr.update(value=f"🗑️ Obriši ovaj prolazak (#{ev_id})", visible=True)
    return None, None, "Događaj nije pronađen.", gr.update(value=None, visible=False), None, gr.update(visible=False)

def handle_open_clear_events_modal():
    stats = db.get_detection_stats()
    num_e = stats.get("total_events", 0)
    num_p = stats.get("unique_persons", 0)
    stats_md = (
        f"📊 **Trenutno u evidenciji:** **{num_e}** zabilježenih prolazaka ({num_p} različitih osoba).\n\n"
        f"⚠️ Potvrdom ove radnje **svi zapisi prolazaka i povezane fotografije bit će trajno obrisani**."
    )
    return gr.update(visible=True), stats_md, False, ""

def handle_close_clear_events_modal():
    return gr.update(visible=False), False, ""

def handle_execute_clear_events(confirmed: bool, search_query=""):
    if not confirmed:
        return (
            gr.update(visible=True),
            "⚠️ **Morate označiti potvrdni okvir** kako biste omogućili brisanje cjelokupnog dnevnika!",
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    db.clear_detection_events()
    s_md, t_rows, cam_p, cr_p, inf, vid_up, f_id, b_del = get_events_ui_data(search_query)
    return gr.update(visible=False), "", s_md, t_rows, cam_p, cr_p, inf, vid_up, f_id, b_del

def handle_open_single_delete_event_modal(selected_id):
    if not selected_id:
        return gr.update(visible=False), ""
    ev = db.get_detection_event_by_id(selected_id)
    if not ev:
        return gr.update(visible=False), ""
    summary_md = (
        f"### 📋 Podaci o prolasku koji će biti obrisan:\n"
        f"* **ID prolaska:** `#{ev['id']}`\n"
        f"* **Prepoznata osoba:** **`{ev['person_name']}`**\n"
        f"* **Vrijeme prolaska:** `{ev['local_time']}`\n"
        f"* **Izvor / Kamera:** `{ev['source_label']}`\n"
        f"* **Sličnost:** `{ev['similarity']*100:.1f}%`\n\n"
        f"⚠️ Ova radnja nepovratno briše ovaj pojedinačni zapis i povezanu sliku detekcije."
    )
    return gr.update(visible=True), summary_md

def handle_close_single_delete_event_modal():
    return gr.update(visible=False), ""

def handle_execute_single_delete_event(selected_id, search_query=""):
    if selected_id:
        db.delete_detection_event(selected_id)
    s_md, t_rows, cam_p, cr_p, inf, vid_up, f_id, b_del = get_events_ui_data(search_query)
    return gr.update(visible=False), s_md, t_rows, cam_p, cr_p, inf, vid_up, f_id, b_del

# ---------------- NVR ARCHIVE HELPERS ----------------
def get_nvr_archive_ui_data(date_filter=""):
    from nvr_recorder import get_nvr_manager
    nvr = get_nvr_manager()
    recs = nvr.list_recordings(date_filter)
    rows = []
    total_mb = 0.0
    for r in recs:
        total_mb += r["size_mb"]
        rows.append([
            r["filename"],
            r["camera"],
            r["day"],
            r["time_str"],
            f"{r['size_mb']:.1f} MB",
            r["path"]
        ])
    total_gb = total_mb / 1024.0
    status_md = (
        f"💾 **Zauzeće video arhive:** `{total_gb:.2f} GB` / `{nvr.max_storage_gb:.1f} GB` (FIFO rotacija) &nbsp;|&nbsp; "
        f"📼 **Ukupno video segmenata:** `{len(recs)}` &nbsp;|&nbsp; "
        f"📁 **Putanja:** `{nvr.recordings_dir}`"
    )
    first_video = recs[0]["path"] if recs else None
    first_label = f"▶️ **Odabrana snimka:** `{recs[0]['filename']}`" if recs else "💡 *Nema zabilježenih snimki.*"
    return status_md, rows, first_video, first_label

def on_nvr_segment_select(evt: gr.SelectData, date_filter=""):
    from nvr_recorder import get_nvr_manager
    nvr = get_nvr_manager()
    recs = nvr.list_recordings(date_filter)
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if 0 <= row_idx < len(recs):
        item = recs[row_idx]
        info_txt = f"▶️ **Odabrana snimka:** `{item['filename']}` ({item['camera']} | {item['time_str']} | {item['size_mb']} MB)"
        return item["path"], info_txt
    return None, ""

def handle_open_nvr_folder():
    from nvr_recorder import DEFAULT_RECORDINGS_DIR
    config.open_folder_in_explorer(DEFAULT_RECORDINGS_DIR)
    return "📂 Otvorena mapa sa snimkama u Exploreru."

def handle_open_external_video(video_path):
    if not video_path or not os.path.isfile(video_path):
        return "⚠️ Nema odabrane video snimke."
    try:
        import subprocess
        if hasattr(os, "startfile"):
            os.startfile(video_path)
            return f"🎬 Pokrenut vanjski player za: `{os.path.basename(video_path)}`"
        else:
            subprocess.Popen(["explorer", video_path])
            return f"🎬 Pokrenut vanjski player za: `{os.path.basename(video_path)}`"
    except Exception as e:
        return f"❌ Greška pri pokretanju playera: {e}"

def handle_delete_nvr_segment(video_path):
    if not video_path or not os.path.isfile(video_path):
        status_md, rows, first_vid, first_lbl = get_nvr_archive_ui_data()
        return "⚠️ Datoteka nije pronađena.", status_md, rows, first_vid, first_lbl
    try:
        os.remove(video_path)
        msg = f"🗑️ Video snimka `{os.path.basename(video_path)}` je uspješno obrisana."
    except Exception as e:
        msg = f"❌ Greška pri brisanju: {e}"
    status_md, rows, first_vid, first_lbl = get_nvr_archive_ui_data()
    return msg, status_md, rows, first_vid, first_lbl

def handle_export_events_csv():
    events = db.get_detection_events(limit=5000)
    if not events:
        return None, "⚠️ Nema zabilježenih događaja za izvoz."
    import csv
    export_path = os.path.join(DATA_DIR, "dnevnik_prolazaka_export.csv")
    with open(export_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Vrijeme", "Ime Osobe", "Sličnost (%)", "Izvor / Kamera", "Putanja do lica", "Putanja do kadra kamere"])
        for ev in events:
            writer.writerow([
                ev["id"],
                ev["local_time"],
                ev["person_name"],
                f"{ev['similarity']*100:.1f}%",
                ev["source_label"],
                ev.get("crop_path", ""),
                ev.get("snapshot_path", "")
            ])
    return gr.update(value=export_path, visible=True), f"✅ Uspješno izvezeno {len(events)} prolazaka u CSV!"

# ---------------- PAMETNI SORTER FOTOGRAFIJA HELPERS ----------------
active_sorter_instance: Optional[photo_sorter.PhotoSorter] = None

def select_folder_dialog(title="Odaberite mapu", initial_dir=None) -> str:
    """Otvara nativni Windows dijalog za grafički odabir mape."""
    init_d = str(initial_dir or "").strip().strip("'\"")
    if not os.path.isdir(init_d):
        init_d = ""

    # 1. Pokušaj preko Tkinter (ugrađen i brz)
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.focus_force()
        selected = filedialog.askdirectory(title=title, initialdir=init_d or None)
        root.destroy()
        if selected:
            return os.path.normpath(selected)
    except Exception:
        pass

    # 2. Fallback preko PowerShell FolderBrowserDialog
    try:
        import subprocess
        init_arg = f"$d.SelectedPath = '{init_d}';" if init_d else ""
        ps_code = f"""
        Add-Type -AssemblyName System.Windows.Forms
        $d = New-Object System.Windows.Forms.FolderBrowserDialog
        $d.Description = '{title}'
        $d.ShowNewFolderButton = $true
        {init_arg}
        if ($d.ShowDialog((New-Object System.Windows.Forms.NativeWindow)) -eq [System.Windows.Forms.DialogResult]::OK) {{
            Write-Output $d.SelectedPath
        }}
        """
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_code],
            capture_output=True,
            text=True,
            timeout=120
        )
        out = proc.stdout.strip()
        if out and os.path.isdir(out):
            return os.path.normpath(out)
    except Exception:
        pass

    return ""

def is_dir_writable(path: str) -> bool:
    """Provjerava može li se pisati u zadanu mapu."""
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, f".uniface_perm_test_{os.getpid()}")
        with open(test_file, "w") as f:
            f.write("ok")
        os.remove(test_file)
        return True
    except Exception:
        return False

def get_smart_default_output_dir(input_folder_path: str) -> str:
    """
    Predlaže optimalnu i zajamčeno upisivu mapu za sortirane fotografije.
    1. Pokušava <roditelj>/<ime>_SORTIRANO (na istom disku).
    2. Ako roditelj nema dozvolu pisanja (npr. vanjski disk s restriktivnim NTFS ovlastima):
       pokušava na korijenu istog diska <Drive>:\\UniFace_Sortirano\\<ime>_SORTIRANO (omogućuje Hardlink!).
    3. Ako ni to nije dostupno, nudi korisničku mapu Slike na C:\\.
    """
    clean_in = os.path.abspath(input_folder_path.strip().strip("'\""))
    base_name = os.path.basename(clean_in) or "Fotografije"
    parent_dir = os.path.dirname(clean_in)

    candidate_1 = os.path.join(parent_dir, f"{base_name}_SORTIRANO")
    if is_dir_writable(candidate_1):
        return candidate_1

    drive, _ = os.path.splitdrive(clean_in)
    if drive:
        candidate_2 = os.path.join(drive + os.sep, "UniFace_Sortirano", f"{base_name}_SORTIRANO")
        if is_dir_writable(candidate_2):
            return candidate_2

    user_pictures = os.path.join(os.path.expanduser("~"), "Pictures", "UniFace_Sortirano", f"{base_name}_SORTIRANO")
    if is_dir_writable(user_pictures):
        return user_pictures

    return os.path.join(APP_DIR, "data", "sortirano", f"{base_name}_SORTIRANO")

def get_sorter_person_choices():
    """Vraća listu imena osoba iz baze za odabir u sorteru."""
    try:
        persons = db.get_all_persons()
        return [f"{p['name']} (ID: {p['id']})" for p in persons]
    except Exception:
        return []

def handle_validate_input_folder(input_folder_path: str):
    """Provjerava postojanje ulazne mape i broji podržane fotografije."""
    path = (input_folder_path or "").strip().strip('"\'')
    if not path:
        return "⚠️ Unesite putanju do mape s fotografijama ili kliknite 'Odaberi mapu...'.", ""
    if not os.path.isdir(path):
        return f"❌ Mapa ne postoji ili nije dostupna: `{path}`", ""
    
    files = []
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            if ext in photo_sorter.SUPPORTED_IMAGE_EXTS:
                files.append(os.path.join(root, fn))
    
    count = len(files)
    if count == 0:
        return f"⚠️ U mapi `{path}` nije pronađena niti jedna slika (JPG, PNG, WebP...).", ""
    
    default_out = get_smart_default_output_dir(path)
    return (
        f"✅ **Pronađeno {count} fotografija** spremnih za analizu i sortiranje.\n"
        f"📁 Ulazna lokacija: `{os.path.abspath(path)}`\n"
        f"💾 Predloženo odredište: `{default_out}`",
        default_out
    )

def on_browse_input_folder(current_val):
    chosen = select_folder_dialog("Odaberite mapu s fotografijama", current_val)
    if not chosen:
        return gr.update(), gr.update(), gr.update()
    info_md, def_out = handle_validate_input_folder(chosen)
    return chosen, info_md, def_out

def on_browse_output_folder(current_val):
    chosen = select_folder_dialog("Odaberite odredišnu mapu za sortirane fotografije", current_val)
    if not chosen:
        return gr.update()
    return chosen

def handle_start_photo_sorting(
    input_folder: str,
    output_folder: str,
    target_person_labels: list,
    similarity_thresh: float,
    action_mode: str,
    enable_combo: bool,
    enable_group: bool,
    group_min_faces: float,
    enable_noface: bool,
    enable_unregistered: bool,
    resolution_mode: str
):
    """Pokreće sortiranje i stream-a napredak u Gradio UI."""
    global active_sorter_instance

    in_dir = (input_folder or "").strip().strip('"\'')
    out_dir = (output_folder or "").strip().strip('"\'')

    if not in_dir or not os.path.isdir(in_dir):
        yield (
            "❌ **Greška:** Ulazna mapa ne postoji ili nije dostupna!",
            "",
            None,
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )
        return

    if not out_dir:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(in_dir)), f"{os.path.basename(os.path.abspath(in_dir))}_SORTIRANO")

    target_ids = []
    if target_person_labels:
        for lbl in target_person_labels:
            if "(ID: " in lbl:
                try:
                    pid = int(lbl.split("(ID: ")[-1].rstrip(")"))
                    target_ids.append(pid)
                except Exception:
                    pass

    if "hardlink" in (action_mode or "").lower():
        file_action = "hardlink"
    elif "premjesti" in (action_mode or "").lower() or "move" in (action_mode or "").lower():
        file_action = "move"
    else:
        file_action = "copy"

    if "1280" in (resolution_mode or ""):
        max_dim = 1280
    elif "2048" in (resolution_mode or ""):
        max_dim = 2048
    else:
        max_dim = 1600

    sorter = photo_sorter.PhotoSorter(
        input_dir=in_dir,
        output_dir=out_dir,
        target_person_ids=target_ids if target_ids else None,
        threshold=float(similarity_thresh),
        file_action=file_action,
        group_min_faces=int(group_min_faces or 4),
        enable_group_folder=bool(enable_group),
        enable_combo_folder=bool(enable_combo),
        enable_no_face_folder=bool(enable_noface),
        enable_unregistered_folder=bool(enable_unregistered),
        max_det_dim=max_dim,
        device="AUTO"
    )
    active_sorter_instance = sorter

    prog_q = queue.Queue()

    def progress_cb(data):
        prog_q.put(data)

    worker_res = {}
    def worker():
        try:
            worker_res["result"] = sorter.run(progress_callback=progress_cb)
        except Exception as e:
            worker_res["error"] = str(e)
        finally:
            prog_q.put({"__done__": True})

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    last_status = "🚀 Inicijalizacija biometrijskog modela i indeksiranje uzoraka..."
    last_stats = ""

    while t.is_alive() or not prog_q.empty():
        try:
            item = prog_q.get(timeout=0.25)
        except queue.Empty:
            continue

        if "__done__" in item:
            break

        cur = item.get("current", 0)
        tot = item.get("total", 0)
        pct = item.get("percent", 0.0)
        fps = item.get("fps", 0.0)
        eta = item.get("eta_sec", 0)
        matched_photos = item.get("matched_photos", 0)
        total_faces = item.get("total_faces", 0)
        cur_file = item.get("current_file", "")
        p_stats = item.get("stats_by_person", {})

        eta_str = f"{eta // 60}m {eta % 60}s" if eta >= 60 else f"{eta}s"
        last_status = (
            f"🔄 **Obrada u tijeku: {cur} / {tot} slika ({pct}%)** &nbsp;|&nbsp; "
            f"⚡ **Brzina:** `{fps:.1f} slika/s` &nbsp;|&nbsp; ⏳ **Preostalo:** `{eta_str}`\n\n"
            f"📄 Trenutna fotografija: `{cur_file}`"
        )

        breakdown_lines = [f"* **{pname}:** `{cnt}` fotografija" for pname, cnt in sorted(p_stats.items(), key=lambda x: x[1], reverse=True)]
        breakdown_text = "\n".join(breakdown_lines) if breakdown_lines else "*Čekanje na prva prepoznavanja lica...*"

        last_stats = (
            f"### 📊 Statistika u stvarnom vremenu\n"
            f"* **Ukupno analizirano fotografija:** `{cur} / {tot}`\n"
            f"* **Fotografija s prepoznatim osobama:** `{matched_photos}`\n"
            f"* **Ukupno pronađeno lica:** `{total_faces}`\n\n"
            f"#### 👥 Razvrstano po mapama osoba:\n{breakdown_text}"
        )

        yield (
            last_status,
            last_stats,
            gr.update(visible=False),
            gr.update(interactive=False),
            gr.update(interactive=True),
            gr.update(interactive=False)
        )

    t.join()

    res = worker_res.get("result", {})
    err = worker_res.get("error", None)

    if err:
        yield (
            f"❌ **Došlo je do greške tijekom obrade:** {err}",
            last_stats,
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=True)
        )
        return

    if res.get("status") == "error":
        yield (
            f"⚠️ **Prekid:** {res.get('message', 'Nepoznata greška')}",
            last_stats,
            gr.update(visible=False),
            gr.update(interactive=True),
            gr.update(interactive=False),
            gr.update(interactive=False)
        )
        return

    is_cancelled = (res.get("status") == "cancelled")
    status_icon = "🛑" if is_cancelled else "✅"
    status_title = "Sortiranje je prekinuto od strane korisnika" if is_cancelled else "Sortiranje je uspješno završeno!"

    fin_status = (
        f"{status_icon} **{status_title}**\n\n"
        f"* **Obrađeno fotografija:** `{res.get('processed', 0)}` od `{res.get('total_files', 0)}`\n"
        f"* **Fotografija s prepoznatim osobama:** `{res.get('matched_photos', 0)}`\n"
        f"* **Detektirano lica ukupno:** `{res.get('total_faces', 0)}`\n"
        f"* **Prosječna brzina:** `{res.get('avg_fps', 0.0)} slika/s` (ukupno vrijeme: `{res.get('elapsed_sec', 0)}s`)\n"
        f"* 📂 **Odredišna mapa:** `{out_dir}`"
    )

    p_stats = res.get("stats_by_person", {})
    breakdown_lines = [f"* **{pname}:** `{cnt}` fotografija" for pname, cnt in sorted(p_stats.items(), key=lambda x: x[1], reverse=True)]
    breakdown_text = "\n".join(breakdown_lines) if breakdown_lines else "*Nema prepoznatih osoba s odabranim pragom točnosti.*"

    fin_stats = (
        f"### 🏆 Završni rezultati sortiranja\n"
        f"* **Odredišna mapa:** `{out_dir}`\n"
        f"* **Način prijenosa:** `{file_action.upper()}`\n\n"
        f"#### 👥 Ukupno slika po mapama osoba:\n{breakdown_text}"
    )

    report_f = res.get("report_path")
    csv_local_path = None
    if report_f and os.path.exists(report_f):
        try:
            csv_local_path = os.path.join(DATA_DIR, "zadnji_izvjestaj_sortiranja.csv")
            shutil.copy2(report_f, csv_local_path)
        except Exception:
            csv_local_path = None

    csv_update = gr.update(value=csv_local_path, visible=bool(csv_local_path and os.path.exists(csv_local_path)))

    yield (
        fin_status,
        fin_stats,
        csv_update,
        gr.update(interactive=True),
        gr.update(interactive=False),
        gr.update(interactive=True)
    )

def handle_cancel_photo_sorting():
    global active_sorter_instance
    if active_sorter_instance and active_sorter_instance.is_running:
        active_sorter_instance.cancel()
        return "⏳ Zaustavljanje sortiranja u tijeku... Molimo pričekajte trenutnu datoteku."
    return "ℹ️ Nema aktivnog procesa sortiranja."

def handle_open_sorter_folder(folder_path: str):
    path = (folder_path or "").strip().strip('"\'')
    if path and os.path.isdir(path):
        config.open_folder_in_explorer(path)
        return f"📂 Otvorena mapa u Exploreru: `{path}`"
    elif path and os.path.exists(os.path.dirname(path)):
        config.open_folder_in_explorer(os.path.dirname(path))
        return f"📂 Otvorena mapa u Exploreru: `{os.path.dirname(path)}`"
    return "⚠️ Mapa još ne postoji."

# ---------------- GRADIO UI THEME & CYBER STYLING ----------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root, :root.dark, :root.light, html, body {
    --cyber-bg: #070a12;
    --cyber-card: #0d1424;
    --cyber-card-elevated: #111a30;
    --cyber-card-border: rgba(56, 189, 248, 0.22);
    --cyber-cyan: #06b6d4;
    --cyber-emerald: #10b981;
    --cyber-blue: #3b82f6;
    --cyber-glow-cyan: 0 0 18px rgba(6, 182, 212, 0.35);
    --cyber-glow-emerald: 0 0 18px rgba(16, 185, 129, 0.35);
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;

    /* Enforce Dark Theme Tokens on Gradio internals */
    --background-fill-primary: #070a12 !important;
    --background-fill-secondary: #0d1424 !important;
    --block-background-fill: #0d1424 !important;
    --block-border-color: rgba(56, 189, 248, 0.22) !important;
    --block-label-background-fill: #0d1424 !important;
    --block-label-text-color: #38bdf8 !important;
    --block-title-text-color: #f8fafc !important;
    --body-text-color: #f8fafc !important;
    --body-text-color-subdued: #94a3b8 !important;
    --input-background-fill: #090d16 !important;
    --input-border-color: rgba(56, 189, 248, 0.25) !important;
    --input-placeholder-color: #64748b !important;
    --checkbox-background-color: #090d16 !important;
    --checkbox-background-color-selected: #0284c7 !important;
    --checkbox-border-color: rgba(56, 189, 248, 0.45) !important;
    --checkbox-border-color-selected: #38bdf8 !important;
    --checkbox-label-background-fill: #0d1424 !important;
    --checkbox-label-background-fill-selected: rgba(14, 165, 233, 0.15) !important;
    --checkbox-label-text-color: #f8fafc !important;
    --checkbox-label-text-color-selected: #38bdf8 !important;
    --panel-background-fill: #0d1424 !important;
    --table-even-background-fill: #0d1424 !important;
    --table-odd-background-fill: #090d16 !important;
    --table-text-color: #f8fafc !important;
    --border-color-primary: rgba(56, 189, 248, 0.22) !important;
    color-scheme: dark !important;
}

body, html {
    background-color: #070a12 !important;
    color: #f8fafc !important;
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    margin: 0;
    padding: 0;
}

.gradio-container {
    background: radial-gradient(circle at 50% 0%, #111d38 0%, #070a12 70%) !important;
    color: #f8fafc !important;
    max-width: 98% !important;
    padding: 10px 16px !important;
}

/* Header Bar */
.cyber-header-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(13, 20, 36, 0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--cyber-card-border);
    border-radius: 14px;
    padding: 14px 22px;
    margin-bottom: 14px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    flex-wrap: wrap;
    gap: 14px;
}

.header-left {
    display: flex;
    align-items: center;
    gap: 14px;
}

.header-logo-icon {
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(16, 185, 129, 0.2));
    border: 1px solid rgba(6, 182, 212, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    color: #06b6d4;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.3);
}

.header-titles {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.header-main-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 1.25rem;
    font-weight: 700;
    color: #ffffff;
}

.title-brand {
    background: linear-gradient(135deg, #38bdf8 0%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.title-divider {
    color: #475569;
    font-weight: 300;
}

.title-desc {
    color: #f1f5f9;
}

.header-subtitle {
    font-size: 0.82rem;
    color: #94a3b8;
}

.header-badges {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
}

.header-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 5px 13px;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.pill-success {
    background: rgba(16, 185, 129, 0.12);
    border-color: rgba(16, 185, 129, 0.35);
    color: #34d399;
}

.pill-neutral {
    background: rgba(30, 41, 59, 0.7);
    border-color: rgba(56, 189, 248, 0.25);
    color: #93c5fd;
}

.pill-ai {
    background: rgba(99, 102, 241, 0.12);
    border-color: rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
}

.pill-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
}

.pulse-green {
    background-color: #10b981;
    box-shadow: 0 0 8px #10b981;
    animation: pulseDot 2s infinite ease-in-out;
}

@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
}

.pill-check {
    color: #34d399;
    font-weight: bold;
}

/* Tabs Styling - High Visibility & Crisp Contrast */
.tabs > .tab-nav,
div[role="tablist"] {
    background: rgba(13, 20, 36, 0.85) !important;
    border-radius: 12px !important;
    padding: 6px !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    gap: 8px !important;
    margin-bottom: 14px !important;
}

.tabs > .tab-nav > button,
div[role="tablist"] button,
button[role="tab"] {
    color: #cbd5e1 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: 8px !important;
    padding: 8px 16px !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    background: rgba(15, 23, 42, 0.6) !important;
    transition: all 0.2s ease !important;
}

.tabs > .tab-nav > button:hover,
div[role="tablist"] button:hover,
button[role="tab"]:hover {
    color: #38bdf8 !important;
    background: rgba(56, 189, 248, 0.15) !important;
    border-color: rgba(56, 189, 248, 0.4) !important;
}

.tabs > .tab-nav > button.selected,
div[role="tablist"] button.selected,
div[role="tablist"] button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.35) 0%, rgba(16, 185, 129, 0.25) 100%) !important;
    border: 1px solid rgba(6, 182, 212, 0.65) !important;
    box-shadow: 0 0 16px rgba(6, 182, 212, 0.35) !important;
}

/* Image Upload & Dropzone - Eliminate ALL stark white backgrounds */
.image-container,
.upload-container,
div[data-testid="image"],
div[data-testid="image"] > div,
.empty,
.drop-zone,
div.upload,
.gr-box,
div:has(> input[type="file"]) {
    background: #0b111e !important;
    background-color: #0b111e !important;
    border: 1px dashed rgba(56, 189, 248, 0.35) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

.upload-container *,
.image-container *,
div[data-testid="image"] * {
    color: #94a3b8 !important;
}

.upload-container button,
div[data-testid="image"] button {
    background: rgba(30, 41, 59, 0.85) !important;
    border: 1px solid rgba(56, 189, 248, 0.35) !important;
    color: #38bdf8 !important;
    border-radius: 8px !important;
}

.upload-container button:hover,
div[data-testid="image"] button:hover {
    background: rgba(56, 189, 248, 0.25) !important;
    color: #ffffff !important;
}

/* Checkboxes, Blocks, and Fieldsets */
.block,
label.block,
fieldset.block,
div.block,
.gradio-checkbox,
label:has(input[type="checkbox"]),
label:has(input[type="radio"]),
label.checkbox-label {
    background: rgba(13, 20, 36, 0.9) !important;
    background-color: rgba(13, 20, 36, 0.9) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    transition: all 0.2s ease !important;
}

label:has(input[type="checkbox"]:checked),
label:has(input[type="radio"]:checked) {
    background: rgba(14, 165, 233, 0.12) !important;
    border-color: rgba(56, 189, 248, 0.55) !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
}

label.block span,
label:has(input[type="checkbox"]) span,
label:has(input[type="radio"]) span,
.gradio-checkbox span,
.gradio-radio span,
.block span {
    color: #f8fafc !important;
    font-weight: 500 !important;
}

label:has(input[type="checkbox"]:checked) span,
label:has(input[type="radio"]:checked) span {
    color: #38bdf8 !important;
    font-weight: 600 !important;
}

label:has(input[type="checkbox"]),
label:has(input[type="radio"]),
label.checkbox-label,
.gradio-checkbox label,
.gradio-radio label {
    cursor: pointer !important;
    user-select: none !important;
}

/* Explicit Cyber Checkbox Styling */
input[type="checkbox"] {
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
    width: 20px !important;
    height: 20px !important;
    min-width: 20px !important;
    min-height: 20px !important;
    max-width: 20px !important;
    max-height: 20px !important;
    margin: 0 10px 0 0 !important;
    cursor: pointer !important;
    background-color: #090d16 !important;
    border: 2px solid rgba(56, 189, 248, 0.5) !important;
    border-radius: 5px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    vertical-align: middle !important;
    position: relative !important;
    outline: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6) !important;
    flex-shrink: 0 !important;
}

input[type="checkbox"]:hover {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4), inset 0 2px 4px rgba(0, 0, 0, 0.6) !important;
}

input[type="checkbox"]:checked {
    background-color: #0284c7 !important;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ffffff' stroke-width='3.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='20 6 9 17 4 12'%3E%3C/polyline%3E%3C/svg%3E") !important;
    background-repeat: no-repeat !important;
    background-position: center !important;
    background-size: 14px 14px !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.7), inset 0 1px 2px rgba(255, 255, 255, 0.2) !important;
}

/* Explicit Cyber Radio Styling */
input[type="radio"] {
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
    width: 20px !important;
    height: 20px !important;
    min-width: 20px !important;
    min-height: 20px !important;
    max-width: 20px !important;
    max-height: 20px !important;
    margin: 0 10px 0 0 !important;
    cursor: pointer !important;
    background-color: #090d16 !important;
    border: 2px solid rgba(56, 189, 248, 0.5) !important;
    border-radius: 50% !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    vertical-align: middle !important;
    position: relative !important;
    outline: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.6) !important;
    flex-shrink: 0 !important;
}

input[type="radio"]:hover {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 10px rgba(56, 189, 248, 0.4) !important;
}

input[type="radio"]:checked {
    background-color: #090d16 !important;
    border-color: #38bdf8 !important;
    box-shadow: 0 0 12px rgba(56, 189, 248, 0.7) !important;
}

input[type="radio"]:checked::after {
    content: '' !important;
    display: block !important;
    width: 10px !important;
    height: 10px !important;
    border-radius: 50% !important;
    background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%) !important;
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.9) !important;
}

label.block p,
.block p,
span.meta-text,
p.meta-text {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* Accordions */
.accordion,
details,
details > summary,
.label-wrap,
button.label-wrap,
.cyber-accordion {
    background: rgba(13, 20, 36, 0.9) !important;
    background-color: rgba(13, 20, 36, 0.9) !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
}

details > summary span,
button.label-wrap span,
.label-wrap .icon {
    color: #38bdf8 !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}

/* Block Labels (Top-left titles on components) */
span[data-testid="block-info"],
.block-label,
label > span.label-text,
.label-wrap {
    color: #38bdf8 !important;
    font-weight: 600 !important;
    background: transparent !important;
}

/* Inputs, Textareas, Textboxes, and Markdown blocks */
input:not([type="checkbox"]):not([type="radio"]):not([type="range"]),
textarea,
select,
.gr-input,
.gr-text-input,
div[data-testid="textbox"] textarea,
div[data-testid="textbox"] input {
    background: #090d16 !important;
    background-color: #090d16 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 8px !important;
}

input::placeholder,
textarea::placeholder {
    color: #64748b !important;
}

/* Tables / Dataframes */
table,
.dataframe,
.table-wrap,
.table {
    background: #0d1424 !important;
    color: #f8fafc !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 8px !important;
}

thead, th {
    background: #111a30 !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
    border-bottom: 2px solid rgba(56, 189, 248, 0.3) !important;
}

tbody tr {
    background: #0d1424 !important;
    color: #f8fafc !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.12) !important;
}

tbody tr:nth-child(even) {
    background: #090d16 !important;
}

tbody tr:hover {
    background: rgba(56, 189, 248, 0.15) !important;
}

td {
    color: #f8fafc !important;
    border-color: rgba(56, 189, 248, 0.12) !important;
}

/* Dropdowns */
.dropdown,
.select,
div[data-testid="dropdown"] {
    background: #090d16 !important;
    color: #f8fafc !important;
}

ul.options,
.options-wrap {
    background: #0d1424 !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    color: #f8fafc !important;
}

ul.options li {
    color: #f8fafc !important;
}

ul.options li:hover,
ul.options li.selected {
    background: rgba(56, 189, 248, 0.2) !important;
    color: #38bdf8 !important;
}

/* Cyber Cards & Panels */
.cyber-card {
    background: rgba(13, 20, 36, 0.85) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--cyber-card-border) !important;
    border-radius: 14px !important;
    padding: 14px !important;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
}

/* Buttons */
.btn-cyber-primary {
    background: linear-gradient(135deg, #059669 0%, #0284c7 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 14px rgba(6, 182, 212, 0.3) !important;
    transition: all 0.25s ease !important;
}

.btn-cyber-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(6, 182, 212, 0.55) !important;
    filter: brightness(1.1) !important;
}

.btn-cyber-live {
    background: rgba(15, 23, 42, 0.85) !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
    border: 1px solid rgba(56, 189, 248, 0.4) !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 10px rgba(56, 189, 248, 0.15) !important;
    transition: all 0.25s ease !important;
}

.btn-cyber-live:hover {
    background: rgba(56, 189, 248, 0.15) !important;
    border-color: #38bdf8 !important;
    color: #ffffff !important;
    box-shadow: 0 0 16px rgba(56, 189, 248, 0.4) !important;
}

.btn-cyber-secondary {
    background: rgba(30, 41, 59, 0.8) !important;
    color: #e2e8f0 !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 10px !important;
}

.btn-cyber-secondary:hover {
    background: rgba(51, 65, 85, 0.9) !important;
    border-color: rgba(56, 189, 248, 0.5) !important;
    color: #ffffff !important;
}

/* Visual Panel & HUD Frame */
.cyber-preview-frame {
    border-radius: 12px !important;
    border: 1px solid rgba(6, 182, 212, 0.35) !important;
    background: radial-gradient(circle at center, rgba(17, 26, 48, 0.6) 0%, rgba(7, 10, 18, 0.95) 100%) !important;
    box-shadow: inset 0 0 20px rgba(6, 182, 212, 0.12), 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

.cyber-status-text {
    margin-top: 8px;
    padding: 8px 12px;
    border-radius: 8px;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.06);
    font-size: 0.85rem;
}

/* Real-time Detection Side Panel */
.detection-panel-container {
    min-height: 420px;
    display: flex;
    flex-direction: column;
}

.detection-panel-inner {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-bottom: 10px;
    border-bottom: 1px solid rgba(56, 189, 248, 0.15);
    margin-bottom: 12px;
}

.panel-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 0.88rem;
    font-weight: 700;
    color: #e2e8f0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.pulse-icon {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background-color: #06b6d4;
    box-shadow: 0 0 8px #06b6d4;
}

.pulse-icon.active {
    background-color: #10b981;
    box-shadow: 0 0 10px #10b981;
    animation: pulseDot 1.5s infinite ease-in-out;
}

.panel-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 6px;
    background: rgba(30, 41, 59, 0.8);
    color: #94a3b8;
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.panel-badge.active {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border-color: rgba(16, 185, 129, 0.35);
}

/* Empty State Radar Scan */
.detection-empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 32px 14px;
    text-align: center;
    background: rgba(15, 23, 42, 0.4);
    border-radius: 12px;
    border: 1px dashed rgba(56, 189, 248, 0.2);
    margin: auto 0;
}

.radar-scan-box {
    width: 54px;
    height: 54px;
    border-radius: 50%;
    border: 2px solid rgba(6, 182, 212, 0.3);
    position: relative;
    margin-bottom: 14px;
    box-shadow: 0 0 14px rgba(6, 182, 212, 0.15);
    overflow: hidden;
}

.radar-beam {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    border-radius: 50%;
    background: conic-gradient(from 0deg, rgba(6, 182, 212, 0.4) 0deg, transparent 60deg, transparent 360deg);
    animation: radarSweep 3s linear infinite;
}

@keyframes radarSweep {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

.empty-title {
    font-size: 0.92rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 4px;
}

.empty-sub {
    font-size: 0.78rem;
    color: #64748b;
    line-height: 1.4;
}

/* Cyber Detection Cards */
.cyber-cards-scroll {
    display: flex;
    flex-direction: column;
    gap: 10px;
    max-height: 460px;
    overflow-y: auto;
    padding-right: 4px;
}

.cyber-detection-card {
    display: flex;
    align-items: center;
    gap: 10px;
    background: rgba(15, 23, 42, 0.7);
    border-radius: 12px;
    padding: 8px 10px;
    border: 1px solid rgba(255, 255, 255, 0.07);
    transition: all 0.25s ease;
}

.cyber-detection-card:hover {
    transform: translateX(2px);
    background: rgba(30, 41, 59, 0.85);
}

.cyber-detection-card.match-success {
    border-left: 3px solid #10b981;
    box-shadow: 0 2px 10px rgba(16, 185, 129, 0.1);
}

.cyber-detection-card.match-warning {
    border-left: 3px solid #f59e0b;
    box-shadow: 0 2px 10px rgba(245, 158, 11, 0.1);
}

.cyber-detection-card.match-unknown {
    border-left: 3px solid #64748b;
}

.card-avatar-wrap {
    position: relative;
    width: 48px;
    height: 48px;
    flex-shrink: 0;
}

.card-avatar {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    object-fit: cover;
    border: 1px solid rgba(56, 189, 248, 0.25);
    background: #1e293b;
}

.card-status-dot {
    position: absolute;
    bottom: -2px;
    right: -2px;
    width: 9px;
    height: 9px;
    border-radius: 50%;
    border: 2px solid #0d1424;
}

.match-success .card-status-dot { background-color: #10b981; }
.match-warning .card-status-dot { background-color: #f59e0b; }
.match-unknown .card-status-dot { background-color: #64748b; }

.card-details {
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.card-name {
    font-size: 0.88rem;
    font-weight: 700;
    color: #f8fafc;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.card-meta {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.75rem;
    color: #94a3b8;
}

.card-meta b {
    color: #cbd5e1;
}

.meta-sep {
    color: #475569;
}

.card-similarity-badge {
    margin-top: 1px;
}

.sim-pill {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 5px;
    font-family: 'JetBrains Mono', monospace;
}

.match-success .sim-pill {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.match-warning .sim-pill {
    background: rgba(245, 158, 11, 0.18);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.match-unknown .sim-pill {
    background: rgba(100, 116, 139, 0.2);
    color: #94a3b8;
    border: 1px solid rgba(100, 116, 139, 0.35);
}

.cyber-detection-card.is-vip {
    border-left: 3px solid #f59e0b !important;
    box-shadow: 0 0 16px rgba(245, 158, 11, 0.22) !important;
}

.cyber-detection-card.is-blacklist {
    border-left: 3px solid #ef4444 !important;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.28) !important;
}

.card-name-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}

.role-pill {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}

.role-verified {
    background: rgba(16, 185, 129, 0.18);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.35);
}

.role-vip {
    background: rgba(245, 185, 11, 0.22);
    color: #fbbf24;
    border: 1px solid rgba(245, 185, 11, 0.5);
    box-shadow: 0 0 8px rgba(245, 185, 11, 0.3);
}

.role-blacklist {
    background: rgba(239, 68, 68, 0.22);
    color: #f87171;
    border: 1px solid rgba(239, 68, 68, 0.5);
    box-shadow: 0 0 8px rgba(239, 68, 68, 0.3);
}

.role-possible {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.card-conf-track {
    width: 100%;
    height: 4px;
    background: rgba(15, 23, 42, 0.7);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 5px;
}

.card-conf-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s ease;
}

.match-success .card-conf-fill { background: linear-gradient(90deg, #059669, #10b981); }
.match-warning .card-conf-fill { background: linear-gradient(90deg, #d97706, #f59e0b); }
.match-unknown .card-conf-fill { background: linear-gradient(90deg, #475569, #64748b); }
.is-vip .card-conf-fill { background: linear-gradient(90deg, #d97706, #f59e0b, #fbbf24) !important; }
.is-blacklist .card-conf-fill { background: linear-gradient(90deg, #b91c1c, #ef4444) !important; }

/* Cyber Shortcuts Card in Live Camera UI */
.cyber-shortcuts-card {
    background: rgba(13, 20, 36, 0.75);
    border: 1px solid rgba(56, 189, 248, 0.20);
    border-radius: 12px;
    padding: 10px 14px;
    margin-top: 10px;
    margin-bottom: 4px;
    width: 100%;
    box-sizing: border-box;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);
}

.shortcuts-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}

.shortcuts-badge {
    background: rgba(6, 182, 212, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    font-size: 0.62rem;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 5px;
    letter-spacing: 0.5px;
}

.shortcuts-title {
    font-size: 0.78rem;
    font-weight: 600;
    color: #cbd5e1;
}

.shortcuts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 6px 10px;
}

.sc-item {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.73rem;
    color: #94a3b8;
}

.sc-item kbd {
    background: #1e293b;
    border: 1px solid rgba(148, 163, 184, 0.35);
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.68rem;
    font-weight: 700;
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.4);
}

.sc-item span {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

input[type="range"] {
    accent-color: #06b6d4 !important;
}

/* Person Mini-Avatar Thumbnail */
.cyber-avatar-wrapper {
    width: 96px !important;
    min-width: 96px !important;
    max-width: 96px !important;
    height: 96px !important;
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.cyber-person-avatar-thumb {
    width: 96px !important;
    height: 96px !important;
    min-width: 96px !important;
    min-height: 96px !important;
    max-width: 96px !important;
    max-height: 96px !important;
    border-radius: 14px !important;
    border: 2px solid #06b6d4 !important;
    box-shadow: 0 0 16px rgba(6, 182, 212, 0.45) !important;
    background: #0f172a !important;
    overflow: hidden !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

.cyber-person-avatar-thumb img {
    object-fit: cover !important;
    border-radius: 12px !important;
    width: 100% !important;
    height: 100% !important;
    display: block !important;
}

.cyber-person-avatar-thumb svg {
    opacity: 0.45;
}

.person-card-top-row {
    align-items: center !important;
    gap: 14px !important;
}

/* ═══════════════════════════════════════════════════════
   PILL TAB NAVIGACIJA — Custom cyber stil
═══════════════════════════════════════════════════════ */
.tabs > .tab-nav {
    background: rgba(7, 10, 18, 0.6) !important;
    border-bottom: 1px solid rgba(56, 189, 248, 0.18) !important;
    padding: 6px 6px 0 6px !important;
    gap: 4px !important;
    flex-wrap: wrap !important;
}

.tabs > .tab-nav > button {
    background: rgba(13, 20, 36, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.18) !important;
    border-bottom: none !important;
    border-radius: 10px 10px 0 0 !important;
    color: #94a3b8 !important;
    font-weight: 600 !important;
    font-size: 0.87rem !important;
    padding: 7px 16px !important;
    letter-spacing: 0.02em !important;
    transition: all 0.22s ease !important;
    position: relative !important;
}

.tabs > .tab-nav > button:hover {
    background: rgba(30, 41, 59, 0.85) !important;
    color: #e2e8f0 !important;
    border-color: rgba(56, 189, 248, 0.35) !important;
}

.tabs > .tab-nav > button.selected {
    background: linear-gradient(180deg, rgba(6,182,212,0.18) 0%, rgba(13,20,36,0.95) 100%) !important;
    border-color: rgba(6, 182, 212, 0.5) !important;
    color: #06b6d4 !important;
    box-shadow: 0 -2px 12px rgba(6,182,212,0.2), inset 0 1px 0 rgba(6,182,212,0.3) !important;
    text-shadow: 0 0 8px rgba(6,182,212,0.5) !important;
}

.tabs > .tab-nav > button.selected::after {
    content: '';
    position: absolute !important;
    bottom: -1px;
    left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #06b6d4, transparent) !important;
}

/* Blur Mode Radio Group — vizualno grupiranje */
.blur-mode-group .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 5px !important;
}

.blur-mode-group label.svelte-1gfkn6j,
.blur-mode-group label {
    background: rgba(13, 20, 36, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.15) !important;
    border-radius: 8px !important;
    padding: 5px 10px !important;
    transition: all 0.18s ease !important;
}

.blur-mode-group label:has(input:checked) {
    background: rgba(6, 182, 212, 0.12) !important;
    border-color: rgba(6, 182, 212, 0.4) !important;
    color: #38bdf8 !important;
}

/* Similarity confidence bar */
.sim-bar-wrap {
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 3px;
    margin-top: 4px;
    overflow: hidden;
}

/* Edit person section */
.edit-person-card {
    border: 1px solid rgba(6, 182, 212, 0.35) !important;
    background: rgba(10, 25, 45, 0.65) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    margin: 10px 0 !important;
    box-shadow: 0 4px 16px rgba(6, 182, 212, 0.08) !important;
}

.edit-person-card h4 {
    color: #38bdf8 !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
}

/* Retention policy card */
.retention-card {
    border: 1px solid rgba(16, 185, 129, 0.35) !important;
    background: rgba(10, 30, 35, 0.65) !important;
    border-radius: 12px !important;
    padding: 14px 16px !important;
    margin-top: 14px !important;
    box-shadow: 0 4px 16px rgba(16, 185, 129, 0.08) !important;
}

.retention-card h4 {
    color: #10b981 !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
}

/* Retention Radio Group */
.retention-radio-group .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}

.retention-radio-group label {
    background: rgba(13, 20, 36, 0.7) !important;
    border: 1px solid rgba(16, 185, 129, 0.2) !important;
    border-radius: 8px !important;
    padding: 7px 12px !important;
    transition: all 0.18s ease !important;
    cursor: pointer !important;
}

.retention-radio-group label:hover {
    border-color: rgba(16, 185, 129, 0.5) !important;
    background: rgba(16, 185, 129, 0.08) !important;
}

.retention-radio-group label:has(input:checked) {
    background: rgba(16, 185, 129, 0.18) !important;
    border-color: rgba(16, 185, 129, 0.6) !important;
    color: #34d399 !important;
}

/* --- DANGER ZONE & CONFIRMATION MODAL --- */
.danger-zone-accordion {
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    background: rgba(13, 20, 36, 0.6) !important;
    border-radius: 12px !important;
    margin-top: 24px !important;
    overflow: hidden !important;
}

.danger-zone-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.08) 0%, rgba(13, 20, 36, 0.95) 100%) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
}

.btn-cyber-danger {
    background: linear-gradient(135deg, #991b1b 0%, #ef4444 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 1px solid #f87171 !important;
    box-shadow: 0 0 16px rgba(239, 68, 68, 0.35) !important;
    border-radius: 8px !important;
    transition: all 0.2s ease !important;
}

.btn-cyber-danger:hover {
    background: linear-gradient(135deg, #dc2626 0%, #ff4d4d 100%) !important;
    box-shadow: 0 0 28px rgba(239, 68, 68, 0.7) !important;
    transform: translateY(-1px) !important;
}

.cyber-modal-overlay {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 100vw !important;
    height: 100vh !important;
    background: rgba(4, 7, 15, 0.85) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    z-index: 999999 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 20px !important;
}

.cyber-modal-box {
    background: #0d1424 !important;
    border: 2px solid #ef4444 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 50px rgba(239, 68, 68, 0.5), inset 0 0 20px rgba(239, 68, 68, 0.1) !important;
    max-width: 640px !important;
    width: 95% !important;
    margin: auto !important;
    padding: 26px !important;
}

/* --- MULTI-SELECT BATCH DELETE TRAY --- */
.multi-select-toggle-row {
    background: rgba(13, 20, 36, 0.7) !important;
    border: 1px solid rgba(56, 189, 248, 0.25) !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    margin: 8px 0 !important;
}

.multi-select-checkbox label {
    font-weight: 600 !important;
    color: #38bdf8 !important;
}

.batch-delete-tray {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.06) 0%, rgba(13, 20, 36, 0.9) 100%) !important;
    border: 1px solid rgba(239, 68, 68, 0.35) !important;
    border-radius: 10px !important;
    padding: 14px 16px !important;
    margin: 10px 0 !important;
    box-shadow: 0 4px 18px rgba(239, 68, 68, 0.1) !important;
}
"""

HEAD_DARK_JS = """
<script>
    (function() {
        document.documentElement.classList.add('dark');
        document.documentElement.setAttribute('data-theme', 'dark');
        if (document.body) {
            document.body.classList.add('dark');
        }
        try {
            localStorage.setItem('color-theme', 'dark');
            localStorage.setItem('theme', 'dark');
        } catch(e) {}
        window.__theme = 'dark';

        const obs = new MutationObserver(function() {
            if (!document.documentElement.classList.contains('dark')) {
                document.documentElement.classList.add('dark');
            }
            if (document.body && !document.body.classList.contains('dark')) {
                document.body.classList.add('dark');
            }
        });
        obs.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

        // Globalno sprječavanje nehotične navigacije prozora pri povlačenju datoteka (Drag & Drop)
        window.addEventListener('dragover', function(e) {
            e.preventDefault();
        }, false);
        window.addEventListener('drop', function(e) {
            var target = e.target;
            if (!target.closest('input[type="file"]') && !target.closest('.upload-container') && !target.closest('.dropzone') && !target.closest('[data-testid="image"]')) {
                e.preventDefault();
            }
        }, false);
    })();
</script>
"""

custom_theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="blue",
    neutral_hue="slate"
).set(
    body_background_fill="#070a12",
    body_background_fill_dark="#070a12",
    background_fill_primary="#070a12",
    background_fill_primary_dark="#070a12",
    background_fill_secondary="#0d1424",
    background_fill_secondary_dark="#0d1424",
    block_background_fill="#0d1424",
    block_background_fill_dark="#0d1424",
    block_border_color="rgba(56, 189, 248, 0.22)",
    block_border_color_dark="rgba(56, 189, 248, 0.22)",
    block_label_background_fill="#0d1424",
    block_label_background_fill_dark="#0d1424",
    block_label_text_color="#38bdf8",
    block_label_text_color_dark="#38bdf8",
    block_title_text_color="#f8fafc",
    block_title_text_color_dark="#f8fafc",
    body_text_color="#f8fafc",
    body_text_color_dark="#f8fafc",
    body_text_color_subdued="#94a3b8",
    body_text_color_subdued_dark="#94a3b8",
    input_background_fill="#090d16",
    input_background_fill_dark="#090d16",
    input_border_color="rgba(56, 189, 248, 0.25)",
    input_border_color_dark="rgba(56, 189, 248, 0.25)",
    input_placeholder_color="#64748b",
    input_placeholder_color_dark="#64748b",
    checkbox_background_color="#090d16",
    checkbox_background_color_selected="#0284c7",
    checkbox_background_color_dark="#090d16",
    checkbox_background_color_selected_dark="#0284c7",
    checkbox_border_color="rgba(56, 189, 248, 0.45)",
    checkbox_border_color_selected="#38bdf8",
    checkbox_border_color_dark="rgba(56, 189, 248, 0.45)",
    checkbox_border_color_selected_dark="#38bdf8",
    checkbox_label_background_fill="#0d1424",
    checkbox_label_background_fill_dark="#0d1424",
    checkbox_label_text_color="#f8fafc",
    checkbox_label_text_color_dark="#f8fafc",
    accordion_text_color="#f8fafc",
    accordion_text_color_dark="#f8fafc",
    table_even_background_fill="#0d1424",
    table_even_background_fill_dark="#0d1424",
    table_odd_background_fill="#090d16",
    table_odd_background_fill_dark="#090d16",
    table_text_color="#f8fafc",
    table_text_color_dark="#f8fafc",
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_dark="#1e293b",
    button_secondary_text_color="#f8fafc",
    button_secondary_text_color_dark="#f8fafc"
)

with gr.Blocks(title="Argusface - Sustav za Prepoznavanje Lica") as demo:
    # Per-session state (eliminates global variables and multi-user race conditions)
    rec_faces_state = gr.State([])
    single_enroll_state = gr.State({
        "faces": [],
        "bgr": None,
        "selected_idx": 1,
        "saved_indices": set()
    })

    # Modern Cyber Glassmorphism Header Bar
    gr.HTML(
        """
        <div class="cyber-header-bar">
            <div class="header-left">
                <div class="header-logo-icon">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M9 3H5a2 2 0 0 0-2 2v4m0 6v4a2 2 0 0 0 2 2h4m6 0h4a2 2 0 0 0 2-2v-4m0-6V5a2 2 0 0 0-2-2h-4"/>
                        <circle cx="12" cy="10" r="3"/>
                        <path d="M7 18a5 5 0 0 1 10 0"/>
                    </svg>
                </div>
                <div class="header-titles">
                    <div class="header-main-title">
                        <span class="title-brand">ArgusFace</span>
                        <span class="title-divider">•</span>
                        <span class="title-desc">Biometrijski Sustav za Prepoznavanje Lica</span>
                    </div>
                    <div class="header-subtitle">
                        100% lokalno i sigurno &nbsp;|&nbsp; RetinaFace detektor • EdgeFace Base prepoznavanje
                    </div>
                </div>
            </div>
            <div class="header-badges">
                <div class="header-pill pill-success">
                    <span class="pill-dot pulse-green"></span>
                    <span>100% Lokalno i sigurno</span>
                </div>
                <div class="header-pill pill-neutral">
                    <span class="pill-icon">🖥️</span>
                    <span>Server aktivan</span>
                    <span class="pill-check">✓</span>
                </div>
                <div class="header-pill pill-ai">
                    <span class="pill-icon">🧠</span>
                    <span>AI Engine (ONNX)</span>
                </div>
            </div>
        </div>
        """
    )
    
    with gr.Tabs():
        # ------------------ TAB 1: PREPOZNAVANJE ------------------
        with gr.TabItem("🔍 Prepoznavanje lica"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=4, min_width=320, elem_classes=["cyber-card"]):
                    input_img = gr.Image(
                        type="pil",
                        label="Učitaj sliku ili snimi web kamerom",
                        sources=["upload", "webcam"]
                    )
                    with gr.Accordion("⚙️ Napredne postavke analize", open=False, elem_classes=["cyber-accordion"]):
                        threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.45,
                            step=0.01,
                            label="Prag prepoznavanja (Kosinusna sličnost)",
                            info="Preporučeno: 0.40 - 0.50 (veće = strože, manje = blaže)"
                        )
                        with gr.Row():
                            landmarks_chk = gr.Checkbox(
                                value=False,
                                label="🔵 Prikaži točke lica (Landmarks)",
                                info="Biometrijska wireframe geometrija lica"
                            )
                        blur_mode_radio = gr.Radio(
                            choices=[
                                "🔓 Bez zamućivanja",
                                "🔒 Zamuti nepoznata lica",
                                "🛡️ Zamuti sva lica (GDPR)"
                            ],
                            value="🔓 Bez zamućivanja",
                            label="Privatnost / Anonimizacija",
                            info="Odaberite razinu zaštite privatnosti na slici",
                            elem_classes=["blur-mode-group"]
                        )
                    with gr.Row():
                        btn_recognize = gr.Button("🚀 Pokreni prepoznavanje slike", variant="primary", scale=2, elem_classes=["btn-cyber-primary"])
                            
                    with gr.Accordion("📹 Live Nadzor i Kamere (Pojedinačna ili 2×2 Mreža)", open=False, elem_classes=["cyber-accordion"]):
                        live_mode_radio = gr.Radio(
                            choices=["Pojedinačna kamera (Single View)", "Mreža više kamera (2×2 Grid)"],
                            value="Pojedinačna kamera (Single View)",
                            label="Način rada nadzora"
                        )
                        
                        # 1. Pojedinačna kamera panel
                        with gr.Column(visible=True) as single_cam_box:
                            cam_source_type = gr.Radio(
                                choices=[
                                    "USB Web Kamera", 
                                    "IP / RTSP Kamera za nadzor", 
                                    "YouTube / Web Video", 
                                    "Lokalna Video Datoteka"
                                ],
                                value="USB Web Kamera",
                                label="Vrsta video izvora"
                            )
                            cam_usb_idx = gr.Dropdown(
                                choices=["0", "1", "2", "3"],
                                value="0",
                                label="Indeks lokalne USB kamere",
                                info="0 je ugrađena ili prva spojena kamera, 1 je druga..."
                            )
                            cam_rtsp_url = gr.Textbox(
                                label="RTSP ili HTTP adresa IP kamere za nadzor",
                                value="rtsp://admin:admin@192.168.50.236:554/11",
                                placeholder="npr. rtsp://admin:admin@192.168.50.236:554/11",
                                info="Podržava RTSP streamove sigurnosnih kamera ili HTTP MJPEG stream s mobitela",
                                visible=False
                            )
                            cam_youtube_url = gr.Textbox(
                                label="YouTube / Vimeo video link",
                                placeholder="npr. https://www.youtube.com/watch?v=... ili https://youtu.be/...",
                                info="Automatski dohvaća i reproducira video stream u stvarnom vremenu s prepoznavanjem lica",
                                visible=False
                            )
                            cam_video_file = gr.File(
                                label="Učitaj video datoteku s računala (.mp4, .mkv, .avi, .mov)",
                                file_types=[".mp4", ".mkv", ".avi", ".mov"],
                                visible=False
                            )
                            cam_start_sec = gr.Slider(
                                minimum=0,
                                maximum=7200,
                                value=0,
                                step=5,
                                label="⏩ Početak reprodukcije videa (u sekundama)",
                                info="Omogućuje pokretanje YouTube ili lokalnog videa od željene minute/sekunde",
                                visible=False
                            )

                        # 2. 2x2 Nadzorna mreža panel
                        with gr.Column(visible=False) as multi_cam_box:
                            gr.Markdown("#### 🎛️ Konfiguracija kamera za 2×2 Nadzornu Mrežu")
                            with gr.Row():
                                mc_c1_on = gr.Checkbox(value=True, label="Kamera 1 aktivna", scale=1)
                                mc_c1_name = gr.Textbox(value="USB Web Kamera", label="Naziv Kamere 1", scale=2)
                                mc_c1_src = gr.Textbox(value="0", label="Izvor (USB indeks ili RTSP)", scale=3)
                            with gr.Row():
                                mc_c2_on = gr.Checkbox(value=True, label="Kamera 2 aktivna", scale=1)
                                mc_c2_name = gr.Textbox(value="Denver IP Nadzor", label="Naziv Kamere 2", scale=2)
                                mc_c2_src = gr.Textbox(value="rtsp://admin:admin@192.168.50.236:554/11", label="Izvor (RTSP link)", scale=3)
                            with gr.Row():
                                mc_c3_on = gr.Checkbox(value=False, label="Kamera 3 aktivna", scale=1)
                                mc_c3_name = gr.Textbox(value="Kamera 3", label="Naziv Kamere 3", scale=2)
                                mc_c3_src = gr.Textbox(value="", placeholder="Opcionalno: RTSP ili USB indeks", label="Izvor", scale=3)
                            with gr.Row():
                                mc_c4_on = gr.Checkbox(value=False, label="Kamera 4 aktivna", scale=1)
                                mc_c4_name = gr.Textbox(value="Kamera 4", label="Naziv Kamere 4", scale=2)
                                mc_c4_src = gr.Textbox(value="", placeholder="Opcionalno: RTSP ili USB indeks", label="Izvor", scale=3)

                    with gr.Row():
                        cam_enable_log = gr.Checkbox(
                            value=False,
                            label="📋 Aktiviraj evidenciju prolazaka (Dnevnik)",
                            info="Automatski zapisuje prepoznate osobe u evidenciju",
                            scale=2
                        )
                        cam_cooldown_sec = gr.Slider(
                            minimum=5,
                            maximum=300,
                            value=30,
                            step=5,
                            label="⏱️ Cooldown filter (sekunde)",
                            info="Istu osobu ne bilježi ponovno unutar zadanog broja sekundi",
                            visible=False,
                            scale=2
                        )
                    with gr.Row():
                        cam_enable_nvr = gr.Checkbox(
                            value=False,
                            label="🔴 24/7 NVR Video Snimanje (MP4)",
                            info="Kontinuirano snima video segmente uz automatsko brisanje starih snimki",
                            scale=2
                        )
                        cam_nvr_segment_min = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=5,
                            step=1,
                            label="⏱️ Trajanje segmenta (minute)",
                            info="Automatska rotacija video datoteka (1 - 60 min)",
                            visible=False,
                            scale=2
                        )
                    with gr.Row():
                        cam_enable_spoof = gr.Checkbox(
                            value=True,
                            label="🛡️ Zaštita od lažiranja (Anti-Spoofing / Liveness)",
                            info="Sprječava prevaru pokazivanjem fotografija ili ekrana mobitela. Isključite ako želite identificirati osobe sa slika na mobitelu.",
                            scale=2
                        )

                    with gr.Row():
                        btn_launch_live = gr.Button("🎥 Pokreni Live Kameru", variant="secondary", scale=2, elem_classes=["btn-cyber-live"])
                        btn_launch_grid = gr.Button("🎛️ Pokreni 2×2 Mrežu", variant="secondary", scale=2, visible=False, elem_classes=["btn-cyber-live"])
                        btn_open_snaps_quick = gr.Button("📂 Snimke (S)", variant="secondary", scale=1, elem_classes=["btn-cyber-secondary"])
                    with gr.Row():
                        live_shortcuts_html = gr.HTML("""
                        <div class="cyber-shortcuts-card">
                            <div class="shortcuts-header">
                                <span class="shortcuts-badge">TIPKOVNICA & MIŠ</span>
                                <span class="shortcuts-title">Brze kontrole u video prozoru:</span>
                            </div>
                            <div class="shortcuts-grid">
                                <div class="sc-item"><kbd>SPACE</kbd> <span>Pauza / Nastavak</span></div>
                                <div class="sc-item"><kbd>←</kbd> <kbd>→</kbd> <span>±5s skok</span></div>
                                <div class="sc-item"><kbd>,</kbd> <kbd>.</kbd> <span>Kadar-po-kadar</span></div>
                                <div class="sc-item"><kbd>0</kbd>–<kbd>9</kbd> <span>Skok na % videa</span></div>
                                <div class="sc-item"><kbd>Miš</kbd> <span>Klik / vučenje trake</span></div>
                                <div class="sc-item"><kbd>F</kbd> <span>AntiSpoof ON/OFF</span></div>
                                <div class="sc-item"><kbd>M</kbd> <span>Filter (Samo zelena)</span></div>
                                <div class="sc-item"><kbd>S</kbd> <span>Spremi kadar (Foto)</span></div>
                            </div>
                        </div>
                        """)
                    
                # 2. Srednji stupac: Vizualni rezultat (cca 42% širine)
                with gr.Column(scale=5, min_width=380, elem_classes=["cyber-card"]):
                    annotated_out = gr.Image(
                        type="numpy", 
                        label="Vizualni rezultat prepoznavanja",
                        elem_classes=["cyber-preview-frame"]
                    )
                    rec_status_md = gr.Markdown(
                        "Učitajte sliku i kliknite 'Pokreni prepoznavanje'.",
                        elem_classes=["cyber-status-text"]
                    )

                # 3. Desni stupac: Real-time Detection panel (cca 25% širine)
                with gr.Column(scale=3, min_width=270, elem_classes=["cyber-card"]):
                    cards_show_all_chk = gr.Checkbox(
                        value=False,
                        label="Prikaži i nepoznata lica (% sličnosti)",
                        info="Zadano: prikaz samo prepoznatih (zelena)"
                    )
                    detection_cards_html = gr.HTML(
                        value=generate_detection_cards_html([], show_all_faces=False),
                        elem_classes=["detection-panel-container"]
                    )
                    
            # Donja zona: Detaljna biometrijska tablica & brzi unos
            with gr.Row():
                with gr.Column(elem_classes=["cyber-card"]):
                    with gr.Accordion("📊 Detaljna biometrijska tablica & analitika detektiranih lica", open=False):
                        results_table = gr.Dataframe(
                            headers=[
                                "Lice #",
                                "Status",
                                "Identificirana Osoba",
                                "Sličnost",
                                "Prag",
                                "Margina sigurnosti",
                                "Kvaliteta profila",
                                "Dob (procjena)",
                                "Spol"
                            ],
                            label="Rezultati prepoznavanja i biometrijske procjene",
                            interactive=False
                        )

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["cyber-card"]):
                    gr.Markdown("### 🖼️ Izrezana lica s fotografije *(kliknite na lice za brzi unos)*")
                    crops_gallery_out = gr.Gallery(
                        label="Galerija detektiranih lica",
                        columns=4,
                        height="auto",
                        allow_preview=False
                    )
                with gr.Column(scale=1, elem_classes=["cyber-card"]):
                    gr.Markdown("### ➕ Brzi unos nepoznate osobe u bazu")
                    gr.Markdown("Kliknite na lice u galeriji s lijeve strane ili odaberite iz padajućeg izbornika:")
                    with gr.Row():
                        unknown_face_dropdown = gr.Dropdown(
                            label="Odaberite detektirano lice sa slike",
                            choices=[],
                            allow_custom_value=True
                        )
                        quick_name_input = gr.Textbox(label="Ime osobe", placeholder="npr. Marko Horvat")
                    btn_quick_add = gr.Button("Spremi ovo lice u bazu za tu osobu", variant="secondary", elem_classes=["btn-cyber-primary"])
                    quick_add_status = gr.Markdown("")

        # ------------------ TAB 2: UPRAVLJANJE BAZOM ------------------
        with gr.TabItem("👥 Baza Osoba"):
            with gr.Row():
                # Lijevi stupac: Unos i označavanje na grupnoj slici
                with gr.Column(scale=1):
                    with gr.Tabs():
                        # Podtab 1: Pojedinačni unos i izrezivanje s grupne slike
                        with gr.TabItem("Pojedinačni unos / Označavanje lica"):
                            # Mini-avatar kartica odabrane osobe i brzi pretraživač
                            with gr.Row(equal_height=True, elem_classes=["person-card-top-row"]):
                                selected_person_avatar = gr.HTML(
                                    value=render_person_avatar_html(None),
                                    elem_classes=["cyber-avatar-wrapper"]
                                )
                                with gr.Column(scale=5):
                                    existing_person_picker = gr.Dropdown(
                                        label="🔍 Odabrana osoba u bazi (ili pretražite drugu):",
                                        choices=get_person_dropdown_choices(),
                                        allow_custom_value=True,
                                        info="Kliknite na redak u tablici ili počnite tipkati ime"
                                    )
                                    
                            with gr.Row():
                                single_name_input = gr.Textbox(
                                    label="Ime i prezime osobe",
                                    placeholder="Upišite novo ime ili odaberite osobu iznad",
                                    scale=3
                                )
                                btn_clear_form = gr.Button("🔄 Očisti", size="sm", scale=1, elem_classes=["btn-cyber-secondary"])
                                
                            single_notes_input = gr.Textbox(label="Bilješke (opcionalno)", placeholder="npr. Član tima, IT odjel")
                            single_role_input = gr.Dropdown(
                                label="Sigurnosni status / Uloga osobe",
                                choices=[
                                    ("Standardna osoba / Korisnik", "standard"),
                                    ("⭐ VIP uzvanik / Važna osoba", "vip"),
                                    ("🚨 Crna lista / Nepoželjni (Zabrana)", "blacklist")
                                ],
                                value="standard",
                                info="Određuje boju okvira na kameri i automatsko slanje Telegram alarma"
                            )
                            
                            single_img_input = gr.Image(
                                type="pil",
                                label="Učitaj sliku ili snimi web kamerom",
                                sources=["upload", "webcam"]
                            )
                            
                            single_annotated_preview = gr.Image(
                                type="numpy",
                                label="Detektirana lica na slici (Kliknite direktno na lice na slici za odabir!)",
                                visible=False,
                                interactive=False
                            )
                            
                            single_face_selector = gr.Radio(
                                label="👉 Odaberite koje lice želite spremiti za ovu osobu:",
                                choices=[],
                                visible=False
                            )
                            
                            single_crops_gallery = gr.Gallery(
                                label="Ili kliknite na sličicu lica ovdje u galeriji:",
                                columns=4,
                                height="auto",
                                allow_preview=False,
                                visible=False
                            )
                            
                            with gr.Row():
                                single_preview_crop = gr.Image(type="numpy", label="Trenutno odabrano lice", height=160, scale=1)
                                single_preview_info = gr.Markdown("Učitajte sliku za automatsku detekciju lica.", scale=2)
                            
                            btn_save_single = gr.Button("💾 Spremi odabrano lice u bazu", variant="primary", size="lg", elem_classes=["btn-cyber-primary"])
                            single_save_status = gr.Markdown("")

                        # Podtab 2: Masovni unos (10+ slika)
                        with gr.TabItem("⚡ Masovni unos (Batch)"):
                            gr.Markdown(
                                """
                                **Savjet:** Možete prenijeti **10-ak ili više fotografija odjednom**!
                                Ako su slike nazvane imenom osobe (npr. `Luka_Vugrek.jpg`, `Ana_Kovacic.jpg`),
                                odaberite prvu opciju i sustav će automatski sve unijeti u par sekundi.
                                """
                            )
                            batch_naming_mode = gr.Radio(
                                ["Ime iz naziva datoteke", "Sve slike pripadaju istoj osobi"],
                                value="Ime iz naziva datoteke",
                                label="Kako odrediti ime?"
                            )
                            batch_single_name = gr.Textbox(
                                label="Ime osobe (samo ako sve slike pripadaju istoj osobi)",
                                placeholder="npr. Marko",
                                visible=False
                            )
                            batch_files_input = gr.File(
                                file_count="multiple",
                                label="Odaberite više slika odjednom (Drag & Drop)",
                                file_types=["image"]
                            )
                            btn_batch_enroll = gr.Button("⚡ Uvezi sve fotografije u bazu", variant="primary", elem_classes=["btn-cyber-primary"])
                            batch_status_md = gr.Markdown("")

                # Desni stupac: Pregled, pretraga i upravljanje
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 Registrirane osobe u bazi")
                    
                    # Pretraživač tablice u realnom vremenu
                    with gr.Row():
                        table_search_input = gr.Textbox(
                            label="🔍 Brzo pretraživanje tablice:",
                            placeholder="Upišite ime, ID ili bilješku za filtriranje...",
                            scale=4
                        )
                        btn_clear_table_search = gr.Button("✖ Poništi", size="sm", scale=1, elem_classes=["btn-cyber-secondary"])
                        
                    # Preklopnik za višestruki odabir (Grupno brisanje)
                    with gr.Row(elem_classes=["multi-select-toggle-row"]):
                        multi_select_mode_cb = gr.Checkbox(
                            label="☑️ Omogući višestruki odabir u tablici (Grupno označavanje za brisanje)",
                            value=False,
                            interactive=True,
                            elem_classes=["multi-select-checkbox"]
                        )
                    
                    # Traka s označenim osobama i kontrolama za brisanje
                    with gr.Group(visible=False, elem_classes=["batch-delete-tray"]) as batch_delete_tray:
                        with gr.Row(equal_height=True):
                            batch_selected_dropdown = gr.Dropdown(
                                label="Označene osobe za grupno brisanje:",
                                choices=get_person_dropdown_choices(),
                                multiselect=True,
                                value=[],
                                interactive=True,
                                scale=4,
                                info="Kliknite na retke u tablici za dodavanje/uklanjanje, ili birajte izravno ovdje."
                            )
                            with gr.Column(scale=2):
                                with gr.Row():
                                    btn_select_all_filtered = gr.Button("☑️ Označi sve iz pretrage", size="sm", variant="secondary")
                                    btn_clear_batch_selection = gr.Button("✖ Poništi odabir", size="sm", variant="secondary")
                                btn_open_batch_delete_modal = gr.Button("🗑️ Obriši označene osobe (0)", size="md", variant="stop", elem_classes=["btn-cyber-danger"])
                        batch_action_status = gr.Markdown("")

                    db_stats_md = gr.Markdown("")
                    db_table = gr.Dataframe(
                        headers=["ID", "Ime", "Uloga", "Broj slika", "Kvaliteta profila", "Bilješke", "Datum registracije"],
                        label="Popis osoba (Kliknite na bilo koji redak za automatski odabir osobe za unos)",
                        interactive=False,
                        max_height=480
                    )
                    btn_refresh_db = gr.Button("🔄 Osvježi cijeli popis", size="sm", elem_classes=["btn-cyber-secondary"])
                    
                    # Sklopiva galerija referentnih slika i profil
                    with gr.Accordion("🖼️ Referentne slike i biometrijski profil odabrane osobe", open=True, elem_classes=["cyber-accordion"]):
                        with gr.Row():
                            manage_person_dropdown = gr.Dropdown(
                                label="Odaberite osobu za pregled, uređivanje ili brisanje",
                                choices=get_person_dropdown_choices(),
                                allow_custom_value=True,
                                scale=3
                            )
                            btn_delete_person = gr.Button("🗑️ Obriši osobu", variant="stop", scale=1)
                            
                        person_info_md = gr.Markdown("Odaberite osobu iznad ili kliknite na nju u tablici za pregled lica.")
                        
                        # Zona za naknadno uređivanje imena, prezimena i bilješki
                        with gr.Group(elem_classes=["cyber-card", "edit-person-card"]):
                            gr.Markdown("#### ✏️ Uređivanje podataka odabrane osobe (Ime, uloga i bilješke)")
                            with gr.Row():
                                edit_person_name = gr.Textbox(
                                    label="Ime i prezime",
                                    placeholder="Upišite novo ime...",
                                    scale=3
                                )
                                edit_person_role = gr.Dropdown(
                                    label="Sigurnosni status / Uloga",
                                    choices=[
                                        ("Standardna osoba / Korisnik", "standard"),
                                        ("⭐ VIP uzvanik / Važna osoba", "vip"),
                                        ("🚨 Crna lista / Nepoželjni (Zabrana)", "blacklist")
                                    ],
                                    value="standard",
                                    scale=2
                                )
                                edit_person_notes = gr.Textbox(
                                    label="Bilješke",
                                    placeholder="Upišite ili dopunite bilješku (odjel, uloga, opaske...)",
                                    lines=2,
                                    scale=3
                                )
                            with gr.Row():
                                btn_save_person_edit = gr.Button(
                                    "💾 Spremi izmjene",
                                    variant="primary",
                                    scale=2,
                                    elem_classes=["btn-cyber-primary"]
                                )
                                edit_person_status = gr.Markdown("", scale=4)

                        person_gallery = gr.Gallery(
                            label="Spremljeni uzorci lica",
                            columns=5,
                            height=170,
                            allow_preview=False
                        )

                        with gr.Row():
                            sample_delete_dropdown = gr.Dropdown(
                                label="Odaberite sliku za brisanje",
                                choices=[],
                                allow_custom_value=True,
                                scale=3
                            )
                            btn_delete_sample = gr.Button("🗑️ Obriši odabranu sliku", variant="stop", scale=1)
                        sample_action_status = gr.Markdown("")

            # ---------------- DANGER ZONE (OPASNA ZONA) NA DNU BAZE OSOBA ----------------
            with gr.Accordion("🚨 Opasna zona: Upravljanje cjelokupnim brisanjem baze (Danger Zone)", open=False, elem_classes=["danger-zone-accordion"]):
                with gr.Group(elem_classes=["danger-zone-card"]):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=4):
                            gr.Markdown(
                                """
                                #### ⚠️ Trajno brisanje cjelokupne baze osoba
                                * **Upozorenje:** Ova radnja nepovratno uklanja **sve registrirane profile**, sve biometrijske vektore (512-D) i sve fotografije izreza lica.
                                * **Sigurnost:** Klikom na gumb otvorit će se skočni prozor s detaljnim upozorenjem i obaveznom potvrdom prije izvršavanja.
                                """
                            )
                        with gr.Column(scale=2):
                            btn_open_wipe_modal = gr.Button(
                                "🚨 Obriši cjelokupnu bazu osoba",
                                variant="stop",
                                elem_classes=["btn-cyber-danger"]
                            )
                    wipe_db_status = gr.Markdown("")

            # ---------------- SKOČNI PROZOR (MODAL) ZA POTVRDU BRISANJA ----------------
            with gr.Group(visible=False, elem_classes=["cyber-modal-overlay"]) as wipe_modal:
                with gr.Group(elem_classes=["cyber-modal-box"]):
                    gr.Markdown(
                        """
                        ## 🚨 UPOZORENJE: BRISANJE CJELOKUPNE BAZE OSOBA
                        ---
                        **Jeste li potpuno sigurni da želite trajno obrisati sve osobe iz baze podataka?**
                        
                        * ⚠️ **Ova radnja je nepovratna!**
                        * 👤 Svi registrirani profili osoba bit će uklonjeni iz baze.
                        * 🧬 Svi 512-dimenzionalni biometrijski vektori bit će trajno obrisani.
                        * 🖼️ Sve povezane fotografije izreza lica (crops) bit će uklonjene s diska.
                        * 🔄 FAISS vektorski indeks bit će u potpunosti resetiran.
                        * 🛡️ **Automatski backup:** Sustav će neposredno prije brisanja automatski kreirati ZIP arhivu u mapi `data/backups/`.
                        """
                    )
                    wipe_modal_stats = gr.Markdown("")
                    wipe_confirm_cb = gr.Checkbox(
                        label="Razumijem posljedice i izričito potvrđujem trajno brisanje cjelokupne baze podataka",
                        value=False,
                        interactive=True
                    )
                    wipe_modal_error = gr.Markdown("")
                    with gr.Row():
                        btn_cancel_wipe = gr.Button("✖️ Odustani (Zatvori prozor)", variant="secondary", scale=1)
                        btn_confirm_wipe = gr.Button("🔥 Potvrdi i trajno obriši", variant="stop", elem_classes=["btn-cyber-danger"], scale=1)

            # ---------------- SKOČNI PROZOR (MODAL) ZA POTVRDU GRUPNOG BRISANJA ----------------
            with gr.Group(visible=False, elem_classes=["cyber-modal-overlay"]) as batch_delete_modal:
                with gr.Group(elem_classes=["cyber-modal-box"]):
                    gr.Markdown(
                        """
                        ## 🗑️ POTVRDA GRUPNOG BRISANJA OSOBA
                        ---
                        **Jeste li sigurni da želite trajno obrisati sve označene osobe?**
                        
                        * ⚠️ Ova radnja nepovratno uklanja odabrane profile iz baze podataka.
                        * 🧬 Svi njihovi 512-dimenzionalni biometrijski vektori bit će trajno obrisani.
                        * 🖼️ Sve njihove povezane fotografije izreza (crops) bit će uklonjene s diska.
                        * 🔄 FAISS vektorski indeks bit će automatski ažuriran.
                        """
                    )
                    batch_delete_modal_summary = gr.Markdown("")
                    with gr.Row():
                        btn_cancel_batch_delete = gr.Button("✖️ Odustani / Zatvori", variant="secondary", scale=1)
                        btn_confirm_batch_delete = gr.Button("🔥 Potvrdi i obriši", variant="stop", elem_classes=["btn-cyber-danger"], scale=1)

        # ------------------ TAB 3: SPREMLJENI KADROVI & VIDEO SNIMKE ------------------
        with gr.TabItem("📁 Spremljeni Kadrovi i Video Snimke") as tab_saved_media:
            with gr.Tabs():
                # POD-TAB 1: SLIKE KADROVA (SNAPSHOTS)
                with gr.TabItem("🖼️ Slike Kadrova (Snapshots)"):
                    with gr.Row():
                        with gr.Column(scale=3):
                            snap_init_g, snap_init_t, snap_init_info, snap_init_dd, snap_init_prev, snap_init_desc = get_snapshots_ui_data()
                            snapshots_info_md = gr.Markdown(snap_init_info)
                        with gr.Column(scale=1):
                            with gr.Row():
                                btn_open_snapshots_folder = gr.Button("📂 Otvori mapu sa slikama", variant="primary")
                                btn_refresh_snapshots = gr.Button("🔄 Osvježi slike", variant="secondary")

                    with gr.Row():
                        # Lijevi stupac: Galerija sličica (male ikone)
                        with gr.Column(scale=3):
                            gr.Markdown("### 🖼️ Sličice spremljenih kadrova *(kliknite na sliku za odabir)*")
                            snapshots_gallery = gr.Gallery(
                                label="Spremljeni kadrovi",
                                columns=4,
                                rows=2,
                                height=420,
                                allow_preview=False,
                                value=snap_init_g
                            )
                            
                            gr.Markdown("### 📋 Popis datoteka na disku")
                            snapshots_table = gr.Dataframe(
                                headers=["Naziv datoteke", "Datum i vrijeme snimanja", "Veličina"],
                                label="Popis snimaka",
                                interactive=False,
                                value=snap_init_t
                            )

                        # Desni stupac: Detalji odabrane snimke i akcije
                        with gr.Column(scale=2):
                            gr.Markdown("### 🔍 Pregled odabranog kadra i akcije")
                            selected_snap_dropdown = gr.Dropdown(
                                label="Odaberite snimku za analizu ili brisanje:",
                                choices=[r[0] for r in snap_init_t],
                                value=snap_init_t[0][0] if snap_init_t else None,
                                interactive=True
                            )
                            selected_snap_preview = gr.Image(
                                type="filepath",
                                label="Prikaz kadra (Annotated / HUD)",
                                value=snap_init_prev,
                                height=280
                            )
                            selected_snap_info = gr.Markdown(snap_init_desc)
                            
                            with gr.Row():
                                btn_send_to_rec = gr.Button("🔍 Pošalji na prepoznavanje lica", variant="primary")
                                btn_delete_snap = gr.Button("🗑️ Obriši sliku", variant="stop")
                            snap_action_status = gr.Markdown("")

                    with gr.Accordion("⚙️ Postavke lokacije spremanja snimaka (Snapshot Folder)", open=False):
                        gr.Markdown(
                            """
                            Ovdje možete promijeniti mapu u koju se automatski spremaju kadrovi kada u live video prozoru pritisnete tipku **S**.
                            Zadana mapa je unutar aplikacije (`data/snapshots`), no možete odabrati bilo koju mapu na vašem disku (npr. `D:\\Nadzor\\Kadrovi`).
                            """
                        )
                        with gr.Row():
                            custom_snap_dir_input = gr.Textbox(
                                label="Putanja do mape za spremanje snimaka na računalu",
                                value=config.get_snapshot_dir(),
                                placeholder="npr. D:\\Nadzor\\Snimke ili C:\\UniFace_Kadrovi",
                                scale=3
                            )
                            btn_save_snap_dir = gr.Button("💾 Spremi novu mapu", variant="primary", scale=1)
                            btn_reset_snap_dir = gr.Button("🔄 Vrati na zadano (data/snapshots)", variant="secondary", scale=1)
                        snap_dir_status_md = gr.Markdown("")

                # POD-TAB 2: NVR VIDEO SNIMKE
                with gr.TabItem("📹 NVR Video Snimke (Video Zapisi)"):
                    nvr_stat_init, nvr_table_init, nvr_vid_init, nvr_lbl_init = get_nvr_archive_ui_data()
                    with gr.Row():
                        with gr.Column(scale=3):
                            nvr_storage_status_md = gr.Markdown(nvr_stat_init)
                        with gr.Column(scale=1):
                            with gr.Row():
                                btn_open_nvr_folder = gr.Button("📂 Otvori mapu sa snimkama", variant="primary")
                                btn_refresh_nvr_tab = gr.Button("🔄 Osvježi video snimke", variant="secondary")

                    with gr.Row():
                        # Lijevi stupac: Popis MP4 video segmenata
                        with gr.Column(scale=3):
                            gr.Markdown("### 📼 Popis snimljenih video segmenata *(kliknite redak za reprodukciju)*")
                            nvr_archive_table = gr.Dataframe(
                                headers=["Datoteka", "Kamera", "Datum", "Vrijeme", "Veličina", "Putanja"],
                                value=nvr_table_init,
                                interactive=False,
                                label="Popis video segmenata na disku",
                                max_height=480
                            )

                        # Desni stupac: Video Player i akcije
                        with gr.Column(scale=2):
                            gr.Markdown("### 🎬 Video Reprodukcija i Detalji")
                            nvr_selected_info_md = gr.Markdown(nvr_lbl_init)
                            nvr_video_preview = gr.Video(
                                value=nvr_vid_init,
                                label="▶️ Reprodukcija video segmenta",
                                interactive=False,
                                height=300
                            )
                            with gr.Row():
                                btn_open_video_player = gr.Button("🖥️ Otvori u vanjskom playeru", variant="secondary")
                                btn_delete_video_seg = gr.Button("🗑️ Obriši ovu snimku", variant="stop")
                            nvr_action_status_md = gr.Markdown("")

        # ------------------ TAB 4: DNEVNIK PROLAZAKA (EVIDENCIJA) ------------------
        with gr.TabItem("📋 Dnevnik Prolazaka (Evidencija)") as tab_events:
            ev_stats_init, ev_table_init, ev_cam_init, ev_crop_init, ev_info_init, ev_video_init, ev_id_init, ev_del_btn_init = get_events_ui_data()
            events_stats_md = gr.Markdown(ev_stats_init)
            selected_event_id_state = gr.State(ev_id_init)
            
            with gr.Row():
                with gr.Column(scale=3):
                    events_search_input = gr.Textbox(
                        label="🔍 Filtriraj evidenciju po imenu osobe",
                        placeholder="Upišite ime za brzu pretragu...",
                        interactive=True
                    )
                with gr.Column(scale=1):
                    with gr.Row():
                        btn_refresh_events = gr.Button("🔄 Osvježi", variant="secondary")
                        btn_clear_events = gr.Button("🗑️ Očisti dnevnik", variant="stop")
            
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### 📜 Kronološki popis detekcija i prolazaka (kliknite redak za prikaz)")
                    events_table = gr.Dataframe(
                        headers=["ID", "Datum i Vrijeme", "Prepoznata Osoba", "Sličnost", "Izvor / Kamera"],
                        value=ev_table_init,
                        interactive=False,
                        label="Zabilježeni prolasci",
                        max_height=520
                    )
                with gr.Column(scale=2):
                    gr.Markdown("### 📷 Prikaz kadra s kamere i detalji prolaska")
                    event_camera_preview = gr.Image(
                        value=ev_cam_init,
                        label="📹 Kadar kamere u trenutku prepoznavanja",
                        interactive=False
                    )
                    with gr.Row():
                        event_crop_preview = gr.Image(
                            value=ev_crop_init,
                            label="👤 Izrezano lice",
                            interactive=False,
                            width=140,
                            height=140
                        )
                        event_details_md = gr.Markdown(ev_info_init)
                    with gr.Row():
                        btn_delete_single_event = gr.Button(
                            value=f"🗑️ Obriši ovaj prolazak (#{ev_id_init})" if ev_id_init else "🗑️ Obriši odabrani prolazak",
                            variant="stop",
                            visible=bool(ev_id_init),
                            scale=1
                        )
                    event_video_player = gr.Video(
                        value=ev_video_init.get("value"),
                        visible=ev_video_init.get("visible", False),
                        label="📹 NVR Video Snimka (Trenutak detekcije)",
                        interactive=False
                    )
            
            with gr.Row():
                btn_export_events_csv = gr.Button("📥 Izvezi cijeli dnevnik u CSV (Excel)", variant="primary", scale=1)
                events_export_file = gr.File(label="Preuzmi izvezenu CSV datoteku", visible=False, scale=2)
            events_status_md = gr.Markdown("")

            # ---------------- MODAL ZA POTVRDU BRISANJA CJELOKUPNOG DNEVNIKA ----------------
            with gr.Group(visible=False, elem_classes=["cyber-modal-overlay"]) as clear_events_modal:
                with gr.Group(elem_classes=["cyber-modal-box"]):
                    gr.Markdown(
                        """
                        ## 🚨 UPOZORENJE: BRISANJE CJELOKUPNOG DNEVNIKA PROLAZAKA
                        ---
                        **Jeste li potpuno sigurni da želite trajno obrisati sve zabilježene prolaske iz evidencije?**
                        
                        * ⚠️ **Ova radnja je nepovratna!**
                        * 📜 Svi povijesni zapisi detekcija i prolazaka bit će uklonjeni iz baze podataka.
                        * 📷 Sve povezane fotografije kadrova i izreza lica vezane uz evidenciju bit će obrisane s diska.
                        * 👥 Registrirane osobe i njihovi biometrijski profili u bazi **ostaju netaknuti**.
                        """
                    )
                    clear_events_modal_stats = gr.Markdown("")
                    clear_events_confirm_cb = gr.Checkbox(
                        label="Razumijem posljedice i izričito potvrđujem trajno brisanje cjelokupnog dnevnika prolazaka",
                        value=False,
                        interactive=True
                    )
                    clear_events_modal_error = gr.Markdown("")
                    with gr.Row():
                        btn_cancel_clear_events = gr.Button("✖️ Odustani / Zatvori", variant="secondary", scale=1)
                        btn_confirm_clear_events = gr.Button("🔥 Potvrdi i obriši cijeli dnevnik", variant="stop", elem_classes=["btn-cyber-danger"], scale=1)

            # ---------------- MODAL ZA POTVRDU BRISANJA POJEDINAČNOG PROLASKA ----------------
            with gr.Group(visible=False, elem_classes=["cyber-modal-overlay"]) as single_event_delete_modal:
                with gr.Group(elem_classes=["cyber-modal-box"]):
                    gr.Markdown(
                        """
                        ## 🗑️ POTVRDA BRISANJA ODABRANOG PROLASKA
                        ---
                        **Jeste li sigurni da želite obrisati ovaj zabilježeni prolazak iz evidencije?**
                        """
                    )
                    single_event_delete_summary = gr.Markdown("")
                    with gr.Row():
                        btn_cancel_single_delete = gr.Button("✖️ Odustani", variant="secondary", scale=1)
                        btn_confirm_single_delete = gr.Button("🗑️ Obriši ovaj zapis", variant="stop", elem_classes=["btn-cyber-danger"], scale=1)

        # ------------------ TAB 5: PAMETNI SORTER FOTOGRAFIJA ------------------
        with gr.TabItem("📸 Pametni Sorter Fotografija") as tab_photo_sorter:
            gr.Markdown(
                """
                ## 📸 Pametni Sorter Fotografija — Automatsko razvrstavanje po osobama
                *Univerzalni biometrijski modul za automatsko razvrstavanje velikih mapa i arhiva fotografija (događaji, konferencije, natjecanja, portreti, poslovni i privatni albumi) po prepoznatim osobama uz instantno povezivanje (Windows Hardlink - 0 MB dodatnog zauzeća diska).*
                """
            )
            
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Group():
                        gr.Markdown("### 📂 Odabir mapa i ulaznih fotografija")
                        with gr.Row():
                            sorter_input_folder = gr.Textbox(
                                label="📁 Izvorna mapa s fotografijama",
                                placeholder="Kliknite 'Odaberi mapu...' ili upišite punu putanju...",
                                scale=4
                            )
                            btn_browse_input_folder = gr.Button("📂 Odaberi mapu...", scale=1, variant="primary")
                            btn_check_input_folder = gr.Button("🔍 Provjeri", scale=1, variant="secondary")
                        
                        sorter_folder_info_md = gr.Markdown("💡 *Kliknite 'Odaberi mapu...' za brzo pronalaženje ili upišite putanju.*")
                        
                        with gr.Row():
                            sorter_output_folder = gr.Textbox(
                                label="📂 Odredišna mapa za sortirane fotografije",
                                placeholder="Zadano: automatski predložena upisiva lokacija",
                                info="Mape za prepoznate osobe, zajedničke kadrove, grupe i fotografije bez lica kreirat će se unutar ove lokacije.",
                                scale=4
                            )
                            btn_browse_output_folder = gr.Button("📂 Promijeni odredište...", scale=1, variant="secondary")

                    with gr.Group():
                        gr.Markdown("### ⚙️ Postavke biometrijskog razvrstavanja")
                        sorter_target_persons = gr.Dropdown(
                            label="👥 Odaberite ciljane osobe za sortiranje (ostavite prazno za sve osobe iz baze)",
                            choices=get_sorter_person_choices(),
                            multiselect=True,
                            interactive=True,
                            info="Odaberite specifične osobe koje želite izdvojiti ili ostavite prazno za automatsko sortiranje svih osoba iz baze."
                        )
                        
                        with gr.Row():
                            sorter_similarity = gr.Slider(
                                minimum=0.35,
                                maximum=0.75,
                                value=0.48,
                                step=0.01,
                                label="🎯 Biometrijski prag sličnosti (Threshold)",
                                info="0.48 je optimalna točnost. Niže = više ulova pri lošijem kutu; Više = maksimalna sigurnost."
                            )
                            sorter_resolution = gr.Dropdown(
                                label="⚡ Brzina / Rezolucija analize",
                                choices=[
                                    "Brzo (1280px - za starija računala)",
                                    "Uravnoteženo (1600px - preporučeno)",
                                    "Maksimalna točnost (2048px - za velike grupne kadrove)"
                                ],
                                value="Uravnoteženo (1600px - preporučeno)",
                                info="Originalna datoteka se nikada ne dira niti komprimira."
                            )

                        sorter_action_mode = gr.Radio(
                            label="💾 Način prijenosa datoteka u sortirane mape",
                            choices=[
                                "⚡ Windows Hardlink (preporučeno - instantno, troši 0 MB dodatnog diska)",
                                "📋 Kopiraj datoteke (stvara fizičke kopije na disku)",
                                "🚚 Premjesti datoteke (Move)"
                            ],
                            value="⚡ Windows Hardlink (preporučeno - instantno, troši 0 MB dodatnog diska)",
                            info="Hardlink omogućuje da slika bude u više mapa istovremeno bez trošenja dodatnih gigabajta!"
                        )

                        gr.Markdown("#### 🏷️ Pametna organizacija posebnih mapa")
                        with gr.Row():
                            sorter_enable_combo = gr.Checkbox(
                                label="👥 Kreiraj mapu 'Zajedno_Ciljane_Osobe' (kad je 2+ ciljanih osoba na istoj slici)",
                                value=True
                            )
                            sorter_enable_group = gr.Checkbox(
                                label="👨‍👩‍👧‍👦 Kreiraj mapu 'Grupne_Fotografije'",
                                value=True
                            )
                            sorter_group_min = gr.Number(
                                label="Minimalno lica za grupu",
                                value=4,
                                precision=0,
                                scale=1
                            )
                        with gr.Row():
                            sorter_enable_noface = gr.Checkbox(
                                label="🖼️ Izdvoji fotografije bez lica u 'Fotografije_Bez_Lica' (objekti, arhitektura, pejzaži, detalji)",
                                value=True
                            )
                            sorter_enable_unregistered = gr.Checkbox(
                                label="👤 Izdvoji osobe koje nisu u bazi u 'Neregistrirana_Lica'",
                                value=False
                            )

                    with gr.Row():
                        btn_start_sorter = gr.Button("🚀 Pokreni automatsko sortiranje", variant="primary", scale=3)
                        btn_cancel_sorter = gr.Button("🛑 Zaustavi obradu", variant="stop", scale=1, interactive=False)
                        btn_open_sorter_dir = gr.Button("📂 Otvori mapu u Exploreru", variant="secondary", scale=2)

                with gr.Column(scale=4):
                    with gr.Group():
                        gr.Markdown("### 📈 Status obrade i metrike u stvarnom vremenu")
                        sorter_status_md = gr.Markdown("⏳ *Sustav je spreman. Odaberite mapu s fotografijama i pokrenite sortiranje.*")
                        sorter_stats_breakdown_md = gr.Markdown("")
                        sorter_export_csv_file = gr.File(label="📥 Preuzmi CSV izvještaj sortiranja (Excel)", visible=False)

        # ------------------ TAB 6: O SUSTAVU & SIGURNOSNA KOPIJA ------------------
        with gr.TabItem("ℹ️ O Sustavu i Sigurnosna Kopija"):
            with gr.Row():
                with gr.Column(scale=1):
                    hw_accel_badge = gr.HTML(hardware.get_hardware_acceleration_badge_html())
                    system_info_md = gr.Markdown(hardware.get_system_report_markdown(DATA_DIR))
                    btn_refresh_sysinfo = gr.Button("🔄 Osvježi podatke o sustavu", size="sm")
                    
                    gr.Markdown("---")
                    gr.Markdown(
                        """
                        ### 🎯 Sustav za maksimalnu točnost (Centroid Multi-Sample)
                        * **Sintetizirani biometrijski profil (Centroid):** Kada za osobu unesete više slika (npr. 2, 3 ili 4 različita kuta), sustav spaja njihove 512-dimenzionalne vektore u optimalni 'središnji' model osobe.
                        * **Hibridno bodovanje:** Usporedba uzima u obzir i najbolji kut i cjelokupni centroid, čime se eliminiraju lažni pozitivni rezultati i znatno povećava točnost na grupnim slikama s otežanim osvjetljenjem.
                        * **Sigurnosna margina:** Prikazuje razliku u postotku između najizglednijeg kandidata i drugog najboljeg, što daje jasan uvid u pouzdanost prepoznavanja.
                        """
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### 📦 Sigurnosna kopija i arhiviranje baze")
                    gr.Markdown("Izvezite cjelokupnu bazu podataka (`database.db`), biometrijske vektore i fotografije lica u ZIP arhivu ili obnovite bazu iz postojeće arhive.")
                    
                    with gr.Group():
                        gr.Markdown("#### 💾 Izvoz sigurnosne kopije (Export)")
                        btn_export_backup = gr.Button("📦 Kreiraj i preuzmi sigurnosnu kopiju (ZIP)", variant="primary")
                        backup_download_file = gr.File(label="Preuzmite ZIP arhivu", interactive=False)
                        btn_open_backup_folder = gr.Button("📂 Otvori mapu sa sigurnosnim kopijama (Windows Explorer)", variant="secondary")
                        backup_export_status = gr.Markdown("")
                        
                    with gr.Group():
                        gr.Markdown("#### 📥 Vraćanje sigurnosne kopije (Restore / Import)")
                        backup_upload_file = gr.File(label="Prenesite ZIP arhivu za uvoz", file_types=[".zip"], file_count="single")
                        btn_import_backup = gr.Button("⚠️ Uvezi arhivu i obnovi bazu", variant="stop")
                        backup_import_status = gr.Markdown("")

                    with gr.Group(elem_classes=["cyber-card", "retention-card"]):
                        gr.Markdown("#### 🛡️ GDPR Upravljanje podacima i automatska rotacija (Data Retention)")
                        gr.Markdown(
                            "Uskladite pohranu sa zakonskim načelom smanjenja količine podataka (*Storage limitation*, GDPR Čl. 5(1)(e)). "
                            "Automatski ili ručno očistite zapise prolazaka, JPEG kadrove i video snimke starije od zadanog razdoblja."
                        )
                        init_ret_days = config.get_retention_days()
                        ret_choices = [
                            "15 dana (Preporučeno za video nadzor / AZOP)",
                            "30 dana (Standardno poslovno čuvanje)",
                            "60 dana",
                            "90 dana",
                            "Trajno (Bez automatskog brisanja)"
                        ]
                        init_choice = ret_choices[1]
                        if init_ret_days == 15:
                            init_choice = ret_choices[0]
                        elif init_ret_days == 30:
                            init_choice = ret_choices[1]
                        elif init_ret_days == 60:
                            init_choice = ret_choices[2]
                        elif init_ret_days == 90:
                            init_choice = ret_choices[3]
                        elif init_ret_days <= 0:
                            init_choice = ret_choices[4]

                        with gr.Row():
                            retention_period_radio = gr.Radio(
                                label="Politika zadržavanja podataka (Odaberite rok automatske rotacije)",
                                choices=ret_choices,
                                value=init_choice,
                                interactive=True,
                                elem_classes=["retention-radio-group"],
                                scale=3
                            )
                            with gr.Column(scale=2):
                                btn_run_retention = gr.Button("🧹 Očisti stare podatke odmah", variant="secondary", elem_classes=["btn-cyber-primary"])
                                retention_status_md = gr.Markdown("")

                    with gr.Group(elem_classes=["cyber-card", "telegram-card"]):
                        gr.Markdown("#### 📲 Telegram Instant Sigurnosne Obavijesti")
                        gr.Markdown(
                            "Povežite Argusface sa svojim Telegram računom ili sigurnosnim kanalom za **trenutne obavijesti s fotografijom lica** "
                            "kada se detektira osoba s Crne liste, dolazak VIP uzvanika ili pokušaj lažiranja (Anti-Spoofing napad s mobitela/slike)."
                        )
                        init_tg = config.get_telegram_config()
                        telegram_enabled_chk = gr.Checkbox(
                            label="🔔 Omogući slanje obavijesti na Telegram",
                            value=init_tg["enabled"],
                            interactive=True
                        )
                        with gr.Row():
                            telegram_token_input = gr.Textbox(
                                label="Telegram Bot Token",
                                placeholder="npr. 7123456789:AAHk-...",
                                value=init_tg["bot_token"],
                                type="password",
                                scale=3,
                                info="Token dobiven od @BotFather bota"
                            )
                            telegram_chat_id_input = gr.Textbox(
                                label="Glavni Telegram Chat ID (ili ID grupe/kanala)",
                                placeholder="npr. 123456789 ili -100123456789 (moguće više odvojeno zarezom)",
                                value=init_tg["chat_id"],
                                scale=2,
                                info="Zadana adresa za obavijesti"
                            )
                        with gr.Accordion("🎯 Selektivno usmjeravanje kanala (Opcije za Zaštitare i Recepciju)", open=False, elem_classes=["cyber-accordion"]):
                            gr.Markdown(
                                "*Ostavite prazno ako svi alarmi idu u glavni Chat ID iznad.* "
                                "Upišite poseban Chat ID grupe ili korisnika kako bi samo određeni odjel primao specifične alarme."
                            )
                            with gr.Row():
                                telegram_chat_id_sec_input = gr.Textbox(
                                    label="Chat ID za Službu Osiguranja / Zaštitare",
                                    placeholder="npr. -10012345678 (Za Crnu listu i Anti-Spoof)",
                                    value=init_tg.get("chat_id_security", ""),
                                    scale=1
                                )
                                telegram_chat_id_vip_input = gr.Textbox(
                                    label="Chat ID za Recepciju / Protokol",
                                    placeholder="npr. -10098765432 (Za VIP dolaske)",
                                    value=init_tg.get("chat_id_vip", ""),
                                    scale=1
                                )
                        with gr.Row():
                            telegram_notify_blacklist_chk = gr.Checkbox(
                                label="🚨 Crna lista (Nepoželjni)",
                                value=init_tg["notify_blacklist"],
                                interactive=True
                            )
                            telegram_notify_vip_chk = gr.Checkbox(
                                label="⭐ VIP dolazak",
                                value=init_tg["notify_vip"],
                                interactive=True
                            )
                            telegram_notify_spoof_chk = gr.Checkbox(
                                label="🛡️ Pokušaj lažiranja (Anti-Spoof)",
                                value=init_tg["notify_spoof"],
                                interactive=True
                            )
                            telegram_notify_unknown_chk = gr.Checkbox(
                                label="❓ Nepoznata lica",
                                value=init_tg["notify_unknown"],
                                interactive=True
                            )
                        telegram_cooldown_slider = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=init_tg["cooldown_min"],
                            step=1,
                            label="Period hlađenja / Cooldown (minute)",
                            info="Spriječava višestruko slanje poruka za istu osobu unutar zadanog broja minuta"
                        )
                        with gr.Row():
                            btn_save_telegram_config = gr.Button("💾 Spremi postavke obavijesti", variant="primary", scale=2, elem_classes=["btn-cyber-primary"])
                            btn_test_telegram = gr.Button("📨 Pošalji testnu poruku", variant="secondary", scale=2, elem_classes=["btn-cyber-secondary"])
                        telegram_status_md = gr.Markdown("")

                    with gr.Group(elem_classes=["cyber-card", "email-card"]):
                        gr.Markdown("#### 📧 E-mail Sigurnosna Upozorenja (SMTP)")
                        gr.Markdown(
                            "Omogućite automatsko slanje formalnih **sigurnosnih e-mail izvještaja s priloženom slikom lica** "
                            "na službene e-mail adrese uprave, voditelja osiguranja ili vanjske zaštitarske službe."
                        )
                        init_em = config.get_email_config()
                        email_enabled_chk = gr.Checkbox(
                            label="🔔 Omogući slanje sigurnosnih e-mail upozorenja",
                            value=init_em["enabled"],
                            interactive=True
                        )
                        with gr.Row():
                            email_smtp_host = gr.Textbox(
                                label="SMTP Poslužitelj (Host)",
                                placeholder="npr. smtp.gmail.com ili smtp.office365.com",
                                value=init_em["smtp_host"],
                                scale=3
                            )
                            email_smtp_port = gr.Number(
                                label="Port",
                                value=init_em["smtp_port"],
                                precision=0,
                                scale=1
                            )
                            enc_default = "STARTTLS (Port 587)" if init_em["use_tls"] else ("SSL / TLS (Port 465)" if init_em["use_ssl"] else "Bez enkripcije")
                            email_encryption_radio = gr.Radio(
                                label="Enkripcija",
                                choices=["STARTTLS (Port 587)", "SSL / TLS (Port 465)", "Bez enkripcije"],
                                value=enc_default,
                                scale=2
                            )
                        with gr.Row():
                            email_sender = gr.Textbox(
                                label="Pošiljatelj (Korisničko ime / Email)",
                                placeholder="npr. argusface.sigurnost@firma.hr",
                                value=init_em["sender"],
                                scale=3
                            )
                            email_password = gr.Textbox(
                                label="Lozinka / App Password",
                                type="password",
                                placeholder="Lozinka ili Google App Password",
                                value=init_em["password"],
                                scale=3
                            )
                        email_recipients = gr.Textbox(
                            label="Primatelji upozorenja (jedan ili više e-mailova odvojenih zarezom)",
                            placeholder="npr. zastita@firma.hr, voditelj.osiguranja@firma.hr, direktor@firma.hr",
                            value=init_em["recipients"],
                            info="Sve navedene adrese primit će formatiran HTML e-mail s detaljima i slikom incidenta"
                        )
                        with gr.Row():
                            email_notify_blacklist_chk = gr.Checkbox(
                                label="🚨 Crna lista (Nepoželjni)",
                                value=init_em["notify_blacklist"],
                                interactive=True
                            )
                            email_notify_spoof_chk = gr.Checkbox(
                                label="🛡️ Pokušaj lažiranja (Anti-Spoof)",
                                value=init_em["notify_spoof"],
                                interactive=True
                            )
                            email_notify_vip_chk = gr.Checkbox(
                                label="⭐ VIP dolazak",
                                value=init_em["notify_vip"],
                                interactive=True
                            )
                        email_cooldown_slider = gr.Slider(
                            minimum=1,
                            maximum=60,
                            value=init_em["cooldown_min"],
                            step=1,
                            label="Period hlađenja / Cooldown (minute)",
                            info="Spriječava preopterećenje pretinca uzastopnim porukama za istu osobu"
                        )
                        with gr.Row():
                            btn_save_email_config = gr.Button("💾 Spremi E-mail postavke", variant="primary", scale=2, elem_classes=["btn-cyber-primary"])
                            btn_test_email = gr.Button("📨 Pošalji testni E-mail", variant="secondary", scale=2, elem_classes=["btn-cyber-secondary"])
                        email_status_md = gr.Markdown("")

            gr.Markdown("---")
            with gr.Accordion("⚖️ Pravne napomene, licence i regulatorna usklađenost (GDPR & EU AI Act)", open=True, elem_classes=["cyber-accordion"]):
                gr.Markdown(
                    """
                    ### 📜 Komercijalne licence AI modela i tehnološkog stoga
                    ArgusFace Studio je konfiguriran s fokusom na **100% legalnu, čistu i sigurnu komercijalnu primjenu** bez akademskih ograničenja ili skrivenih naknada:
                    * **Detekcija lica (RetinaFace):** Licencirano pod **MIT licencom**. Omogućuje ultra-brzo i robusno pronalaženje lica i ključnih točaka.
                    * **Prepoznavanje lica (EdgeFace BASE):** Licencirano pod **BSD-3-Clause licencom** (*Idiap Research Institute*, Švicarska). Generira 512-dimenzionalne normalizirane biometrijske vektore uz točnost od **99.83%** na LFW standardu.
                    * **Inženjerski i grafički stog:** Razvijeno u Pythonu uz **OpenCV** i **ONNX Runtime** (oba pod **Apache 2.0** licencom).
                    * **Distribucija i komercijalizacija:** Cjelokupni stog modela i biblioteka slobodan je za komercijalnu prodaju, instalaciju kod klijenata i licenciranje trećim stranama.

                    ---

                    ### 🛡️ Zaštita osobnih i biometrijskih podataka (GDPR usklađenost)
                    * **100% Lokalna obrada (Edge / On-Premise):** Sva obrada slika, video tokova i biometrijskih vektora odvija se isključivo na lokalnom računalu. Niti jedan podatak, slika ili vektor nikada se ne prenosi na vanjske poslužitelje ili Cloud.
                    * **Biometrijski podaci (Članak 9. GDPR-a):** 512-D vektori tretiraju se kao biometrijski podaci. Korisnik/vlasnik sustava odgovoran je za zakonitost prikupljanja (odgovarajuća privola ili zakonska pravna osnova).
                    * **Pravo na zaborav:** Sustav omogućuje trajno, nepovratno brisanje pojedinačnih uzoraka ili cjelokupnih profila osoba iz baze podataka jednim klikom.

                    ---

                    ### 🇪🇺 Usklađenost sa Zakonom o umjetnoj inteligenciji (EU AI Act)
                    * **Klasifikacija sustava:** ArgusFace Studio namijenjen je za privatnu i internu poslovnu upotrebu (npr. organizacija arhive fotografija, evidencija prisutnosti i verifikacija u kontroliranim privatnim prostorima).
                    * **Ograničenje namjene:** Sustav nije namijenjen niti licenciran za neovlašteno masovno biometrijsko profiliranje ili prepoznavanje u stvarnom vremenu na javno dostupnim površinama.
                    """
                )


    # ------------------ EVENT HANDLERS ------------------
    batch_naming_mode.change(
        fn=lambda m: gr.update(visible=(m == "Sve slike pripadaju istoj osobi")),
        inputs=[batch_naming_mode],
        outputs=[batch_single_name]
    )
    
    # 1. Image uploaded in Single Enroll -> extracts faces and renders visual indicators
    single_img_input.change(
        fn=on_single_image_uploaded,
        inputs=[single_img_input, single_name_input, single_enroll_state],
        outputs=[
            single_annotated_preview,
            single_crops_gallery,
            single_face_selector,
            single_preview_crop,
            single_preview_info,
            btn_save_single,
            single_enroll_state
        ]
    )
    
    # 2. Click directly on photo
    single_annotated_preview.select(
        fn=on_single_image_click,
        inputs=[single_enroll_state],
        outputs=[single_annotated_preview, single_preview_crop, single_preview_info, single_face_selector, single_enroll_state]
    )
    
    # 3. Radio button selector change
    single_face_selector.change(
        fn=on_radio_face_change,
        inputs=[single_face_selector, single_enroll_state],
        outputs=[single_annotated_preview, single_preview_crop, single_preview_info, single_face_selector, single_enroll_state]
    )
    
    # 4. Gallery thumbnail click selector
    single_crops_gallery.select(
        fn=on_crop_gallery_select,
        inputs=[single_enroll_state],
        outputs=[single_annotated_preview, single_preview_crop, single_preview_info, single_face_selector, single_enroll_state]
    )
    
    # 5. Save single face -> advances to next unsaved face & keeps person selected for fluid multi-photo enrollment
    btn_save_single.click(
        fn=save_single_person,
        inputs=[single_name_input, single_notes_input, single_face_selector, single_role_input, single_enroll_state],
        outputs=[
            single_save_status,
            manage_person_dropdown,
            existing_person_picker,
            db_table,
            db_stats_md,
            single_annotated_preview,
            single_preview_crop,
            single_preview_info,
            single_face_selector,
            single_name_input,
            single_notes_input,
            btn_save_single,
            selected_person_avatar,
            edit_person_name,
            edit_person_notes,
            single_crops_gallery,
            single_enroll_state,
            edit_person_role,
            single_role_input
        ]
    ).then(
        fn=view_person_details,
        inputs=[manage_person_dropdown],
        outputs=[person_gallery, person_info_md, sample_delete_dropdown]
    )

    # 5b. Update save button label dynamically when typing name and leaving input
    def on_name_input_blur(name_val):
        n = str(name_val or "").strip()
        if n:
            return f"💾 Spremi dodatno lice za: {n}"
        return "💾 Spremi odabrano lice u bazu"

    single_name_input.blur(
        fn=on_name_input_blur,
        inputs=[single_name_input],
        outputs=[btn_save_single]
    )
    
    # 6. Clear single form
    btn_clear_form.click(
        fn=clear_single_form,
        outputs=[
            single_name_input, single_notes_input, single_img_input,
            single_annotated_preview, single_crops_gallery,
            single_face_selector, single_preview_crop, single_preview_info,
            btn_save_single, single_save_status,
            existing_person_picker, single_enroll_state,
            selected_person_avatar, single_role_input
        ]
    )
    
    # 7. Quick existing person picker in enrollment form
    existing_person_picker.change(
        fn=on_existing_person_picked,
        inputs=[existing_person_picker],
        outputs=[
            single_name_input, single_notes_input, btn_save_single, single_save_status,
            person_gallery, person_info_md, sample_delete_dropdown, manage_person_dropdown,
            selected_person_avatar,
            edit_person_name, edit_person_notes, edit_person_status,
            single_role_input, edit_person_role
        ],
        show_progress="hidden"
    )
    
    # 8. Real-time search in table
    table_search_input.change(
        fn=on_table_search_changed,
        inputs=[table_search_input],
        outputs=[db_table, db_stats_md, manage_person_dropdown]
    )
    
    btn_clear_table_search.click(
        fn=on_table_search_clear,
        outputs=[table_search_input, db_table, db_stats_md, manage_person_dropdown]
    )
    
    btn_batch_enroll.click(
        fn=batch_enroll_files,
        inputs=[batch_naming_mode, batch_single_name, batch_files_input],
        outputs=[batch_status_md, manage_person_dropdown, existing_person_picker, db_table, db_stats_md]
    )
    
    def _blur_flags(blur_mode: str):
        """Convert blur_mode_radio value to (blur_unknown, blur_all) booleans."""
        blur_unknown = blur_mode == "🔒 Zamuti nepoznata lica"
        blur_all     = blur_mode == "🛡️ Zamuti sva lica (GDPR)"
        return blur_unknown, blur_all

    def on_analysis_param_change(image, threshold, draw_landmarks, blur_mode, show_all_faces, rec_faces):
        if image is None:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        cached = None
        if rec_faces and len(rec_faces) > 0:
            cached = [r["raw_face"] for r in rec_faces if isinstance(r, dict) and "raw_face" in r]
            if len(cached) != len(rec_faces):
                cached = None
        blur_unknown, blur_all = _blur_flags(blur_mode or "")
        return recognize_faces(
            image, threshold, draw_landmarks, blur_unknown,
            blur_all=blur_all, show_all_faces=show_all_faces, cached_faces=cached
        )

    def on_cards_filter_toggle(rec_faces, show_all):
        if not rec_faces:
            return generate_detection_cards_html([], show_all_faces=show_all)
        return generate_detection_cards_html(rec_faces, show_all_faces=show_all)

    def on_recognize_action(image, threshold, draw_landmarks, blur_mode, show_all_faces):
        blur_unknown, blur_all = _blur_flags(blur_mode or "")
        return recognize_faces(
            image, threshold, draw_landmarks, blur_unknown,
            blur_all=blur_all, show_all_faces=show_all_faces, cached_faces=None
        )

    rec_inputs   = [input_img, threshold_slider, landmarks_chk, blur_mode_radio, cards_show_all_chk]
    rec_outputs  = [annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state, detection_cards_html]
    param_inputs = [input_img, threshold_slider, landmarks_chk, blur_mode_radio, cards_show_all_chk, rec_faces_state]

    btn_recognize.click(
        fn=on_recognize_action,
        inputs=rec_inputs,
        outputs=rec_outputs
    )

    input_img.upload(
        fn=on_recognize_action,
        inputs=rec_inputs,
        outputs=rec_outputs
    )

    landmarks_chk.change(
        fn=on_analysis_param_change,
        inputs=param_inputs,
        outputs=rec_outputs
    )

    blur_mode_radio.change(
        fn=on_analysis_param_change,
        inputs=param_inputs,
        outputs=rec_outputs
    )

    threshold_slider.release(
        fn=on_analysis_param_change,
        inputs=param_inputs,
        outputs=rec_outputs
    )

    cards_show_all_chk.change(
        fn=on_cards_filter_toggle,
        inputs=[rec_faces_state, cards_show_all_chk],
        outputs=[detection_cards_html]
    )

    def on_live_mode_change(m):
        is_grid = (m == "Mreža više kamera (2×2 Grid)")
        return (
            gr.update(visible=not is_grid),
            gr.update(visible=is_grid),
            gr.update(visible=not is_grid),
            gr.update(visible=is_grid)
        )

    live_mode_radio.change(
        fn=on_live_mode_change,
        inputs=[live_mode_radio],
        outputs=[single_cam_box, multi_cam_box, btn_launch_live, btn_launch_grid]
    )

    def on_cam_source_change(st):
        is_video = st in ("YouTube / Web Video", "Lokalna Video Datoteka")
        btn_label = "▶️ Pokreni Video" if is_video else "🎥 Pokreni Live Kameru"
        return (
            gr.update(visible=(st == "USB Web Kamera")),
            gr.update(visible=(st == "IP / RTSP Kamera za nadzor")),
            gr.update(visible=(st == "YouTube / Web Video")),
            gr.update(visible=(st == "Lokalna Video Datoteka")),
            gr.update(visible=is_video),
            gr.update(value=btn_label)
        )

    cam_source_type.change(
        fn=on_cam_source_change,
        inputs=[cam_source_type],
        outputs=[cam_usb_idx, cam_rtsp_url, cam_youtube_url, cam_video_file, cam_start_sec, btn_launch_live]
    )

    cam_enable_log.change(
        fn=lambda v: gr.update(visible=v),
        inputs=[cam_enable_log],
        outputs=[cam_cooldown_sec]
    )

    cam_enable_nvr.change(
        fn=lambda v: gr.update(visible=v),
        inputs=[cam_enable_nvr],
        outputs=[cam_nvr_segment_min]
    )

    btn_launch_live.click(
        fn=handle_launch_live,
        inputs=[cam_source_type, cam_usb_idx, cam_rtsp_url, cam_youtube_url, cam_video_file, cam_start_sec, cam_enable_log, cam_cooldown_sec, cam_enable_nvr, cam_nvr_segment_min, cam_enable_spoof],
        outputs=[rec_status_md]
    )

    btn_launch_grid.click(
        fn=handle_launch_multicam,
        inputs=[
            mc_c1_on, mc_c1_name, mc_c1_src,
            mc_c2_on, mc_c2_name, mc_c2_src,
            mc_c3_on, mc_c3_name, mc_c3_src,
            mc_c4_on, mc_c4_name, mc_c4_src,
            threshold_slider,
            cam_enable_log,
            cam_cooldown_sec,
            cam_enable_nvr,
            cam_nvr_segment_min
        ],
        outputs=[rec_status_md]
    )
    
    crops_gallery_out.select(
        fn=on_recognition_gallery_click,
        inputs=[rec_faces_state],
        outputs=[unknown_face_dropdown, quick_name_input]
    )
    
    btn_quick_add.click(
        fn=quick_add_face_to_db,
        inputs=[unknown_face_dropdown, quick_name_input, rec_faces_state],
        outputs=[quick_add_status, manage_person_dropdown, existing_person_picker, db_table, db_stats_md]
    )
    
    btn_refresh_db.click(
        fn=refresh_database_view,
        outputs=[db_table, db_stats_md]
    ).then(
        fn=update_both_person_dropdowns,
        outputs=[manage_person_dropdown, existing_person_picker]
    )
    
    # CLICKING ON TABLE AUTOMATICALLY INSERTS NAME AND LOADS PROFILE ON THE LEFT (OR TOGGLES MULTI-SELECT)
    db_table.select(
        fn=on_table_select,
        inputs=[db_table, multi_select_mode_cb, batch_selected_dropdown],
        outputs=[
            manage_person_dropdown, person_gallery, person_info_md, sample_delete_dropdown,
            single_name_input, single_notes_input, btn_save_single, single_save_status,
            existing_person_picker,
            selected_person_avatar,
            edit_person_name, edit_person_notes, edit_person_status,
            batch_selected_dropdown, btn_open_batch_delete_modal, batch_action_status,
            single_role_input, edit_person_role
        ],
        show_progress="hidden"
    )
    
    manage_person_dropdown.change(
        fn=on_manage_person_change,
        inputs=[manage_person_dropdown],
        outputs=[person_gallery, person_info_md, sample_delete_dropdown, edit_person_name, edit_person_notes, edit_person_status, edit_person_role]
    )

    btn_save_person_edit.click(
        fn=update_person_handler,
        inputs=[manage_person_dropdown, edit_person_name, edit_person_notes, edit_person_role],
        outputs=[
            edit_person_status,
            db_table,
            db_stats_md,
            manage_person_dropdown,
            existing_person_picker,
            single_name_input,
            single_notes_input,
            person_info_md,
            selected_person_avatar,
            btn_save_single,
            single_role_input
        ]
    )
    
    btn_delete_person.click(
        fn=delete_selected_person,
        inputs=[manage_person_dropdown],
        outputs=[
            sample_action_status,
            manage_person_dropdown,
            existing_person_picker,
            db_table,
            db_stats_md,
            person_gallery,
            person_info_md,
            edit_person_name,
            edit_person_notes,
            edit_person_status
        ]
    )
    
    btn_delete_sample.click(
        fn=delete_selected_sample,
        inputs=[sample_delete_dropdown, manage_person_dropdown],
        outputs=[sample_action_status, person_gallery, person_info_md]
    ).then(
        fn=refresh_database_view,
        outputs=[db_table, db_stats_md]
    )

    # Danger Zone - Cjelokupno brisanje baze (Modal)
    btn_open_wipe_modal.click(
        fn=handle_open_wipe_modal,
        outputs=[wipe_modal, wipe_modal_stats, wipe_confirm_cb, wipe_modal_error]
    )

    btn_cancel_wipe.click(
        fn=handle_close_wipe_modal,
        outputs=[wipe_modal, wipe_confirm_cb, wipe_modal_error]
    )

    btn_confirm_wipe.click(
        fn=handle_execute_wipe,
        inputs=[wipe_confirm_cb],
        outputs=[
            wipe_modal,
            wipe_modal_error,
            wipe_db_status,
            db_table,
            db_stats_md,
            manage_person_dropdown,
            existing_person_picker,
            person_gallery,
            person_info_md,
            edit_person_name,
            edit_person_notes,
            selected_person_avatar,
            system_info_md
        ]
    )

    # Grupno označavanje i brisanje osoba (Batch Delete)
    multi_select_mode_cb.change(
        fn=lambda active: gr.update(visible=active),
        inputs=[multi_select_mode_cb],
        outputs=[batch_delete_tray]
    )

    batch_selected_dropdown.change(
        fn=on_batch_selection_change,
        inputs=[batch_selected_dropdown],
        outputs=[btn_open_batch_delete_modal]
    )

    btn_select_all_filtered.click(
        fn=handle_select_all_filtered,
        inputs=[table_search_input, batch_selected_dropdown],
        outputs=[batch_selected_dropdown, btn_open_batch_delete_modal, batch_action_status]
    )

    btn_clear_batch_selection.click(
        fn=handle_clear_batch_selection,
        outputs=[batch_selected_dropdown, btn_open_batch_delete_modal, batch_action_status]
    )

    btn_open_batch_delete_modal.click(
        fn=handle_open_batch_delete_modal,
        inputs=[batch_selected_dropdown],
        outputs=[batch_delete_modal, batch_delete_modal_summary, batch_action_status, btn_confirm_batch_delete]
    )

    btn_cancel_batch_delete.click(
        fn=handle_close_batch_delete_modal,
        outputs=[batch_delete_modal]
    )

    btn_confirm_batch_delete.click(
        fn=handle_execute_batch_delete,
        inputs=[batch_selected_dropdown],
        outputs=[
            batch_delete_modal,
            batch_action_status,
            batch_selected_dropdown,
            btn_open_batch_delete_modal,
            db_table,
            db_stats_md,
            manage_person_dropdown,
            existing_person_picker,
            person_gallery,
            person_info_md,
            edit_person_name,
            edit_person_notes,
            selected_person_avatar,
            system_info_md
        ]
    )

    btn_open_snaps_quick.click(
        fn=lambda: config.open_folder_in_explorer()[1],
        outputs=[rec_status_md]
    )

    btn_open_snapshots_folder.click(
        fn=lambda: config.open_folder_in_explorer()[1],
        outputs=[snap_action_status]
    )

    btn_refresh_snapshots.click(
        fn=get_snapshots_ui_data,
        outputs=[snapshots_gallery, snapshots_table, snapshots_info_md, selected_snap_dropdown, selected_snap_preview, selected_snap_info]
    )

    snapshots_gallery.select(
        fn=on_snapshot_gallery_select,
        outputs=[selected_snap_preview, selected_snap_info, selected_snap_dropdown]
    )

    selected_snap_dropdown.change(
        fn=on_snapshot_dropdown_change,
        inputs=[selected_snap_dropdown],
        outputs=[selected_snap_preview, selected_snap_info]
    )

    btn_delete_snap.click(
        fn=on_delete_snapshot_click,
        inputs=[selected_snap_dropdown],
        outputs=[snap_action_status, snapshots_gallery, snapshots_table, snapshots_info_md, selected_snap_dropdown, selected_snap_preview, selected_snap_info]
    )

    btn_save_snap_dir.click(
        fn=on_save_snapshot_dir_click,
        inputs=[custom_snap_dir_input],
        outputs=[snap_dir_status_md, snapshots_info_md, snapshots_gallery, snapshots_table, selected_snap_dropdown, selected_snap_preview, selected_snap_info]
    )

    btn_reset_snap_dir.click(
        fn=on_reset_snapshot_dir_click,
        outputs=[snap_dir_status_md, custom_snap_dir_input, snapshots_info_md, snapshots_gallery, snapshots_table, selected_snap_dropdown, selected_snap_preview, selected_snap_info]
    )

    btn_send_to_rec.click(
        fn=on_send_snapshot_to_recognition,
        inputs=[selected_snap_dropdown, threshold_slider, landmarks_chk, blur_mode_radio, cards_show_all_chk],
        outputs=[input_img, annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state, detection_cards_html, snap_action_status]
    )

    # 9. Detection Events wiring
    events_out_all = [
        events_stats_md, events_table, event_camera_preview, event_crop_preview, 
        event_details_md, event_video_player, selected_event_id_state, btn_delete_single_event
    ]

    events_search_input.change(
        fn=on_events_search,
        inputs=[events_search_input],
        outputs=events_out_all
    )

    btn_refresh_events.click(
        fn=on_events_search,
        inputs=[events_search_input],
        outputs=events_out_all
    )

    tab_events.select(
        fn=on_events_search,
        inputs=[events_search_input],
        outputs=events_out_all
    )

    events_table.select(
        fn=on_event_select,
        inputs=[events_search_input],
        outputs=[event_camera_preview, event_crop_preview, event_details_md, event_video_player, selected_event_id_state, btn_delete_single_event]
    )

    # Sigurnosni modal za brisanje cijelog dnevnika
    btn_clear_events.click(
        fn=handle_open_clear_events_modal,
        outputs=[clear_events_modal, clear_events_modal_stats, clear_events_confirm_cb, clear_events_modal_error]
    )

    btn_cancel_clear_events.click(
        fn=handle_close_clear_events_modal,
        outputs=[clear_events_modal, clear_events_confirm_cb, clear_events_modal_error]
    )

    btn_confirm_clear_events.click(
        fn=handle_execute_clear_events,
        inputs=[clear_events_confirm_cb, events_search_input],
        outputs=[clear_events_modal, clear_events_modal_error, events_stats_md, events_table, event_camera_preview, event_crop_preview, event_details_md, event_video_player, selected_event_id_state, btn_delete_single_event]
    )

    # Sigurnosni modal za brisanje pojedinačnog odabranog prolaska
    btn_delete_single_event.click(
        fn=handle_open_single_delete_event_modal,
        inputs=[selected_event_id_state],
        outputs=[single_event_delete_modal, single_event_delete_summary]
    )

    btn_cancel_single_delete.click(
        fn=handle_close_single_delete_event_modal,
        outputs=[single_event_delete_modal, single_event_delete_summary]
    )

    btn_confirm_single_delete.click(
        fn=handle_execute_single_delete_event,
        inputs=[selected_event_id_state, events_search_input],
        outputs=[single_event_delete_modal, events_stats_md, events_table, event_camera_preview, event_crop_preview, event_details_md, event_video_player, selected_event_id_state, btn_delete_single_event]
    )

    btn_export_events_csv.click(
        fn=handle_export_events_csv,
        outputs=[events_export_file, events_status_md]
    )

    # 10. NVR Video Archive wiring in Tab 3
    nvr_archive_table.select(
        fn=on_nvr_segment_select,
        outputs=[nvr_video_preview, nvr_selected_info_md]
    )

    btn_refresh_nvr_tab.click(
        fn=get_nvr_archive_ui_data,
        outputs=[nvr_storage_status_md, nvr_archive_table, nvr_video_preview, nvr_selected_info_md]
    )

    btn_open_nvr_folder.click(
        fn=handle_open_nvr_folder,
        outputs=[nvr_action_status_md]
    )

    btn_open_video_player.click(
        fn=handle_open_external_video,
        inputs=[nvr_video_preview],
        outputs=[nvr_action_status_md]
    )

    btn_delete_video_seg.click(
        fn=handle_delete_nvr_segment,
        inputs=[nvr_video_preview],
        outputs=[nvr_action_status_md, nvr_storage_status_md, nvr_archive_table, nvr_video_preview, nvr_selected_info_md]
    )

    tab_saved_media.select(
        fn=get_snapshots_ui_data,
        outputs=[snapshots_gallery, snapshots_table, snapshots_info_md, selected_snap_dropdown, selected_snap_preview, selected_snap_info]
    ).then(
        fn=get_nvr_archive_ui_data,
        outputs=[nvr_storage_status_md, nvr_archive_table, nvr_video_preview, nvr_selected_info_md]
    )

    btn_refresh_sysinfo.click(
        fn=handle_refresh_sysinfo,
        outputs=[hw_accel_badge, system_info_md]
    )

    btn_export_backup.click(
        fn=handle_export_backup,
        outputs=[backup_download_file, backup_export_status]
    )

    btn_open_backup_folder.click(
        fn=handle_open_backup_folder,
        outputs=[backup_export_status]
    )

    btn_import_backup.click(
        fn=handle_import_backup,
        inputs=[backup_upload_file],
        outputs=[
            backup_import_status,
            db_table,
            db_stats_md,
            manage_person_dropdown,
            existing_person_picker,
            system_info_md
        ]
    )

    retention_period_radio.change(
        fn=handle_retention_period_change,
        inputs=[retention_period_radio],
        outputs=[retention_status_md]
    )

    btn_run_retention.click(
        fn=handle_run_retention_cleanup,
        inputs=[retention_period_radio],
        outputs=[retention_status_md, system_info_md]
    )

    btn_save_telegram_config.click(
        fn=handle_save_telegram_config,
        inputs=[
            telegram_enabled_chk,
            telegram_token_input,
            telegram_chat_id_input,
            telegram_notify_blacklist_chk,
            telegram_notify_vip_chk,
            telegram_notify_spoof_chk,
            telegram_notify_unknown_chk,
            telegram_cooldown_slider,
            telegram_chat_id_sec_input,
            telegram_chat_id_vip_input
        ],
        outputs=[telegram_status_md]
    )

    btn_test_telegram.click(
        fn=handle_test_telegram,
        inputs=[telegram_token_input, telegram_chat_id_input],
        outputs=[telegram_status_md]
    )

    btn_save_email_config.click(
        fn=handle_save_email_config,
        inputs=[
            email_enabled_chk,
            email_smtp_host,
            email_smtp_port,
            email_encryption_radio,
            email_sender,
            email_password,
            email_recipients,
            email_notify_blacklist_chk,
            email_notify_spoof_chk,
            email_notify_vip_chk,
            email_cooldown_slider
        ],
        outputs=[email_status_md]
    )

    btn_test_email.click(
        fn=handle_test_email,
        inputs=[
            email_smtp_host,
            email_smtp_port,
            email_encryption_radio,
            email_sender,
            email_password,
            email_recipients
        ],
        outputs=[email_status_md]
    )

    # ------------------ EVENT HANDLERS: TAB 6 PHOTO SORTER ------------------
    btn_browse_input_folder.click(
        fn=on_browse_input_folder,
        inputs=[sorter_input_folder],
        outputs=[sorter_input_folder, sorter_folder_info_md, sorter_output_folder]
    )

    btn_browse_output_folder.click(
        fn=on_browse_output_folder,
        inputs=[sorter_output_folder],
        outputs=[sorter_output_folder]
    )

    btn_check_input_folder.click(
        fn=handle_validate_input_folder,
        inputs=[sorter_input_folder],
        outputs=[sorter_folder_info_md, sorter_output_folder]
    )

    btn_start_sorter.click(
        fn=handle_start_photo_sorting,
        inputs=[
            sorter_input_folder,
            sorter_output_folder,
            sorter_target_persons,
            sorter_similarity,
            sorter_action_mode,
            sorter_enable_combo,
            sorter_enable_group,
            sorter_group_min,
            sorter_enable_noface,
            sorter_enable_unregistered,
            sorter_resolution
        ],
        outputs=[
            sorter_status_md,
            sorter_stats_breakdown_md,
            sorter_export_csv_file,
            btn_start_sorter,
            btn_cancel_sorter,
            btn_open_sorter_dir
        ]
    )

    btn_cancel_sorter.click(
        fn=handle_cancel_photo_sorting,
        inputs=[],
        outputs=[sorter_status_md]
    )

    btn_open_sorter_dir.click(
        fn=handle_open_sorter_folder,
        inputs=[sorter_output_folder],
        outputs=[sorter_status_md]
    )

    tab_photo_sorter.select(
        fn=lambda: gr.update(choices=get_sorter_person_choices()),
        inputs=[],
        outputs=[sorter_target_persons]
    )

    demo.load(
        fn=refresh_database_view,
        outputs=[db_table, db_stats_md]
    ).then(
        fn=update_both_person_dropdowns,
        outputs=[manage_person_dropdown, existing_person_picker]
    )

def launch_app(desktop: bool = True, port: int = 7860):
    """
    Pokreće ArgusFace Studio.
    - Ako je desktop=True i pywebview je dostupan: pokreće samostalni nativni prozor (WebView2)
      s čistim automatskim gašenjem svih servisa na zatvaranje prozora ('X').
    - Ako pywebview nije instaliran ili je proslijeđen argument --browser / --web:
      pokreće aplikaciju u zadanom web pregledniku.
    """
    import time
    import urllib.request

    if "--browser" in sys.argv or "--web" in sys.argv:
        desktop = False

    # Tiho automatsko GDPR čišćenje pri svakom startu aplikacije
    try:
        ret_days = config.get_retention_days()
        if ret_days > 0:
            config.execute_gdpr_retention(ret_days)
    except Exception as e:
        print(f"[GDPR RETENTION] Greška pri startnoj provjeri: {e}")

    has_webview = False
    if desktop:
        try:
            import webview
            has_webview = True
        except ImportError:
            has_webview = False

    allowed_list = [
        os.path.abspath(DATA_DIR),
        os.path.abspath(os.path.join(DATA_DIR, "backups")),
        os.path.abspath(APP_DIR)
    ]

    if has_webview:
        print("===================================================")
        print("      ArgusFace Studio - Samostalni Radni Prozor    ")
        print("===================================================")
        print("Pokrećem pozadinski servis...")

        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=False,
            prevent_thread_lock=True,
            theme=custom_theme,
            css=CUSTOM_CSS,
            head=HEAD_DARK_JS,
            show_error=True,
            allowed_paths=allowed_list
        )

        url = getattr(demo, "local_url", None) or f"http://127.0.0.1:{port}"
        print(f"Lokalni poslužitelj spreman na: {url}")

        # Provjeri je li web server spreman za primanje zahtjeva
        for _ in range(50):
            try:
                with urllib.request.urlopen(url, timeout=1) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                time.sleep(0.1)

        # Pronađi ikonu aplikacije
        icon_path = None
        for candidate in [
            os.path.join(APP_DIR, "uniface.ico"),
            os.path.join(APP_DIR, "repo", "uniface.ico"),
            os.path.join(APP_DIR, "assets", "uniface.ico"),
        ]:
            if os.path.exists(candidate):
                icon_path = candidate
                break

        # Otvori samostalni desktop prozor
        window = webview.create_window(
            title="ArgusFace Studio - Sustav za biometrijsku identifikaciju i NVR nadzor",
            url=url,
            width=1400,
            height=880,
            min_size=(1024, 700),
            background_color="#0b0f19",
            text_select=True,
            zoomable=True
        )

        try:
            webview.start(icon=icon_path)
        finally:
            print("\nZatvaranje ArgusFace Studio prozora i gašenje poslužitelja...")
            try:
                demo.close()
            except Exception:
                pass
            os._exit(0)
    else:
        print("===================================================")
        print("       ArgusFace Studio - Web Preglednik           ")
        print("===================================================")
        print(f"Pokrećem u web pregledniku na http://127.0.0.1:{port}...")
        demo.launch(
            server_name="127.0.0.1",
            server_port=port,
            inbrowser=True,
            theme=custom_theme,
            css=CUSTOM_CSS,
            head=HEAD_DARK_JS,
            allowed_paths=allowed_list
        )

if __name__ == "__main__":
    launch_app(desktop=True)

