import os
import sys
import uuid
import re
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db
import face_engine
import hardware
import backup
import config
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

def refresh_database_view(search_query=""):
    all_persons = db.get_all_persons()
    stats = db.get_stats()
    query = (search_query or "").strip().lower()
    
    if query:
        filtered = [
            p for p in all_persons
            if query in p["name"].lower() or query == str(p["id"]) or query in (p["notes"] or "").lower()
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
            p["sample_count"],
            get_profile_badge(p["sample_count"]),
            p["notes"] or "-",
            p["created_at"]
        ])
        
    return rows, stats_text

# ---------------- PREPOZNAVANJE ----------------
def recognize_faces(image, threshold, draw_landmarks, blur_unknown):
    if image is None:
        return None, [], [], "⚠️ Molimo učitajte sliku za analizu.", gr.update(choices=[], value=None), []
    
    img_bgr, err = imread_unicode(image)
    if img_bgr is None:
        return None, [], [], f"❌ Greška pri obradi slike: {err}", gr.update(choices=[], value=None), []
        
    annotated_bgr, results = face_engine.process_and_annotate(
        img_bgr,
        threshold=float(threshold),
        draw_landmarks=bool(draw_landmarks),
        blur_unknown=bool(blur_unknown)
    )
    
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    
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
        crop_rgb = cv2.cvtColor(r["crop_bgr"], cv2.COLOR_BGR2RGB)
        
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
    return annotated_rgb, crops_gallery, table_data, summary, dropdown_update, results

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
    return msg, gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), table_view, stats_view

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

def on_single_image_uploaded(image, state):
    new_state = {"faces": [], "bgr": None, "selected_idx": 1, "saved_indices": set()}
    if image is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(choices=[], value=None, visible=False),
            None, "Učitajte fotografiju za automatsku detekciju lica.",
            new_state
        )
        
    img_bgr, err = imread_unicode(image)
    if img_bgr is None:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(choices=[], value=None, visible=False),
            None, f"❌ Greška pri čitanju slike: {err}",
            new_state
        )
        
    faces = face_engine.extract_faces_from_image(img_bgr)
    if not faces:
        return (
            gr.update(value=None, visible=False),
            gr.update(value=[], visible=False),
            gr.update(choices=[], value=None, visible=False),
            None, "⚠️ Na slici NIJE pronađeno lice. Pokušajte s jasnijom slikom.",
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
    info1 = f"👥 Pronađeno **{len(faces)}** lica na slici! Trenutno je odabrano: **Lice #1**."
    if f1.get("age") is not None:
        info1 += f" | Procjena dobi: ~{int(f1['age'])} god, Spol: {f1['gender']}"
        
    return (
        gr.update(value=annotated_rgb, visible=True),
        gr.update(value=gallery, visible=True),
        gr.update(choices=choices, value=sel_choice, visible=True),
        crop1_rgb,
        info1,
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

def save_single_person(name, notes, face_choice_str, state):
    state = _get_single_state(state)
    faces = state["faces"]
    bgr = state["bgr"]
    saved = state["saved_indices"]
    
    if not name or not name.strip():
        return (
            "⚠️ Ime osobe je obavezno!",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            state
        )
    if not faces:
        return (
            "⚠️ Niste učitali sliku ili na slici nema lica!",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
            state
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
    person_id = db.get_or_create_person(person_name, notes or "")
    
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
            
    if next_idx is not None:
        state["selected_idx"] = next_idx
        msg = f"🎉 **Lice #{target_idx}** uspješno spremljeno za osobu **{person_name}**! Sada upišite ime za sljedeću osobu (**Lice #{next_idx}**)."
    else:
        msg = f"🎉 **Lice #{target_idx}** uspješno spremljeno za osobu **{person_name}**! Sva lica sa slike ({len(faces)}) su unesena u bazu!"
        
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
        gr.update(choices=choices_db, value=f"{person_id}: {person_name} ({len(samples)} slika)"),
        gr.update(choices=choices_db, value=None),
        table_view,
        stats_view,
        annotated_rgb,
        crop_rgb,
        info,
        gr.update(choices=new_choices, value=sel_choice),
        "",  # clear name field so user can type next person's name
        new_gallery,
        state
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
        empty_state
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
    * **Kvaliteta profila:** {badge}
    * **Biometrijski model:** {centroid_status}
    * **Bilješke:** {person['notes'] or 'Nema bilješki'}
    * **Registrirano slika:** {count}
    
    {rec}
    """
    return gallery, info_text, gr.update(choices=sample_choices, value=sample_choices[0] if sample_choices else None)

def on_table_select(table_data, evt: gr.SelectData):
    if evt.index is None:
        return gr.update(), [], "", gr.update(), gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "", gr.update()
        
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
        return gr.update(), [], "", gr.update(), gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "", gr.update()
        
    p = db.get_person(person_id)
    if not p:
        return gr.update(), [], "", gr.update(), gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "", gr.update()
        
    choice = f"{p['id']}: {p['name']} ({p['sample_count']} slika)"
    gallery, info, sample_drop = view_person_details(choice)
    
    name_val = p["name"]
    notes_val = p["notes"] or ""
    btn_text = f"💾 Spremi odabrano lice za: {p['name']}"
    status_msg = f"📌 Odabrano za unos: **{p['name']}** ({p['sample_count']} slika). Odaberite lice sa slike i kliknite Spremi."
    
    return (
        choice, gallery, info, sample_drop,
        name_val, notes_val, btn_text, status_msg,
        gr.update(value=choice)
    )

def on_existing_person_picked(selected_choice):
    person_id = parse_person_id(selected_choice)
    if person_id is None:
        return gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "", [], "", gr.update(), gr.update()
        
    p = db.get_person(person_id)
    if not p:
        return gr.update(), gr.update(), "💾 Spremi odabrano lice u bazu", "", [], "", gr.update(), gr.update()
        
    name_val = p["name"]
    notes_val = p["notes"] or ""
    btn_text = f"💾 Spremi odabrano lice za: {p['name']}"
    status_msg = f"📌 Odabrano za unos: **{p['name']}** ({p['sample_count']} slika). Odaberite lice sa slike i kliknite Spremi."
    choice_str = f"{p['id']}: {p['name']} ({p['sample_count']} slika)"
    gallery, info, sample_drop = view_person_details(choice_str)
    
    return (
        name_val, notes_val, btn_text, status_msg,
        gallery, info, sample_drop, gr.update(value=choice_str)
    )

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
        return "⚠️ Niste odabrali osobu.", gr.update(), gr.update(), gr.update(), gr.update(), [], ""
        
    p = db.get_person(person_id)
    if p:
        db.delete_person(person_id)
        msg = f"🗑️ Osoba **{p['name']}** i sve njezine fotografije su obrisane iz baze."
    else:
        msg = "Osoba ne postoji."
        
    choices = get_person_dropdown_choices()
    table_view, stats_view = refresh_database_view()
    return msg, gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), table_view, stats_view, [], ""

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
        return zip_path, f"✅ **Sigurnosna kopija uspješno generirana:** `{filename}`"
    except Exception as e:
        return None, f"❌ **Greška pri izvozu:** {e}"

def handle_import_backup(file_obj):
    if not file_obj:
        return "⚠️ Niste odabrali ZIP datoteku za uvoz.", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    success, msg = backup.import_database_zip(file_obj, DATA_DIR)
    table_view, stats_view = refresh_database_view()
    choices = get_person_dropdown_choices()
    sys_report = hardware.get_system_report_markdown(DATA_DIR)
    return msg, table_view, stats_view, gr.update(choices=choices, value=None), gr.update(choices=choices, value=None), sys_report

def handle_refresh_sysinfo():
    return hardware.get_system_report_markdown(DATA_DIR)

def handle_launch_live(source_type="USB Web Kamera", usb_idx="0", rtsp_url="", youtube_url="", video_file=None, start_sec=0, log_events=False, cooldown_sec=30):
    live_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_cam.py")
    if not os.path.exists(live_script):
        return "⚠️ Skripta `live_cam.py` nije pronađena."
    try:
        import subprocess
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_CONSOLE
            
        candidates = [
            os.path.join(APP_DIR, ".venv", "Scripts", "python.exe"),
            os.path.join(os.path.dirname(APP_DIR), ".venv", "Scripts", "python.exe"),
            os.path.join(sys.prefix, "Scripts", "python.exe"),
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

        subprocess.Popen(cmd, cwd=APP_DIR, env=env, creationflags=creationflags)
        
        return f"🎥 **Live prepoznavanje [{target_name}] je uspješno pokrenuto u novom prozoru!**\n*(Pritisnite tipku `S` za spremanje kadra u mapu, `O` za otvaranje mape, klizač ili `A`/`D`/`J`/`L` za premotavanje videa, `Q` za izlaz)*"
    except Exception as e:
        return f"❌ Greška pri pokretanju: {e}"

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

def on_send_snapshot_to_recognition(filename, threshold, landmarks, blur):
    if not filename:
        return None, None, [], None, "⚠️ Nema odabrane snimke za analizu.", gr.update(choices=[]), [], "⚠️ Nema odabrane snimke."
    snap_dir = config.get_snapshot_dir()
    path = os.path.join(snap_dir, filename)
    if not os.path.isfile(path):
        return None, None, [], None, "⚠️ Datoteka nije pronađena.", gr.update(choices=[]), [], "⚠️ Datoteka nije pronađena."
    
    pil_img = Image.open(path).convert("RGB")
    annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state = recognize_faces(
        pil_img, threshold, landmarks, blur
    )
    msg = f"✅ Kadar `{filename}` je prebačen u Tab 1 i analiziran!"
    return pil_img, annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state, msg

# ---------------- DETECTION EVENT LOG HELPERS ----------------
def get_events_ui_data(search_query=""):
    events = db.get_detection_events(limit=300, name_filter=search_query)
    stats = db.get_detection_stats()
    
    table_rows = []
    gallery_items = []
    
    for ev in events:
        sim_pct = f"{ev['similarity']*100:.1f}%"
        table_rows.append([
            ev["id"],
            ev["local_time"],
            ev["person_name"],
            sim_pct,
            ev["source_label"]
        ])
        snap_p = ev.get("snapshot_path", "")
        crop_p = ev.get("crop_path", "")
        thumb = crop_p if (crop_p and os.path.exists(crop_p)) else (snap_p if (snap_p and os.path.exists(snap_p)) else None)
        if thumb and os.path.exists(thumb):
            caption = f"{ev['person_name']} ({sim_pct}) - {ev['local_time']}"
            gallery_items.append((thumb, caption))
            
    stats_md = (
        f"📊 **Ukupno prolazaka:** `{stats['total_events']}` &nbsp;|&nbsp; "
        f"👥 **Jedinstvenih osoba:** `{stats['unique_persons']}` &nbsp;|&nbsp; "
        f"⏱️ **Zadnji zabilježeni prolazak:** `{stats['latest_event']}`"
    )
    first_cam = None
    first_crop = None
    first_info = "💡 *Kliknite na redak u tablici ili sličicu u galeriji za pregled kadra kamere i detalja.*"
    if events:
        first_ev = events[0]
        c_p = first_ev.get("snapshot_path", "")
        cr_p = first_ev.get("crop_path", "")
        first_cam = c_p if (c_p and os.path.exists(c_p)) else (cr_p if (cr_p and os.path.exists(cr_p)) else None)
        first_crop = cr_p if (cr_p and os.path.exists(cr_p)) else None
        cam_desc = "Kadar s kamere (nadzor)" if (c_p and os.path.exists(c_p)) else "Izrezano lice"
        first_info = (
            f"### 📋 Detalji odabranog prolaska\n"
            f"* **Prepoznata osoba:** **`{first_ev['person_name']}`**\n"
            f"* **Vrijeme prolaska:** `{first_ev['local_time']}`\n"
            f"* **Pouzdanost / Sličnost:** **`{first_ev['similarity']*100:.1f}%`**\n"
            f"* **Izvor / Kamera:** `{first_ev['source_label']}`\n"
            f"* **Prikaz slike:** {cam_desc}"
        )
        
    return stats_md, table_rows, gallery_items, first_cam, first_crop, first_info

def on_events_search(search_query=""):
    return get_events_ui_data(search_query)

def on_event_select(evt: gr.SelectData, search_query=""):
    events = db.get_detection_events(limit=300, name_filter=search_query)
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if 0 <= row_idx < len(events):
        ev = events[row_idx]
        cam_p = ev.get("snapshot_path", "")
        crop_p = ev.get("crop_path", "")
        main_img = cam_p if (cam_p and os.path.exists(cam_p)) else (crop_p if (crop_p and os.path.exists(crop_p)) else None)
        crop_img = crop_p if (crop_p and os.path.exists(crop_p)) else None
        sim_pct = f"{ev['similarity']*100:.1f}%"
        cam_desc = "✅ Puni kadar kamere" if (cam_p and os.path.exists(cam_p)) else "ℹ️ Prikaz izrezanog lica"
        info = (
            f"### 📋 Detalji prolaska #{ev['id']}\n"
            f"* **Prepoznata osoba:** **`{ev['person_name']}`**\n"
            f"* **Vrijeme prolaska:** `{ev['local_time']}`\n"
            f"* **Pouzdanost / Sličnost:** **`{sim_pct}`**\n"
            f"* **Izvor / Kamera:** `{ev['source_label']}`\n"
            f"* **Prikaz slike:** {cam_desc}"
        )
        return main_img, crop_img, info
    return None, None, "Događaj nije pronađen."

def handle_clear_events():
    db.clear_detection_events()
    return get_events_ui_data("")

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

# ---------------- GRADIO UI ----------------
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
)

with gr.Blocks(title="UniFace - Sustav za Prepoznavanje Lica") as demo:
    # Per-session state (eliminates global variables and multi-user race conditions)
    rec_faces_state = gr.State([])
    single_enroll_state = gr.State({
        "faces": [],
        "bgr": None,
        "selected_idx": 1,
        "saved_indices": set()
    })

    gr.Markdown(
        """
        # 👤 UniFace - Biometrijski Sustav za Prepoznavanje Lica
        **100% lokalno i sigurno** | RetinaFace detektor + ArcFace ResNet50 prepoznavanje + FairFace analiza dobi i spola.
        """
    )
    
    with gr.Tabs():
        # ------------------ TAB 1: PREPOZNAVANJE ------------------
        with gr.TabItem("🔍 Prepoznavanje lica"):
            with gr.Row():
                with gr.Column(scale=1):
                    input_img = gr.Image(
                        type="pil",
                        label="Učitaj sliku ili snimi web kamerom",
                        sources=["upload", "webcam"]
                    )
                    with gr.Accordion("⚙️ Napredne postavke analize", open=True):
                        threshold_slider = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.45,
                            step=0.01,
                            label="Prag prepoznavanja (Kosinusna sličnost)",
                            info="Preporučeno: 0.40 - 0.50 (veće = strože, manje = blaže)"
                        )
                        with gr.Row():
                            landmarks_chk = gr.Checkbox(value=False, label="Prikaži točke lica (Landmarks)")
                            blur_chk = gr.Checkbox(value=False, label="Zamućenje nepoznatih lica")
                            
                    with gr.Accordion("📹 Odabir izvora za Live Prikaz (USB / IP Nadzor / YouTube / Video)", open=False):
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
                            placeholder="npr. rtsp://admin:lozinka@192.168.1.100:554/stream1 ili http://192.168.1.15:8080/video",
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
                            info="Omogućuje pokretanje YouTube ili lokalnog videa od željene minute/sekunde (npr. 60 = 01:00 min, 300 = 05:00 min)",
                            visible=False
                        )

                    with gr.Row():
                        cam_enable_log = gr.Checkbox(
                            value=False,
                            label="📋 Aktiviraj evidenciju prolazaka (Dnevnik)",
                            info="Automatski zapisuje prepoznate osobe u evidenciju prolazaka uz vrijeme i sličnost",
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
                        btn_recognize = gr.Button("🚀 Pokreni prepoznavanje slike", variant="primary", scale=2)
                        btn_launch_live = gr.Button("🎥 Pokreni Live Kameru (Prozor)", variant="secondary", scale=2)
                        btn_open_snaps_quick = gr.Button("📂 Otvori mapu snimki (S)", variant="secondary", scale=1)
                    
                with gr.Column(scale=1):
                    annotated_out = gr.Image(type="numpy", label="Vizualni rezultat prepoznavanja")
                    rec_status_md = gr.Markdown("Učitajte sliku i kliknite 'Pokreni prepoznavanje'.")
                    
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Detaljna analiza svakog detektiranog lica")
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
                with gr.Column(scale=1):
                    gr.Markdown("### 🖼️ Izrezana lica s fotografije *(kliknite na lice za brzi unos)*")
                    crops_gallery_out = gr.Gallery(
                        label="Galerija detektiranih lica",
                        columns=5,
                        height="auto",
                        allow_preview=False
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### ➕ Brzi unos nepoznate osobe u bazu")
                    gr.Markdown("Kliknite na lice u galeriji s lijeve strane ili odaberite iz padajućeg izbornika:")
                    with gr.Row():
                        unknown_face_dropdown = gr.Dropdown(
                            label="Odaberite detektirano lice sa slike",
                            choices=[],
                            allow_custom_value=True
                        )
                        quick_name_input = gr.Textbox(label="Ime osobe", placeholder="npr. Marko Horvat")
                    btn_quick_add = gr.Button("Spremi ovo lice u bazu za tu osobu", variant="secondary")
                    quick_add_status = gr.Markdown("")

        # ------------------ TAB 2: UPRAVLJANJE BAZOM ------------------
        with gr.TabItem("👥 Baza Osoba"):
            with gr.Row():
                # Lijevi stupac: Unos i označavanje na grupnoj slici
                with gr.Column(scale=1):
                    with gr.Tabs():
                        # Podtab 1: Pojedinačni unos i izrezivanje s grupne slike
                        with gr.TabItem("Pojedinačni unos / Označavanje lica"):
                            # Brzi pretraživač za postojeću osobu
                            existing_person_picker = gr.Dropdown(
                                label="🔍 Pretraži i odaberi postojeću osobu iz baze (za dodavanje nove slike):",
                                choices=get_person_dropdown_choices(),
                                allow_custom_value=True,
                                info="Počnite tipkati ime i odaberite osobu za brzi unos nove slike"
                            )
                            
                            with gr.Row():
                                single_name_input = gr.Textbox(
                                    label="Ime i prezime osobe",
                                    placeholder="Upišite novo ime ili odaberite osobu iznad",
                                    scale=3
                                )
                                btn_clear_form = gr.Button("🔄 Očisti", size="sm", scale=1)
                                
                            single_notes_input = gr.Textbox(label="Bilješke (opcionalno)", placeholder="npr. Član tima, IT odjel")
                            
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
                            
                            btn_save_single = gr.Button("💾 Spremi odabrano lice u bazu", variant="primary", size="lg")
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
                            btn_batch_enroll = gr.Button("⚡ Uvezi sve fotografije u bazu", variant="primary")
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
                        btn_clear_table_search = gr.Button("✖ Poništi", size="sm", scale=1)
                        
                    db_stats_md = gr.Markdown("")
                    db_table = gr.Dataframe(
                        headers=["ID", "Ime", "Broj slika", "Kvaliteta profila", "Bilješke", "Datum registracije"],
                        label="Popis osoba (Kliknite na bilo koji redak za automatski odabir osobe za unos)",
                        interactive=False
                    )
                    btn_refresh_db = gr.Button("🔄 Osvježi cijeli popis", size="sm")
                    
                    gr.Markdown("---")
                    gr.Markdown("### 🖼️ Galerija lica i biometrijski profil")
                    with gr.Row():
                        manage_person_dropdown = gr.Dropdown(
                            label="Odaberite osobu za pregled ili brisanje",
                            choices=get_person_dropdown_choices(),
                            allow_custom_value=True
                        )
                        btn_delete_person = gr.Button("🗑️ Obriši osobu", variant="stop")
                        
                    person_info_md = gr.Markdown("Odaberite osobu iznad ili kliknite na nju u tablici za pregled lica.")
                    person_gallery = gr.Gallery(label="Spremljeni uzorci lica", columns=4, height="auto")

                    with gr.Row():
                        sample_delete_dropdown = gr.Dropdown(
                            label="Odaberite sliku za brisanje",
                            choices=[],
                            allow_custom_value=True
                        )
                        btn_delete_sample = gr.Button("Obriši odabranu sliku", variant="secondary")
                    sample_action_status = gr.Markdown("")

        # ------------------ TAB 3: SPREMLJENI KADROVI (SNAPSHOTS) ------------------
        with gr.TabItem("📸 Spremljeni Kadrovi (Snapshots)"):
            with gr.Row():
                with gr.Column(scale=3):
                    snap_init_g, snap_init_t, snap_init_info, snap_init_dd, snap_init_prev, snap_init_desc = get_snapshots_ui_data()
                    snapshots_info_md = gr.Markdown(snap_init_info)
                with gr.Column(scale=1):
                    with gr.Row():
                        btn_open_snapshots_folder = gr.Button("📂 Otvori mapu u Exploreru", variant="primary")
                        btn_refresh_snapshots = gr.Button("🔄 Osvježi galeriju", variant="secondary")

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
                        btn_delete_snap = gr.Button("🗑️ Obriši snimku", variant="stop")
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

        # ------------------ TAB 4: DNEVNIK PROLAZAKA (EVIDENCIJA) ------------------
        with gr.TabItem("📋 Dnevnik Prolazaka (Evidencija)") as tab_events:
            ev_stats_init, ev_table_init, ev_gallery_init, ev_cam_init, ev_crop_init, ev_info_init = get_events_ui_data()
            events_stats_md = gr.Markdown(ev_stats_init)
            
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
                        label="Zabilježeni prolasci"
                    )
                    gr.Markdown("### 🖼️ Galerija lica u trenutku prolaska (kliknite sličicu za prikaz)")
                    events_gallery = gr.Gallery(
                        value=ev_gallery_init,
                        columns=3,
                        height=260,
                        allow_preview=False,
                        label="Sličice prolazaka"
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
                btn_export_events_csv = gr.Button("📥 Izvezi cijeli dnevnik u CSV (Excel)", variant="primary", scale=1)
                events_export_file = gr.File(label="Preuzmi izvezenu CSV datoteku", visible=False, scale=2)
            events_status_md = gr.Markdown("")

        # ------------------ TAB 5: O SUSTAVU & SIGURNOSNA KOPIJA ------------------
        with gr.TabItem("ℹ️ O Sustavu i Sigurnosna Kopija"):
            with gr.Row():
                with gr.Column(scale=1):
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
                        backup_export_status = gr.Markdown("")
                        
                    with gr.Group():
                        gr.Markdown("#### 📥 Vraćanje sigurnosne kopije (Restore / Import)")
                        backup_upload_file = gr.File(label="Prenesite ZIP arhivu za uvoz", file_types=[".zip"], file_count="single")
                        btn_import_backup = gr.Button("⚠️ Uvezi arhivu i obnovi bazu", variant="stop")
                        backup_import_status = gr.Markdown("")


    # ------------------ EVENT HANDLERS ------------------
    batch_naming_mode.change(
        fn=lambda m: gr.update(visible=(m == "Sve slike pripadaju istoj osobi")),
        inputs=[batch_naming_mode],
        outputs=[batch_single_name]
    )
    
    # 1. Image uploaded in Single Enroll -> extracts faces and renders visual indicators
    single_img_input.change(
        fn=on_single_image_uploaded,
        inputs=[single_img_input, single_enroll_state],
        outputs=[
            single_annotated_preview,
            single_crops_gallery,
            single_face_selector,
            single_preview_crop,
            single_preview_info,
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
    
    # 5. Save single face -> advances to next unsaved face & clears name field
    btn_save_single.click(
        fn=save_single_person,
        inputs=[single_name_input, single_notes_input, single_face_selector, single_enroll_state],
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
            single_crops_gallery,
            single_enroll_state
        ]
    ).then(
        fn=view_person_details,
        inputs=[manage_person_dropdown],
        outputs=[person_gallery, person_info_md, sample_delete_dropdown]
    )
    
    # 6. Clear single form
    btn_clear_form.click(
        fn=clear_single_form,
        outputs=[
            single_name_input, single_notes_input, single_img_input,
            single_annotated_preview, single_crops_gallery,
            single_face_selector, single_preview_crop, single_preview_info,
            btn_save_single, single_save_status,
            existing_person_picker, single_enroll_state
        ]
    )
    
    # 7. Quick existing person picker in enrollment form
    existing_person_picker.change(
        fn=on_existing_person_picked,
        inputs=[existing_person_picker],
        outputs=[
            single_name_input, single_notes_input, btn_save_single, single_save_status,
            person_gallery, person_info_md, sample_delete_dropdown, manage_person_dropdown
        ]
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
    
    btn_recognize.click(
        fn=recognize_faces,
        inputs=[input_img, threshold_slider, landmarks_chk, blur_chk],
        outputs=[annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state]
    )

    def on_cam_source_change(st):
        is_video = st in ("YouTube / Web Video", "Lokalna Video Datoteka")
        return (
            gr.update(visible=(st == "USB Web Kamera")),
            gr.update(visible=(st == "IP / RTSP Kamera za nadzor")),
            gr.update(visible=(st == "YouTube / Web Video")),
            gr.update(visible=(st == "Lokalna Video Datoteka")),
            gr.update(visible=is_video)
        )

    cam_source_type.change(
        fn=on_cam_source_change,
        inputs=[cam_source_type],
        outputs=[cam_usb_idx, cam_rtsp_url, cam_youtube_url, cam_video_file, cam_start_sec]
    )

    cam_enable_log.change(
        fn=lambda v: gr.update(visible=v),
        inputs=[cam_enable_log],
        outputs=[cam_cooldown_sec]
    )

    btn_launch_live.click(
        fn=handle_launch_live,
        inputs=[cam_source_type, cam_usb_idx, cam_rtsp_url, cam_youtube_url, cam_video_file, cam_start_sec, cam_enable_log, cam_cooldown_sec],
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
    
    # CLICKING ON TABLE AUTOMATICALLY INSERTS NAME AND LOADS PROFILE ON THE LEFT
    db_table.select(
        fn=on_table_select,
        inputs=[db_table],
        outputs=[
            manage_person_dropdown, person_gallery, person_info_md, sample_delete_dropdown,
            single_name_input, single_notes_input, btn_save_single, single_save_status,
            existing_person_picker
        ]
    )
    
    manage_person_dropdown.change(
        fn=view_person_details,
        inputs=[manage_person_dropdown],
        outputs=[person_gallery, person_info_md, sample_delete_dropdown]
    )
    
    btn_delete_person.click(
        fn=delete_selected_person,
        inputs=[manage_person_dropdown],
        outputs=[sample_action_status, manage_person_dropdown, existing_person_picker, db_table, db_stats_md, person_gallery, person_info_md]
    )
    
    btn_delete_sample.click(
        fn=delete_selected_sample,
        inputs=[sample_delete_dropdown, manage_person_dropdown],
        outputs=[sample_action_status, person_gallery, person_info_md]
    ).then(
        fn=refresh_database_view,
        outputs=[db_table, db_stats_md]
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
        inputs=[selected_snap_dropdown, threshold_slider, landmarks_chk, blur_chk],
        outputs=[input_img, annotated_out, crops_gallery_out, results_table, rec_status_md, unknown_face_dropdown, rec_faces_state, snap_action_status]
    )

    # 9. Detection Events wiring
    events_search_input.change(
        fn=on_events_search,
        inputs=[events_search_input],
        outputs=[events_stats_md, events_table, events_gallery, event_camera_preview, event_crop_preview, event_details_md]
    )

    btn_refresh_events.click(
        fn=on_events_search,
        inputs=[events_search_input],
        outputs=[events_stats_md, events_table, events_gallery, event_camera_preview, event_crop_preview, event_details_md]
    )

    btn_clear_events.click(
        fn=handle_clear_events,
        outputs=[events_stats_md, events_table, events_gallery, event_camera_preview, event_crop_preview, event_details_md]
    )

    events_table.select(
        fn=on_event_select,
        inputs=[events_search_input],
        outputs=[event_camera_preview, event_crop_preview, event_details_md]
    )

    events_gallery.select(
        fn=on_event_select,
        inputs=[events_search_input],
        outputs=[event_camera_preview, event_crop_preview, event_details_md]
    )

    btn_export_events_csv.click(
        fn=handle_export_events_csv,
        outputs=[events_export_file, events_status_md]
    )

    tab_events.select(
        fn=on_events_search,
        inputs=[events_search_input],
        outputs=[events_stats_md, events_table, events_gallery, event_camera_preview, event_crop_preview, event_details_md]
    )

    btn_refresh_sysinfo.click(
        fn=handle_refresh_sysinfo,
        outputs=[system_info_md]
    )

    btn_export_backup.click(
        fn=handle_export_backup,
        outputs=[backup_download_file, backup_export_status]
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

    demo.load(
        fn=refresh_database_view,
        outputs=[db_table, db_stats_md]
    ).then(
        fn=update_both_person_dropdowns,
        outputs=[manage_person_dropdown, existing_person_picker]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True, theme=custom_theme)
