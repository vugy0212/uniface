# Tab 2: Baza Osoba (unos, uređivanje, galerija, masovno brisanje)
import os
import sys
import re
import cv2
import uuid
import html
import numpy as np
import pandas as pd
import gradio as gr
from PIL import Image
from typing import Optional, List, Dict, Any

import db
import config
import face_engine
from ui.theme import format_role_badge, get_profile_badge, render_person_avatar_html, get_header_bar_html
from ui.common import APP_DIR, DATA_DIR, UPLOADS_DIR, CROPS_DIR, save_image_dedup, imread_unicode, imwrite_unicode

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
        
    person_name = re.sub(r'[\r\n\t\x00-\x1f]', '', str(name or "")).strip()[:100]
    person_notes = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(notes or "")).strip()[:500]
    if not person_name:
        return (
            "⚠️ Ime osobe ne može biti prazno ili sadržavati samo kontrolne znakove!",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(),
            state,
            gr.update(), gr.update()
        )
    person_role = (role or "standard").strip().lower()
    person_id = db.get_or_create_person(person_name, person_notes, role=person_role)
    try:
        db.update_person(person_id, person_name, person_notes, role=person_role)
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
            if not single_name or not str(single_name).strip():
                return "⚠️ Morate upisati ime osobe za sve slike.", gr.update(), gr.update(), gr.update(), gr.update()
            person_name = str(single_name).strip()
        person_name = re.sub(r'[\r\n\t\x00-\x1f]', '', person_name).strip()[:100]
        if not person_name:
            fail_list.append(f"{base_display_name} (neispravno ili prazno ime)")
            continue
            
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
    
    new_name = re.sub(r'[\r\n\t\x00-\x1f]', '', str(new_name or "")).strip()[:100]
    new_notes = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(new_notes or "")).strip()[:500]
    if not new_name:
        return (
            "⚠️ Ime i prezime osobe ne smije biti prazno.",
            gr.update(), gr.update(), gr.update(), gr.update(),
            gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        )
    
    role_clean = (new_role or "standard").strip().lower()
    try:
        db.update_person(person_id, new_name, new_notes, role=role_clean)
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



def on_name_input_blur(name_val):
    n = str(name_val or "").strip()
    if n:
        return f"💾 Spremi dodatno lice za: {n}"
    return "💾 Spremi odabrano lice u bazu"



def create_tab_database():
    single_enroll_state = gr.State({
        "faces": [],
        "bgr": None,
        "selected_idx": 1,
        "saved_indices": set()
    })
    with gr.TabItem("Baza Osoba"):
        with gr.Row():
            # Lijevi stupac: Unos i označavanje na grupnoj slici
            with gr.Column(scale=1):
                with gr.Tabs():
                    # Podtab 1: Pojedinačni unos i izrezivanje s grupne slike
                    with gr.TabItem("Pojedinačni unos"):
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

    return {
        "single_img_input": single_img_input,
        "single_name_input": single_name_input,
        "single_notes_input": single_notes_input,
        "single_role_input": single_role_input,
        "single_face_selector": single_face_selector,
        "single_annotated_preview": single_annotated_preview,
        "single_crops_gallery": single_crops_gallery,
        "single_preview_crop": single_preview_crop,
        "single_preview_info": single_preview_info,
        "btn_save_single": btn_save_single,
        "btn_clear_form": btn_clear_form,
        "single_save_status": single_save_status,
        "single_enroll_state": single_enroll_state,
        "batch_naming_mode": batch_naming_mode,
        "batch_files_input": batch_files_input,
        "batch_single_name": batch_single_name,
        "btn_batch_enroll": btn_batch_enroll,
        "batch_status_md": batch_status_md,
        "table_search_input": table_search_input,
        "btn_clear_table_search": btn_clear_table_search,
        "multi_select_mode_cb": multi_select_mode_cb,
        "batch_delete_tray": batch_delete_tray,
        "btn_open_batch_delete_modal": btn_open_batch_delete_modal,
        "batch_selected_dropdown": batch_selected_dropdown,
        "btn_select_all_filtered": btn_select_all_filtered,
        "btn_clear_batch_selection": btn_clear_batch_selection,
        "batch_action_status": batch_action_status,
        "db_stats_md": db_stats_md,
        "btn_refresh_db": btn_refresh_db,
        "db_table": db_table,
        "existing_person_picker": existing_person_picker,
        "selected_person_avatar": selected_person_avatar,
        "manage_person_dropdown": manage_person_dropdown,
        "person_gallery": person_gallery,
        "person_info_md": person_info_md,
        "sample_delete_dropdown": sample_delete_dropdown,
        "btn_delete_sample": btn_delete_sample,
        "sample_action_status": sample_action_status,
        "edit_person_name": edit_person_name,
        "edit_person_notes": edit_person_notes,
        "edit_person_role": edit_person_role,
        "btn_save_person_edit": btn_save_person_edit,
        "btn_delete_person": btn_delete_person,
        "edit_person_status": edit_person_status,
        "btn_open_wipe_modal": btn_open_wipe_modal,
        "wipe_db_status": wipe_db_status,
        "wipe_modal": wipe_modal,
        "wipe_modal_stats": wipe_modal_stats,
        "wipe_confirm_cb": wipe_confirm_cb,
        "wipe_modal_error": wipe_modal_error,
        "btn_cancel_wipe": btn_cancel_wipe,
        "btn_confirm_wipe": btn_confirm_wipe,
        "batch_delete_modal": batch_delete_modal,
        "batch_delete_modal_summary": batch_delete_modal_summary,
        "btn_cancel_batch_delete": btn_cancel_batch_delete,
        "btn_confirm_batch_delete": btn_confirm_batch_delete,
    }

def wire_tab_database_events(c):
    c['batch_naming_mode'].change(
        fn=lambda m: gr.update(visible=(m == "Sve slike pripadaju istoj osobi")),
        inputs=[c['batch_naming_mode']],
        outputs=[c['batch_single_name']]
    )

    c['single_img_input'].change(
        fn=on_single_image_uploaded,
        inputs=[c['single_img_input'], c['single_name_input'], c['single_enroll_state']],
        outputs=[
            c['single_annotated_preview'],
            c['single_crops_gallery'],
            c['single_face_selector'],
            c['single_preview_crop'],
            c['single_preview_info'],
            c['btn_save_single'],
            c['single_enroll_state']
        ]
    )

    c['single_annotated_preview'].select(
        fn=on_single_image_click,
        inputs=[c['single_enroll_state']],
        outputs=[c['single_annotated_preview'], c['single_preview_crop'], c['single_preview_info'], c['single_face_selector'], c['single_enroll_state']]
    )

    c['single_face_selector'].change(
        fn=on_radio_face_change,
        inputs=[c['single_face_selector'], c['single_enroll_state']],
        outputs=[c['single_annotated_preview'], c['single_preview_crop'], c['single_preview_info'], c['single_face_selector'], c['single_enroll_state']]
    )

    c['single_crops_gallery'].select(
        fn=on_crop_gallery_select,
        inputs=[c['single_enroll_state']],
        outputs=[c['single_annotated_preview'], c['single_preview_crop'], c['single_preview_info'], c['single_face_selector'], c['single_enroll_state']]
    )

    c['btn_save_single'].click(
        fn=save_single_person,
        inputs=[c['single_name_input'], c['single_notes_input'], c['single_face_selector'], c['single_role_input'], c['single_enroll_state']],
        outputs=[
            c['single_save_status'],
            c['manage_person_dropdown'],
            c['existing_person_picker'],
            c['db_table'],
            c['db_stats_md'],
            c['single_annotated_preview'],
            c['single_preview_crop'],
            c['single_preview_info'],
            c['single_face_selector'],
            c['single_name_input'],
            c['single_notes_input'],
            c['btn_save_single'],
            c['selected_person_avatar'],
            c['edit_person_name'],
            c['edit_person_notes'],
            c['single_crops_gallery'],
            c['single_enroll_state'],
            c['edit_person_role'],
            c['single_role_input']
        ]
    ).then(
        fn=view_person_details,
        inputs=[c['manage_person_dropdown']],
        outputs=[c['person_gallery'], c['person_info_md'], c['sample_delete_dropdown']]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['single_name_input'].blur(
        fn=on_name_input_blur,
        inputs=[c['single_name_input']],
        outputs=[c['btn_save_single']]
    )

    c['btn_clear_form'].click(
        fn=clear_single_form,
        outputs=[
            c['single_name_input'], c['single_notes_input'], c['single_img_input'],
            c['single_annotated_preview'], c['single_crops_gallery'],
            c['single_face_selector'], c['single_preview_crop'], c['single_preview_info'],
            c['btn_save_single'], c['single_save_status'],
            c['existing_person_picker'], c['single_enroll_state'],
            c['selected_person_avatar'], c['single_role_input']
        ]
    )

    c['existing_person_picker'].change(
        fn=on_existing_person_picked,
        inputs=[c['existing_person_picker']],
        outputs=[
            c['single_name_input'], c['single_notes_input'], c['btn_save_single'], c['single_save_status'],
            c['person_gallery'], c['person_info_md'], c['sample_delete_dropdown'], c['manage_person_dropdown'],
            c['selected_person_avatar'],
            c['edit_person_name'], c['edit_person_notes'], c['edit_person_status'],
            c['single_role_input'], c['edit_person_role']
        ],
        show_progress="hidden"
    )

    c['table_search_input'].change(
        fn=on_table_search_changed,
        inputs=[c['table_search_input']],
        outputs=[c['db_table'], c['db_stats_md'], c['manage_person_dropdown']]
    )

    c['btn_clear_table_search'].click(
        fn=on_table_search_clear,
        outputs=[c['table_search_input'], c['db_table'], c['db_stats_md'], c['manage_person_dropdown']]
    )

    c['btn_batch_enroll'].click(
        fn=batch_enroll_files,
        inputs=[c['batch_naming_mode'], c['batch_single_name'], c['batch_files_input']],
        outputs=[c['batch_status_md'], c['manage_person_dropdown'], c['existing_person_picker'], c['db_table'], c['db_stats_md']]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['btn_refresh_db'].click(
        fn=refresh_database_view,
        outputs=[c['db_table'], c['db_stats_md']]
    ).then(
        fn=update_both_person_dropdowns,
        outputs=[c['manage_person_dropdown'], c['existing_person_picker']]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['db_table'].select(
        fn=on_table_select,
        inputs=[c['db_table'], c['multi_select_mode_cb'], c['batch_selected_dropdown']],
        outputs=[
            c['manage_person_dropdown'], c['person_gallery'], c['person_info_md'], c['sample_delete_dropdown'],
            c['single_name_input'], c['single_notes_input'], c['btn_save_single'], c['single_save_status'],
            c['existing_person_picker'],
            c['selected_person_avatar'],
            c['edit_person_name'], c['edit_person_notes'], c['edit_person_status'],
            c['batch_selected_dropdown'], c['btn_open_batch_delete_modal'], c['batch_action_status'],
            c['single_role_input'], c['edit_person_role']
        ],
        show_progress="hidden"
    )

    c['multi_select_mode_cb'].change(
        fn=lambda active: gr.update(visible=active),
        inputs=[c['multi_select_mode_cb']],
        outputs=[c['batch_delete_tray']]
    )

    c['batch_selected_dropdown'].change(
        fn=on_batch_selection_change,
        inputs=[c['batch_selected_dropdown']],
        outputs=[c['btn_open_batch_delete_modal']]
    )

    c['btn_select_all_filtered'].click(
        fn=handle_select_all_filtered,
        inputs=[c['table_search_input'], c['batch_selected_dropdown']],
        outputs=[c['batch_selected_dropdown'], c['btn_open_batch_delete_modal'], c['batch_action_status']]
    )

    c['btn_clear_batch_selection'].click(
        fn=handle_clear_batch_selection,
        outputs=[c['batch_selected_dropdown'], c['btn_open_batch_delete_modal'], c['batch_action_status']]
    )

    c['btn_open_batch_delete_modal'].click(
        fn=handle_open_batch_delete_modal,
        inputs=[c['batch_selected_dropdown']],
        outputs=[c['batch_delete_modal'], c['batch_delete_modal_summary'], c['batch_action_status'], c['btn_confirm_batch_delete']]
    )

    c['btn_cancel_batch_delete'].click(
        fn=handle_close_batch_delete_modal,
        outputs=[c['batch_delete_modal']]
    )

    c['btn_confirm_batch_delete'].click(
        fn=handle_execute_batch_delete,
        inputs=[c['batch_selected_dropdown']],
        outputs=[
            c['batch_delete_modal'],
            c['batch_action_status'],
            c['batch_selected_dropdown'],
            c['btn_open_batch_delete_modal'],
            c['db_table'],
            c['db_stats_md'],
            c['manage_person_dropdown'],
            c['existing_person_picker'],
            c['person_gallery'],
            c['person_info_md'],
            c['edit_person_name'],
            c['edit_person_notes'],
            c['selected_person_avatar'],
            c['system_info_md']
        ]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['manage_person_dropdown'].change(
        fn=on_manage_person_change,
        inputs=[c['manage_person_dropdown']],
        outputs=[c['person_gallery'], c['person_info_md'], c['sample_delete_dropdown'], c['edit_person_name'], c['edit_person_notes'], c['edit_person_status'], c['edit_person_role']]
    )

    c['btn_save_person_edit'].click(
        fn=update_person_handler,
        inputs=[c['manage_person_dropdown'], c['edit_person_name'], c['edit_person_notes'], c['edit_person_role']],
        outputs=[
            c['edit_person_status'],
            c['db_table'],
            c['db_stats_md'],
            c['manage_person_dropdown'],
            c['existing_person_picker'],
            c['single_name_input'],
            c['single_notes_input'],
            c['person_info_md'],
            c['selected_person_avatar'],
            c['btn_save_single'],
            c['single_role_input']
        ]
    )

    c['btn_delete_sample'].click(
        fn=delete_selected_sample,
        inputs=[c['sample_delete_dropdown'], c['manage_person_dropdown']],
        outputs=[c['sample_action_status'], c['person_gallery'], c['person_info_md']]
    ).then(
        fn=refresh_database_view,
        outputs=[c['db_table'], c['db_stats_md']]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['btn_delete_person'].click(
        fn=delete_selected_person,
        inputs=[c['manage_person_dropdown']],
        outputs=[
            c['sample_action_status'],
            c['manage_person_dropdown'],
            c['existing_person_picker'],
            c['db_table'],
            c['db_stats_md'],
            c['person_gallery'],
            c['person_info_md'],
            c['edit_person_name'],
            c['edit_person_notes'],
            c['edit_person_status']
        ]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['btn_open_wipe_modal'].click(
        fn=handle_open_wipe_modal,
        outputs=[c['wipe_modal'], c['wipe_modal_stats'], c['wipe_confirm_cb'], c['wipe_modal_error']]
    )

    c['btn_cancel_wipe'].click(
        fn=handle_close_wipe_modal,
        outputs=[c['wipe_modal'], c['wipe_confirm_cb'], c['wipe_modal_error']]
    )

    c['btn_confirm_wipe'].click(
        fn=handle_execute_wipe,
        inputs=[c['wipe_confirm_cb']],
        outputs=[
            c['wipe_modal'],
            c['wipe_modal_error'],
            c['wipe_db_status'],
            c['db_table'],
            c['db_stats_md'],
            c['manage_person_dropdown'],
            c['existing_person_picker'],
            c['person_gallery'],
            c['person_info_md'],
            c['edit_person_name'],
            c['edit_person_notes'],
            c['selected_person_avatar'],
            c['system_info_md']
        ]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )
