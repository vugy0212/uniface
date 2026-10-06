# Tab 1: Live Feed / Nadzor, detekcija lica, prepoznavanje
import os
import sys
import re
import cv2
import time
import html
import uuid
import datetime
import atexit
import subprocess
import numpy as np
import gradio as gr
from PIL import Image
from typing import Optional, List, Dict, Any

import db
import config
import face_engine
import notifier
from ui.theme import generate_detection_cards_html, get_header_bar_html
from ui.common import APP_DIR, DATA_DIR, UPLOADS_DIR, CROPS_DIR, SNAPSHOTS_DIR, imread_unicode, imwrite_unicode
from ui.tabs.tab_database import get_person_dropdown_choices, refresh_database_view

_active_camera_processes: List[Any] = []

def _cleanup_active_processes():
    global _active_camera_processes
    for p in list(_active_camera_processes):
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            pass
    _active_camera_processes.clear()

atexit.register(_cleanup_active_processes)


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
        
    person_name = re.sub(r'[\r\n\t\x00-\x1f]', '', str(new_name or "")).strip()[:100]
    if not person_name:
        return "⚠️ Molimo unesite valjano ime osobe.", gr.update(), gr.update(), gr.update(), gr.update()
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


def handle_launch_live(source_type="USB Web Kamera", usb_idx="0", rtsp_url="", youtube_url="", video_file=None, start_sec=0, log_events=False, cooldown_sec=30, record_nvr=False, segment_min=5, anti_spoof=True):
    live_script = os.path.join(APP_DIR, "src", "live_cam.py")
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

        if not source_arg or source_arg.startswith("-"):
            return "⚠️ Neispravna adresa izvora ili format (izvor ne smije biti prazan niti počinjati znakom '-')."

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

        proc = subprocess.Popen(cmd, cwd=APP_DIR, env=env, creationflags=creationflags)
        _active_camera_processes.append(proc)
        _active_camera_processes[:] = [p for p in _active_camera_processes if p.poll() is None]
        
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
            bool(c1_on and str(c1_src).strip() and not str(c1_src).strip().startswith("-")),
            bool(c2_on and str(c2_src).strip() and not str(c2_src).strip().startswith("-")),
            bool(c3_on and str(c3_src).strip() and not str(c3_src).strip().startswith("-")),
            bool(c4_on and str(c4_src).strip() and not str(c4_src).strip().startswith("-"))
        ])
        if active_count == 0:
            return "⚠️ Morate omogućiti barem jednu kameru s valjanom adresom za 2×2 mrežu."

        proc = subprocess.Popen(cmd, cwd=APP_DIR, env=env, creationflags=creationflags)
        _active_camera_processes.append(proc)
        _active_camera_processes[:] = [p for p in _active_camera_processes if p.poll() is None]
        nvr_tag = f" uz 24/7 NVR snimanje ({int(segment_min)}m segmenti)" if record_nvr else ""
        return f"🎛️ **Multi-Camera 2×2 mreža ({active_count} kamere) je uspješno pokrenuta u novom prozoru{nvr_tag}!**\n*(Pritisnite tipke `1`-`4` za Solo prikaz, `0` ili `ESC` za mrežu, `R` za NVR snimanje, `E` za evidenciju, `S` za kadar, `Q` za izlaz)*"
    except Exception as e:
        return f"❌ Greška pri pokretanju 2×2 mreže: {e}"

# ---------------- SNAPSHOTS & ARCHIVE HELPERS ----------------


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



def on_live_mode_change(m):
    is_grid = (m == "Mreža više kamera (2×2 Grid)")
    return (
        gr.update(visible=not is_grid),
        gr.update(visible=is_grid),
        gr.update(visible=not is_grid),
        gr.update(visible=is_grid)
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



def create_tab_live(rec_faces_state):
    with gr.TabItem("Live Feed"):
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
                            label="Prikaži točke lica (Landmarks)"
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
                    btn_recognize = gr.Button("Pokreni analizu", variant="primary", scale=2, elem_classes=["btn-industrial-primary"])
                    
                with gr.Accordion("📹 IZVORI: Live Nadzor i Kamere (Web kamera, IP, RTSP / Pojedinačna ili 2×2 Mreža)", open=False, elem_classes=["cyber-accordion"]):
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
                        label="Evidencija prolazaka",
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
                        label="24/7 NVR snimanje",
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
                        label="Anti-spoofing zaštita",
                        scale=2
                    )

                with gr.Row():
                    btn_launch_live = gr.Button("Pokreni Live Kameru", variant="primary", scale=2, elem_classes=["btn-industrial-primary"])
                    btn_launch_grid = gr.Button("Pokreni 2×2 Mrežu", variant="primary", scale=2, visible=False, elem_classes=["btn-industrial-primary"])
                    btn_open_snaps_quick = gr.Button("Snimke", variant="secondary", scale=1, elem_classes=["btn-industrial-secondary"])
                with gr.Row():
                    live_shortcuts_html = gr.HTML("""
                    <div class="industrial-shortcuts-card">
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
                    label="Prikaži i nepoznata lica"
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

    return {
        "input_img": input_img,
        "threshold_slider": threshold_slider,
        "landmarks_chk": landmarks_chk,
        "blur_mode_radio": blur_mode_radio,
        "btn_recognize": btn_recognize,
        "live_mode_radio": live_mode_radio,
        "single_cam_box": single_cam_box,
        "cam_source_type": cam_source_type,
        "cam_usb_idx": cam_usb_idx,
        "cam_rtsp_url": cam_rtsp_url,
        "cam_youtube_url": cam_youtube_url,
        "cam_video_file": cam_video_file,
        "cam_start_sec": cam_start_sec,
        "cam_enable_log": cam_enable_log,
        "cam_cooldown_sec": cam_cooldown_sec,
        "cam_enable_nvr": cam_enable_nvr,
        "cam_nvr_segment_min": cam_nvr_segment_min,
        "cam_enable_spoof": cam_enable_spoof,
        "btn_launch_live": btn_launch_live,
        "multi_cam_box": multi_cam_box,
        "mc_c1_on": mc_c1_on, "mc_c1_name": mc_c1_name, "mc_c1_src": mc_c1_src,
        "mc_c2_on": mc_c2_on, "mc_c2_name": mc_c2_name, "mc_c2_src": mc_c2_src,
        "mc_c3_on": mc_c3_on, "mc_c3_name": mc_c3_name, "mc_c3_src": mc_c3_src,
        "mc_c4_on": mc_c4_on, "mc_c4_name": mc_c4_name, "mc_c4_src": mc_c4_src,
        "btn_launch_grid": btn_launch_grid,
        "btn_open_snaps_quick": btn_open_snaps_quick,
        "cards_show_all_chk": cards_show_all_chk,
        "detection_cards_html": detection_cards_html,
        "annotated_out": annotated_out,
        "crops_gallery_out": crops_gallery_out,
        "results_table": results_table,
        "rec_status_md": rec_status_md,
        "unknown_face_dropdown": unknown_face_dropdown,
        "quick_name_input": quick_name_input,
        "btn_quick_add": btn_quick_add,
        "quick_add_status": quick_add_status,
    }

def wire_tab_live_events(c):
    rec_inputs = [c['input_img'], c['threshold_slider'], c['landmarks_chk'], c['blur_mode_radio'], c['cards_show_all_chk']]
    rec_outputs = [c['annotated_out'], c['crops_gallery_out'], c['results_table'], c['rec_status_md'], c['unknown_face_dropdown'], c['rec_faces_state'], c['detection_cards_html']]
    param_inputs = [c['input_img'], c['threshold_slider'], c['landmarks_chk'], c['blur_mode_radio'], c['cards_show_all_chk'], c['rec_faces_state']]

    c['btn_recognize'].click(
        fn=on_recognize_action,
        inputs=rec_inputs,
        outputs=rec_outputs
    )

    c['input_img'].upload(
        fn=on_recognize_action,
        inputs=rec_inputs,
        outputs=rec_outputs
    )

    c['landmarks_chk'].change(
        fn=on_analysis_param_change,
        inputs=param_inputs,
        outputs=rec_outputs
    )

    c['blur_mode_radio'].change(
        fn=on_analysis_param_change,
        inputs=param_inputs,
        outputs=rec_outputs
    )

    c['threshold_slider'].release(
        fn=on_analysis_param_change,
        inputs=param_inputs,
        outputs=rec_outputs
    )

    c['cards_show_all_chk'].change(
        fn=on_cards_filter_toggle,
        inputs=[c['rec_faces_state'], c['cards_show_all_chk']],
        outputs=[c['detection_cards_html']]
    )

    c['live_mode_radio'].change(
        fn=on_live_mode_change,
        inputs=[c['live_mode_radio']],
        outputs=[c['single_cam_box'], c['multi_cam_box'], c['btn_launch_live'], c['btn_launch_grid']]
    )

    c['cam_source_type'].change(
        fn=on_cam_source_change,
        inputs=[c['cam_source_type']],
        outputs=[c['cam_usb_idx'], c['cam_rtsp_url'], c['cam_youtube_url'], c['cam_video_file'], c['cam_start_sec'], c['btn_launch_live']]
    )

    c['cam_enable_log'].change(
        fn=lambda v: gr.update(visible=v),
        inputs=[c['cam_enable_log']],
        outputs=[c['cam_cooldown_sec']]
    )

    c['cam_enable_nvr'].change(
        fn=lambda v: gr.update(visible=v),
        inputs=[c['cam_enable_nvr']],
        outputs=[c['cam_nvr_segment_min']]
    )

    c['btn_launch_live'].click(
        fn=handle_launch_live,
        inputs=[c['cam_source_type'], c['cam_usb_idx'], c['cam_rtsp_url'], c['cam_youtube_url'], c['cam_video_file'], c['cam_start_sec'], c['cam_enable_log'], c['cam_cooldown_sec'], c['cam_enable_nvr'], c['cam_nvr_segment_min'], c['cam_enable_spoof']],
        outputs=[c['rec_status_md']]
    )

    c['btn_launch_grid'].click(
        fn=handle_launch_multicam,
        inputs=[
            c['mc_c1_on'], c['mc_c1_name'], c['mc_c1_src'],
            c['mc_c2_on'], c['mc_c2_name'], c['mc_c2_src'],
            c['mc_c3_on'], c['mc_c3_name'], c['mc_c3_src'],
            c['mc_c4_on'], c['mc_c4_name'], c['mc_c4_src'],
            c['threshold_slider'],
            c['cam_enable_log'],
            c['cam_cooldown_sec'],
            c['cam_enable_nvr'],
            c['cam_nvr_segment_min']
        ],
        outputs=[c['rec_status_md']]
    )

    c['crops_gallery_out'].select(
        fn=on_recognition_gallery_click,
        inputs=[c['rec_faces_state']],
        outputs=[c['unknown_face_dropdown'], c['quick_name_input']]
    )

    c['btn_quick_add'].click(
        fn=quick_add_face_to_db,
        inputs=[c['unknown_face_dropdown'], c['quick_name_input'], c['rec_faces_state']],
        outputs=[c['quick_add_status'], c['manage_person_dropdown'], c['existing_person_picker'], c['db_table'], c['db_stats_md']]
    ).then(
        fn=get_header_bar_html,
        outputs=[c['top_header_bar']]
    )

    c['btn_open_snaps_quick'].click(
        fn=lambda: config.open_folder_in_explorer()[1],
        outputs=[c['rec_status_md']]
    )
