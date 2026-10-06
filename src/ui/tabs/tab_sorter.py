# Tab 5: Pametni Sorter Fotografija
import os
import sys
import queue
import threading
import time
from typing import Optional, List, Dict, Any
import gradio as gr

import db
import config
import photo_sorter
from ui.common import APP_DIR

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



def create_tab_sorter():
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

    return {
        "tab_photo_sorter": tab_photo_sorter,
        "sorter_input_folder": sorter_input_folder,
        "btn_browse_input_folder": btn_browse_input_folder,
        "btn_check_input_folder": btn_check_input_folder,
        "sorter_folder_info_md": sorter_folder_info_md,
        "sorter_output_folder": sorter_output_folder,
        "btn_browse_output_folder": btn_browse_output_folder,
        "sorter_target_persons": sorter_target_persons,
        "sorter_similarity": sorter_similarity,
        "sorter_resolution": sorter_resolution,
        "sorter_action_mode": sorter_action_mode,
        "sorter_enable_combo": sorter_enable_combo,
        "sorter_enable_group": sorter_enable_group,
        "sorter_group_min": sorter_group_min,
        "sorter_enable_noface": sorter_enable_noface,
        "sorter_enable_unregistered": sorter_enable_unregistered,
        "btn_start_sorter": btn_start_sorter,
        "btn_cancel_sorter": btn_cancel_sorter,
        "btn_open_sorter_dir": btn_open_sorter_dir,
        "sorter_status_md": sorter_status_md,
        "sorter_stats_breakdown_md": sorter_stats_breakdown_md,
        "sorter_export_csv_file": sorter_export_csv_file,
    }

def wire_tab_sorter_events(c):
    c['btn_browse_input_folder'].click(
        fn=on_browse_input_folder,
        inputs=[c['sorter_input_folder']],
        outputs=[c['sorter_input_folder'], c['sorter_folder_info_md'], c['sorter_output_folder']]
    )

    c['btn_browse_output_folder'].click(
        fn=on_browse_output_folder,
        inputs=[c['sorter_output_folder']],
        outputs=[c['sorter_output_folder']]
    )

    c['btn_check_input_folder'].click(
        fn=handle_validate_input_folder,
        inputs=[c['sorter_input_folder']],
        outputs=[c['sorter_folder_info_md'], c['sorter_output_folder']]
    )

    c['btn_start_sorter'].click(
        fn=handle_start_photo_sorting,
        inputs=[
            c['sorter_input_folder'],
            c['sorter_output_folder'],
            c['sorter_target_persons'],
            c['sorter_similarity'],
            c['sorter_action_mode'],
            c['sorter_enable_combo'],
            c['sorter_enable_group'],
            c['sorter_group_min'],
            c['sorter_enable_noface'],
            c['sorter_enable_unregistered'],
            c['sorter_resolution']
        ],
        outputs=[
            c['sorter_status_md'],
            c['sorter_stats_breakdown_md'],
            c['sorter_export_csv_file'],
            c['btn_start_sorter'],
            c['btn_cancel_sorter'],
            c['btn_open_sorter_dir']
        ]
    )

    c['btn_cancel_sorter'].click(
        fn=handle_cancel_photo_sorting,
        inputs=[],
        outputs=[c['sorter_status_md']]
    )

    c['btn_open_sorter_dir'].click(
        fn=handle_open_sorter_folder,
        inputs=[c['sorter_output_folder']],
        outputs=[c['sorter_status_md']]
    )

    c['tab_photo_sorter'].select(
        fn=lambda: gr.update(choices=get_sorter_person_choices()),
        inputs=[],
        outputs=[c['sorter_target_persons']]
    )
