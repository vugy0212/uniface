# Tab 3: Spremljeni kadrovi (Snapshots) i NVR video arhiva
import os
import sys
import cv2
import time
import gradio as gr
from PIL import Image

import config
import db
from ui.theme import generate_detection_cards_html
from ui.common import APP_DIR, DATA_DIR, SNAPSHOTS_DIR
from ui.tabs.tab_live import recognize_faces

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



def create_tab_media():
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

    return {
        "tab_saved_media": tab_saved_media,
        "btn_open_snapshots_folder": btn_open_snapshots_folder,
        "btn_refresh_snapshots": btn_refresh_snapshots,
        "snapshots_gallery": snapshots_gallery,
        "snapshots_table": snapshots_table,
        "selected_snap_preview": selected_snap_preview,
        "selected_snap_dropdown": selected_snap_dropdown,
        "selected_snap_info": selected_snap_info,
        "btn_send_to_rec": btn_send_to_rec,
        "btn_delete_snap": btn_delete_snap,
        "custom_snap_dir_input": custom_snap_dir_input,
        "btn_save_snap_dir": btn_save_snap_dir,
        "btn_reset_snap_dir": btn_reset_snap_dir,
        "snap_dir_status_md": snap_dir_status_md,
        "snapshots_info_md": snapshots_info_md,
        "snap_action_status": snap_action_status,
        "btn_refresh_nvr_tab": btn_refresh_nvr_tab,
        "btn_open_nvr_folder": btn_open_nvr_folder,
        "nvr_storage_status_md": nvr_storage_status_md,
        "nvr_archive_table": nvr_archive_table,
        "nvr_video_preview": nvr_video_preview,
        "nvr_selected_info_md": nvr_selected_info_md,
        "btn_open_video_player": btn_open_video_player,
        "btn_delete_video_seg": btn_delete_video_seg,
        "nvr_action_status_md": nvr_action_status_md,
    }

def wire_tab_media_events(c):
    c['btn_open_snapshots_folder'].click(
        fn=lambda: config.open_folder_in_explorer()[1],
        outputs=[c['snap_action_status']]
    )

    c['btn_refresh_snapshots'].click(
        fn=get_snapshots_ui_data,
        outputs=[c['snapshots_gallery'], c['snapshots_table'], c['snapshots_info_md'], c['selected_snap_dropdown'], c['selected_snap_preview'], c['selected_snap_info']]
    )

    c['snapshots_gallery'].select(
        fn=on_snapshot_gallery_select,
        outputs=[c['selected_snap_preview'], c['selected_snap_info'], c['selected_snap_dropdown']]
    )

    c['selected_snap_dropdown'].change(
        fn=on_snapshot_dropdown_change,
        inputs=[c['selected_snap_dropdown']],
        outputs=[c['selected_snap_preview'], c['selected_snap_info']]
    )

    c['btn_delete_snap'].click(
        fn=on_delete_snapshot_click,
        inputs=[c['selected_snap_dropdown']],
        outputs=[c['snap_action_status'], c['snapshots_gallery'], c['snapshots_table'], c['snapshots_info_md'], c['selected_snap_dropdown'], c['selected_snap_preview'], c['selected_snap_info']]
    )

    c['btn_save_snap_dir'].click(
        fn=on_save_snapshot_dir_click,
        inputs=[c['custom_snap_dir_input']],
        outputs=[c['snap_dir_status_md'], c['snapshots_info_md'], c['snapshots_gallery'], c['snapshots_table'], c['selected_snap_dropdown'], c['selected_snap_preview'], c['selected_snap_info']]
    )

    c['btn_reset_snap_dir'].click(
        fn=on_reset_snapshot_dir_click,
        outputs=[c['snap_dir_status_md'], c['custom_snap_dir_input'], c['snapshots_info_md'], c['snapshots_gallery'], c['snapshots_table'], c['selected_snap_dropdown'], c['selected_snap_preview'], c['selected_snap_info']]
    )

    c['btn_send_to_rec'].click(
        fn=on_send_snapshot_to_recognition,
        inputs=[c['selected_snap_dropdown'], c['threshold_slider'], c['landmarks_chk'], c['blur_mode_radio'], c['cards_show_all_chk']],
        outputs=[c['input_img'], c['annotated_out'], c['crops_gallery_out'], c['results_table'], c['rec_status_md'], c['unknown_face_dropdown'], c['rec_faces_state'], c['detection_cards_html'], c['snap_action_status']]
    )

    c['nvr_archive_table'].select(
        fn=on_nvr_segment_select,
        outputs=[c['nvr_video_preview'], c['nvr_selected_info_md']]
    )

    c['btn_refresh_nvr_tab'].click(
        fn=get_nvr_archive_ui_data,
        outputs=[c['nvr_storage_status_md'], c['nvr_archive_table'], c['nvr_video_preview'], c['nvr_selected_info_md']]
    )

    c['btn_open_nvr_folder'].click(
        fn=handle_open_nvr_folder,
        outputs=[c['nvr_action_status_md']]
    )

    c['btn_open_video_player'].click(
        fn=handle_open_external_video,
        inputs=[c['nvr_video_preview']],
        outputs=[c['nvr_action_status_md']]
    )

    c['btn_delete_video_seg'].click(
        fn=handle_delete_nvr_segment,
        inputs=[c['nvr_video_preview']],
        outputs=[c['nvr_action_status_md'], c['nvr_storage_status_md'], c['nvr_archive_table'], c['nvr_video_preview'], c['nvr_selected_info_md']]
    )

    c['tab_saved_media'].select(
        fn=get_snapshots_ui_data,
        outputs=[c['snapshots_gallery'], c['snapshots_table'], c['snapshots_info_md'], c['selected_snap_dropdown'], c['selected_snap_preview'], c['selected_snap_info']]
    ).then(
        fn=get_nvr_archive_ui_data,
        outputs=[c['nvr_storage_status_md'], c['nvr_archive_table'], c['nvr_video_preview'], c['nvr_selected_info_md']]
    )
