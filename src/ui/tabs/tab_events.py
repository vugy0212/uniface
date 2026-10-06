# Tab 4: Dnevnik Prolazaka i Evidencija
import os
import sys
import html
import datetime
import gradio as gr

import db
import config
from ui.common import DATA_DIR
from ui.tabs.tab_media import handle_open_external_video

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



def create_tab_events():
    with gr.TabItem("Dnevnik Prolazaka") as tab_events:
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

    return {
        "tab_events": tab_events,
        "events_stats_md": events_stats_md,
        "selected_event_id_state": selected_event_id_state,
        "events_search_input": events_search_input,
        "btn_refresh_events": btn_refresh_events,
        "btn_clear_events": btn_clear_events,
        "events_table": events_table,
        "event_camera_preview": event_camera_preview,
        "event_crop_preview": event_crop_preview,
        "event_details_md": event_details_md,
        "btn_delete_single_event": btn_delete_single_event,
        "event_video_player": event_video_player,
        "btn_export_events_csv": btn_export_events_csv,
        "events_export_file": events_export_file,
        "events_status_md": events_status_md,
        "clear_events_modal": clear_events_modal,
        "clear_events_modal_stats": clear_events_modal_stats,
        "clear_events_confirm_cb": clear_events_confirm_cb,
        "clear_events_modal_error": clear_events_modal_error,
        "btn_cancel_clear_events": btn_cancel_clear_events,
        "btn_confirm_clear_events": btn_confirm_clear_events,
        "single_event_delete_modal": single_event_delete_modal,
        "single_event_delete_summary": single_event_delete_summary,
        "btn_cancel_single_delete": btn_cancel_single_delete,
        "btn_confirm_single_delete": btn_confirm_single_delete,
    }

def wire_tab_events_events(c):
    events_out_all = [
        c['events_stats_md'], c['events_table'], c['event_camera_preview'], c['event_crop_preview'], 
        c['event_details_md'], c['event_video_player'], c['selected_event_id_state'], c['btn_delete_single_event']
    ]

    c['events_search_input'].change(
        fn=on_events_search,
        inputs=[c['events_search_input']],
        outputs=events_out_all
    )

    c['btn_refresh_events'].click(
        fn=on_events_search,
        inputs=[c['events_search_input']],
        outputs=events_out_all
    )

    c['tab_events'].select(
        fn=on_events_search,
        inputs=[c['events_search_input']],
        outputs=events_out_all
    )

    c['events_table'].select(
        fn=on_event_select,
        inputs=[c['events_search_input']],
        outputs=[c['event_camera_preview'], c['event_crop_preview'], c['event_details_md'], c['event_video_player'], c['selected_event_id_state'], c['btn_delete_single_event']]
    )

    c['btn_clear_events'].click(
        fn=handle_open_clear_events_modal,
        outputs=[c['clear_events_modal'], c['clear_events_modal_stats'], c['clear_events_confirm_cb'], c['clear_events_modal_error']]
    )

    c['btn_cancel_clear_events'].click(
        fn=handle_close_clear_events_modal,
        outputs=[c['clear_events_modal'], c['clear_events_confirm_cb'], c['clear_events_modal_error']]
    )

    c['btn_confirm_clear_events'].click(
        fn=handle_execute_clear_events,
        inputs=[c['clear_events_confirm_cb'], c['events_search_input']],
        outputs=[c['clear_events_modal'], c['clear_events_modal_error'], c['events_stats_md'], c['events_table'], c['event_camera_preview'], c['event_crop_preview'], c['event_details_md'], c['event_video_player'], c['selected_event_id_state'], c['btn_delete_single_event']]
    )

    c['btn_delete_single_event'].click(
        fn=handle_open_single_delete_event_modal,
        inputs=[c['selected_event_id_state']],
        outputs=[c['single_event_delete_modal'], c['single_event_delete_summary']]
    )

    c['btn_cancel_single_delete'].click(
        fn=handle_close_single_delete_event_modal,
        outputs=[c['single_event_delete_modal'], c['single_event_delete_summary']]
    )

    c['btn_confirm_single_delete'].click(
        fn=handle_execute_single_delete_event,
        inputs=[c['selected_event_id_state'], c['events_search_input']],
        outputs=[c['single_event_delete_modal'], c['events_stats_md'], c['events_table'], c['event_camera_preview'], c['event_crop_preview'], c['event_details_md'], c['event_video_player'], c['selected_event_id_state'], c['btn_delete_single_event']]
    )

    c['btn_export_events_csv'].click(
        fn=handle_export_events_csv,
        outputs=[c['events_export_file'], c['events_status_md']]
    )
