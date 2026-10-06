import os
import sys
import atexit
import gradio as gr

# Osiguraj da je src mapa u sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import db
import config
import hardware
import face_engine

# Zajedničke putanje i funkcije za kompatibilnost
from ui.common import (
    APP_DIR,
    DATA_DIR,
    UPLOADS_DIR,
    CROPS_DIR,
    SNAPSHOTS_DIR,
    imread_unicode,
    imwrite_unicode,
    save_image_dedup,
)

# Teme, stilovi i HTML zaglavlja
from ui.theme import (
    CUSTOM_CSS,
    HEAD_DARK_JS,
    custom_theme,
    get_header_bar_html,
    format_role_badge,
    get_profile_badge,
    generate_detection_cards_html,
    render_person_avatar_html,
)

# Pojedinačni tab moduli
from ui.tabs.tab_live import create_tab_live, wire_tab_live_events, _cleanup_active_processes
from ui.tabs.tab_database import (
    create_tab_database,
    wire_tab_database_events,
    refresh_database_view,
    update_both_person_dropdowns,
    get_person_dropdown_choices,
)
from ui.tabs.tab_media import create_tab_media, wire_tab_media_events
from ui.tabs.tab_events import create_tab_events, wire_tab_events_events
from ui.tabs.tab_sorter import create_tab_sorter, wire_tab_sorter_events
from ui.tabs.tab_settings import create_tab_settings, wire_tab_settings_events

# Registracija atexit čišćenja pozadinskih kamera
atexit.register(_cleanup_active_processes)


def build_app() -> gr.Blocks:
    """
    Kreira i sklapa Gradio sučelje sa svim tabovima i povezanim događajima.
    """
    with gr.Blocks(title="Argusface - Sustav za Prepoznavanje Lica") as demo_app:
        # Per-session stanje za prepoznata lica
        rec_faces_state = gr.State([])

        top_header_bar = gr.HTML(get_header_bar_html(), elem_id="top_header_bar")

        with gr.Tabs() as main_tabs:
            comps_live = create_tab_live(rec_faces_state)
            comps_db = create_tab_database()
            comps_media = create_tab_media()
            comps_events = create_tab_events()
            comps_sorter = create_tab_sorter()
            comps_settings = create_tab_settings()

        # Objedinjeni rječnik komponenti za međusobno povezivanje događaja
        all_comps = {
            **comps_live,
            **comps_db,
            **comps_media,
            **comps_events,
            **comps_sorter,
            **comps_settings,
            "top_header_bar": top_header_bar,
            "main_tabs": main_tabs,
            "rec_faces_state": rec_faces_state,
        }

        # Povezivanje događaja za svaki tab
        wire_tab_live_events(all_comps)
        wire_tab_database_events(all_comps)
        wire_tab_media_events(all_comps)
        wire_tab_events_events(all_comps)
        wire_tab_sorter_events(all_comps)
        wire_tab_settings_events(all_comps)

        # Globalni događaji: osvježavanje zaglavlja i početno učitavanje baze
        main_tabs.select(
            fn=get_header_bar_html,
            outputs=[top_header_bar]
        )

        demo_app.load(
            fn=get_header_bar_html,
            outputs=[top_header_bar]
        ).then(
            fn=refresh_database_view,
            outputs=[all_comps["db_table"], all_comps["db_stats_md"]]
        ).then(
            fn=update_both_person_dropdowns,
            outputs=[all_comps["manage_person_dropdown"], all_comps["existing_person_picker"]]
        )

    return demo_app


# Globalni demo objekt za izravno pokretanje ili import
demo = build_app()


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
