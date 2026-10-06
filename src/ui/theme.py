# UI Theme, Styles, Head Scripts and Card Generators
import os
import html
import base64
import datetime
import cv2
import numpy as np
import gradio as gr
import hardware
import db

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


def generate_detection_cards_html(results, show_all_faces=False):
    import base64
    import datetime
    now_str = datetime.datetime.now().strftime("%H:%M:%S")

    if not results:
        return f"""
        <div class="detection-panel-inner">
            <div class="panel-header">
                <div class="panel-title">
                    <span class="status-indicator-dot"></span> Log Detekcija
                </div>
                <span class="panel-badge">Čekanje</span>
            </div>
            <div class="detection-empty-state">
                <div class="empty-icon-box">
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#6E7681" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <polyline points="12 6 12 12 16 14"></polyline>
                    </svg>
                </div>
                <div class="empty-title">Nema novih detekcija</div>
                <div class="empty-sub">Učitajte sliku ili pokrenite live kameru za analizu.</div>
            </div>
        </div>
        """

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
                    <span class="status-indicator-dot active"></span> Log Detekcija
                </div>
                <span class="panel-badge">{badge_text}</span>
            </div>
            <div class="detection-empty-state">
                <div class="empty-title" style="color: #F59E0B;">Nema prepoznatih lica</div>
                <div class="empty-sub">
                    Pronađeno je <b>{num_total}</b> lica, ali nijedno ne prelazi prag.<br>
                    Uključite <i>"Prikaži i nepoznata lica"</i> za detalje.
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
        else:
            try:
                pct = float(str(sim_val).replace('%', '').strip())
            except Exception:
                pct = 0.0

        status = r.get("status", "Nepoznat")
        name = r.get("best_name", "Nepoznata osoba")

        # Determine person role badge
        p_role = "standard"
        if status in ("Prepoznat", "Moguće poklapanje") and name not in ("Nepoznata osoba", "Nepoznato"):
            try:
                p_role = db.get_person_role(name)
            except Exception:
                p_role = "standard"

        stroke_color = "#10B981"
        if p_role == "blacklist":
            badge_cls = "match-blacklist"
            role_badge = '<span class="role-pill role-blacklist">CRNA LISTA</span>'
            stroke_color = "#EF4444"
        elif p_role == "vip":
            badge_cls = "match-vip"
            role_badge = '<span class="role-pill role-vip">VIP</span>'
            stroke_color = "#38BDF8"
        elif status == "Prepoznat":
            badge_cls = "match-success"
            role_badge = '<span class="role-pill role-verified">OK</span>'
            stroke_color = "#10B981"
        elif status == "Moguće poklapanje":
            badge_cls = "match-warning"
            role_badge = '<span class="role-pill role-possible">PROVJERA</span>'
            stroke_color = "#F59E0B"
        else:
            badge_cls = "match-unknown"
            role_badge = ''
            stroke_color = "#6E7681"

        safe_name = html.escape(str(name))
        safe_age = html.escape(str(r.get("age", "-")))
        safe_gender = html.escape(str(r.get("gender", "-")))
        meta_parts = [f'<span class="meta-time">{html.escape(now_str)}</span>']
        if safe_age and safe_age != "-":
            meta_parts.append(f'<span>~{safe_age}g</span>')
        if safe_gender and safe_gender != "-":
            meta_parts.append(f'<span>{safe_gender}</span>')
        meta_html = ' <span class="meta-sep">•</span> '.join(meta_parts)

        circ_svg = f"""
        <div class="circle-gauge" title="Pouzdanost: {pct:.1f}%">
            <svg viewBox="0 0 36 36" class="circular-chart">
                <path class="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                <path class="circle-val" stroke="{stroke_color}" stroke-dasharray="{min(100.0, max(0.0, pct)):.0f}, 100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"/>
                <text x="18" y="20.5" class="percentage">{pct:.0f}%</text>
            </svg>
        </div>
        """

        cards.append(f"""
        <div class="cyber-detection-card {badge_cls}">
            <div class="card-avatar-wrap">
                <img src="{img_src}" class="card-avatar" alt="{safe_name}" />
            </div>
            <div class="card-details">
                <div class="card-name-row">
                    <span class="card-name" title="{safe_name}">{safe_name}</span>
                    {role_badge}
                </div>
                <div class="card-meta">
                    {meta_html}
                </div>
            </div>
            {circ_svg}
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
                <span class="status-indicator-dot active"></span> Log Detekcija
            </div>
            <span class="panel-badge active">{badge_text}</span>
        </div>
        <div class="cyber-cards-scroll">
            {cards_html}
        </div>
    </div>
    """

# ---------------- PREPOZNAVANJE ----------------

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


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root, :root.dark, :root.light, html, body {
    --ind-bg: #0D1117;
    --ind-card: #161B22;
    --ind-card-hover: #1C2128;
    --ind-border: #262C36;
    --ind-border-subtle: #21262D;
    --ind-blue: #2563EB;
    --ind-blue-hover: #1D4ED8;
    --ind-emerald: #10B981;
    --ind-amber: #F59E0B;
    --ind-red: #EF4444;
    --text-primary: #F0F6FC;
    --text-secondary: #C9D1D9;
    --text-muted: #8B949E;

    /* Enforce Dark Theme Tokens on Gradio internals */
    --background-fill-primary: #0D1117 !important;
    --background-fill-secondary: #161B22 !important;
    --block-background-fill: #161B22 !important;
    --block-border-color: #262C36 !important;
    --block-radius: 6px !important;
    --container-radius: 6px !important;
    --block-label-background-fill: #161B22 !important;
    --block-label-text-color: #8B949E !important;
    --block-title-text-color: #F0F6FC !important;
    --body-text-color: #C9D1D9 !important;
    --body-text-color-subdued: #8B949E !important;
    --input-background-fill: #0D1117 !important;
    --input-border-color: #262C36 !important;
    --input-radius: 6px !important;
    --input-placeholder-color: #6E7681 !important;
    --checkbox-background-color: #21262D !important;
    --checkbox-background-color-selected: #2563EB !important;
    --checkbox-border-color: #30363D !important;
    --checkbox-border-color-selected: #3B82F6 !important;
    --checkbox-label-background-fill: transparent !important;
    --checkbox-label-text-color: #C9D1D9 !important;
    --table-even-background-fill: #161B22 !important;
    --table-odd-background-fill: #161B22 !important;
    --table-text-color: #C9D1D9 !important;
    --table-border-color: #21262D !important;
    --border-color-primary: #262C36 !important;
    color-scheme: dark !important;
}

body, html {
    background-color: #0D1117 !important;
    background: #0D1117 !important;
    color: #C9D1D9 !important;
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
    margin: 0;
    padding: 0;
}

/* Custom Minimalist Scrollbar */
::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
::-webkit-scrollbar-track {
    background: #0D1117 !important;
}
::-webkit-scrollbar-thumb {
    background: #262C36 !important;
    border-radius: 3px !important;
}
::-webkit-scrollbar-thumb:hover {
    background: #30363D !important;
}

.gradio-container {
    background-color: #0D1117 !important;
    background: #0D1117 !important;
    color: #C9D1D9 !important;
    max-width: 100% !important;
    padding: 12px !important;
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* 1. Header Bar: Fixed, Minimalist, Matte #161B22 */
.industrial-header-bar,
.cyber-header-bar {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    background: #161B22 !important;
    background-color: #161B22 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    margin-bottom: 12px !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    box-shadow: none !important;
}

.header-left {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
}

.header-brand {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #F0F6FC !important;
    letter-spacing: -0.01em !important;
}

.header-tagline {
    font-size: 0.78rem !important;
    color: #8B949E !important;
    padding-left: 8px !important;
    border-left: 1px solid #262C36 !important;
}

.header-middle {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    flex-wrap: wrap !important;
}

.status-pill {
    display: inline-flex !important;
    align-items: center !important;
    gap: 6px !important;
    background: #21262D !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    padding: 3px 10px !important;
    font-size: 0.78rem !important;
    color: #C9D1D9 !important;
    font-weight: 500 !important;
}

.status-dot {
    width: 6px !important;
    height: 6px !important;
    border-radius: 50% !important;
    flex-shrink: 0 !important;
}

.status-dot.green { background-color: #10B981 !important; }
.status-dot.blue { background-color: #2563EB !important; }
.status-dot.slate { background-color: #8B949E !important; }

/* 2. Flat Navigation Tabs */
.tabs > .tab-nav,
div[role="tablist"] {
    background: #161B22 !important;
    border-radius: 6px !important;
    padding: 4px !important;
    border: 1px solid #262C36 !important;
    gap: 4px !important;
    margin-bottom: 12px !important;
    display: flex !important;
    box-shadow: none !important;
}

.tabs > .tab-nav > button,
div[role="tablist"] button,
button[role="tab"] {
    color: #8B949E !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 6px !important;
    padding: 6px 14px !important;
    border: 1px solid transparent !important;
    background: transparent !important;
    transition: all 0.15s ease !important;
    box-shadow: none !important;
}

.tabs > .tab-nav > button:hover,
div[role="tablist"] button:hover,
button[role="tab"]:hover {
    color: #C9D1D9 !important;
    background: #21262D !important;
    border-color: #30363D !important;
    transform: none !important;
}

.tabs > .tab-nav > button.selected,
div[role="tablist"] button.selected,
div[role="tablist"] button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    color: #F0F6FC !important;
    background: #21262D !important;
    border: 1px solid #30363D !important;
    font-weight: 600 !important;
    box-shadow: none !important;
}

.tabs > .tab-nav > button.selected::after,
div[role="tablist"] button[aria-selected="true"]::after {
    display: none !important;
}

/* 3. Containers, Cards & Panels - Matte #161B22, 6px radius */
.cyber-card,
.block,
fieldset.block,
div.block,
.gr-box,
.panel {
    background: #161B22 !important;
    background-color: #161B22 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    padding: 12px !important;
    box-shadow: none !important;
    transition: none !important;
}

.cyber-card:hover,
.block:hover {
    border-color: #262C36 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* 4. Video Viewport (Zone B) - Pure Matte Black */
.cyber-preview-frame,
div[data-testid="image"],
div[data-testid="image"] > div,
.image-container,
.upload-container,
.empty,
.drop-zone {
    background-color: #000000 !important;
    background: #000000 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
    color: #C9D1D9 !important;
}

div[data-testid="image"] button,
.upload-container button {
    background: #21262D !important;
    border: 1px solid #30363D !important;
    color: #C9D1D9 !important;
    border-radius: 6px !important;
}

/* 5. Switch-Style Checkboxes (Zone A) */
.gradio-checkbox,
label:has(input[type="checkbox"]) {
    display: flex !important;
    align-items: center !important;
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    padding: 4px 0 !important;
    cursor: pointer !important;
    color: #C9D1D9 !important;
    box-shadow: none !important;
}

input[type="checkbox"] {
    -webkit-appearance: none !important;
    -moz-appearance: none !important;
    appearance: none !important;
    width: 32px !important;
    height: 18px !important;
    min-width: 32px !important;
    min-height: 18px !important;
    max-width: 32px !important;
    max-height: 18px !important;
    background: #21262D !important;
    background-color: #21262D !important;
    border: 1px solid #30363D !important;
    border-radius: 10px !important;
    position: relative !important;
    cursor: pointer !important;
    outline: none !important;
    transition: all 0.18s ease !important;
    margin: 0 10px 0 0 !important;
    flex-shrink: 0 !important;
    box-shadow: none !important;
}

input[type="checkbox"]::before {
    content: "" !important;
    position: absolute !important;
    top: 2px !important;
    left: 2px !important;
    width: 12px !important;
    height: 12px !important;
    background: #8B949E !important;
    border-radius: 50% !important;
    transition: all 0.18s ease !important;
}

input[type="checkbox"]:checked {
    background: #2563EB !important;
    background-color: #2563EB !important;
    border-color: #3B82F6 !important;
    background-image: none !important;
}

input[type="checkbox"]:checked::before {
    transform: translateX(14px) !important;
    background: #FFFFFF !important;
}

/* 6. Inputs, Dropdowns & Controls */
input[type="text"],
input[type="number"],
textarea,
select,
.dropdown,
.select,
div[data-testid="dropdown"] {
    background: #0D1117 !important;
    background-color: #0D1117 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    color: #F0F6FC !important;
    font-size: 0.85rem !important;
    box-shadow: none !important;
}

input::placeholder,
textarea::placeholder {
    color: #6E7681 !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #2563EB !important;
    outline: none !important;
}

ul.options, .options-wrap {
    background: #161B22 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    color: #F0F6FC !important;
}

ul.options li:hover, ul.options li.selected {
    background: #21262D !important;
    color: #F0F6FC !important;
}

/* 7. Buttons: Primary (#2563EB) & Secondary (#21262D) */
.btn-industrial-primary,
.btn-cyber-primary,
button.primary {
    background: #2563EB !important;
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    border: 1px solid #3B82F6 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
    cursor: pointer !important;
}

.btn-industrial-primary:hover,
.btn-cyber-primary:hover,
button.primary:hover {
    background: #1D4ED8 !important;
    background-color: #1D4ED8 !important;
    border-color: #2563EB !important;
    transform: none !important;
    box-shadow: none !important;
}

.btn-industrial-secondary,
.btn-cyber-secondary,
.btn-cyber-live,
button.secondary {
    background: #21262D !important;
    background-color: #21262D !important;
    color: #C9D1D9 !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    padding: 8px 16px !important;
    box-shadow: none !important;
    transition: background-color 0.15s ease !important;
    cursor: pointer !important;
}

.btn-industrial-secondary:hover,
.btn-cyber-secondary:hover,
.btn-cyber-live:hover,
button.secondary:hover {
    background: #30363D !important;
    background-color: #30363D !important;
    color: #F0F6FC !important;
    border-color: #30363D !important;
    transform: none !important;
}

.btn-cyber-danger,
button.stop {
    background: #21262D !important;
    background-color: #21262D !important;
    color: #F85149 !important;
    border: 1px solid rgba(248, 81, 73, 0.35) !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

.btn-cyber-danger:hover,
button.stop:hover {
    background: #B62324 !important;
    background-color: #B62324 !important;
    color: #FFFFFF !important;
    border-color: #B62324 !important;
}

/* 8. Industrial Tables - NO vertical lines, horizontal 1px #21262D, 8px+ padding */
table,
.dataframe,
.table-wrap,
div[data-testid="table"],
div[data-testid="dataframe"] {
    background: #161B22 !important;
    background-color: #161B22 !important;
    color: #C9D1D9 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    border-collapse: collapse !important;
    overflow: hidden !important;
}

thead, th, .dataframe thead tr th {
    background: #161B22 !important;
    background-color: #161B22 !important;
    color: #8B949E !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 10px 14px !important;
    border: none !important;
    border-bottom: 1px solid #21262D !important;
    border-left: none !important;
    border-right: none !important;
}

tbody tr, .dataframe tbody tr {
    background: #161B22 !important;
    background-color: #161B22 !important;
    color: #C9D1D9 !important;
    border: none !important;
    border-bottom: 1px solid #21262D !important;
    border-left: none !important;
    border-right: none !important;
    transition: background-color 0.15s ease !important;
}

tbody tr:nth-child(even), .dataframe tbody tr:nth-child(even) {
    background: #161B22 !important;
    background-color: #161B22 !important;
}

tbody tr:hover, .dataframe tbody tr:hover {
    background: #1C2128 !important;
    background-color: #1C2128 !important;
}

td, .dataframe tbody tr td {
    color: #C9D1D9 !important;
    padding: 10px 14px !important;
    border: none !important;
    border-bottom: 1px solid #21262D !important;
    border-left: none !important;
    border-right: none !important;
    font-size: 0.85rem !important;
}

/* 9. Zone C: Clean Log Card & Circular Confidence Gauge */
.detection-panel-inner {
    display: flex !important;
    flex-direction: column !important;
    gap: 8px !important;
}

.panel-header {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid #262C36 !important;
}

.panel-title {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    color: #F0F6FC !important;
}

.status-indicator-dot {
    width: 7px !important;
    height: 7px !important;
    border-radius: 50% !important;
    background: #8B949E !important;
}

.status-indicator-dot.active {
    background: #10B981 !important;
}

.panel-badge {
    font-size: 0.72rem !important;
    background: #21262D !important;
    border: 1px solid #30363D !important;
    border-radius: 6px !important;
    padding: 2px 8px !important;
    color: #8B949E !important;
}

.panel-badge.active {
    color: #C9D1D9 !important;
}

.cyber-cards-scroll {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
    max-height: 480px !important;
    overflow-y: auto !important;
}

.cyber-detection-card {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    background: #161B22 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    padding: 8px 10px !important;
    transition: background-color 0.15s ease !important;
    box-shadow: none !important;
}

.cyber-detection-card:hover {
    background: #1C2128 !important;
    transform: none !important;
}

.cyber-detection-card.match-success {
    border-left: 3px solid #10B981 !important;
}

.cyber-detection-card.match-warning {
    border-left: 3px solid #F59E0B !important;
}

.cyber-detection-card.match-unknown {
    border-left: 3px solid #6E7681 !important;
}

.cyber-detection-card.match-blacklist {
    border-left: 3px solid #EF4444 !important;
}

.cyber-detection-card.match-vip {
    border-left: 3px solid #38BDF8 !important;
}

.card-avatar-wrap {
    width: 38px !important;
    height: 38px !important;
    flex-shrink: 0 !important;
}

.card-avatar {
    width: 38px !important;
    height: 38px !important;
    border-radius: 6px !important;
    object-fit: cover !important;
    background: #21262D !important;
    border: 1px solid #262C36 !important;
}

.card-details {
    flex: 1 !important;
    min-width: 0 !important;
    display: flex !important;
    flex-direction: column !important;
    gap: 2px !important;
}

.card-name-row {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.card-name {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #F0F6FC !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

.card-meta {
    display: flex !important;
    align-items: center !important;
    gap: 4px !important;
    font-size: 0.72rem !important;
    color: #8B949E !important;
}

.meta-time {
    color: #8B949E !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.meta-sep {
    color: #484F58 !important;
}

.role-pill {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    padding: 1px 5px !important;
    border-radius: 4px !important;
    letter-spacing: 0.02em !important;
}

.role-verified { background: #0E4429 !important; color: #3FB950 !important; }
.role-possible { background: #4D2D00 !important; color: #E3B341 !important; }
.role-vip { background: #1F3B66 !important; color: #58A6FF !important; }
.role-blacklist { background: #490202 !important; color: #F85149 !important; }

/* Circular Percentage Gauge */
.circle-gauge {
    width: 34px !important;
    height: 34px !important;
    flex-shrink: 0 !important;
}

.circular-chart {
    display: block !important;
    margin: 0 auto !important;
    max-width: 100% !important;
    max-height: 100% !important;
}

.circle-bg {
    fill: none !important;
    stroke: #21262D !important;
    stroke-width: 3.5 !important;
}

.circle-val {
    fill: none !important;
    stroke-width: 3.5 !important;
    stroke-linecap: round !important;
    transition: stroke-dasharray 0.3s ease !important;
}

.percentage {
    fill: #C9D1D9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 600 !important;
    text-anchor: middle !important;
}

.detection-empty-state {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 28px 12px !important;
    text-align: center !important;
}

.empty-icon-box {
    margin-bottom: 10px !important;
}

.empty-title {
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: #C9D1D9 !important;
    margin-bottom: 4px !important;
}

.empty-sub {
    font-size: 0.75rem !important;
    color: #8B949E !important;
    line-height: 1.4 !important;
}

/* 10. Clean Shortcuts Card */
.industrial-shortcuts-card,
.industrial-shortcuts-card {
    background: #161B22 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
    margin-top: 8px !important;
    box-shadow: none !important;
}

.shortcuts-header {
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    margin-bottom: 8px !important;
}

.shortcuts-badge {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    background: #21262D !important;
    border: 1px solid #30363D !important;
    color: #8B949E !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

.shortcuts-title {
    font-size: 0.78rem !important;
    color: #8B949E !important;
}

.shortcuts-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 6px 12px !important;
    font-size: 0.75rem !important;
    color: #C9D1D9 !important;
}

.sc-item {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

.sc-item kbd {
    background: #21262D !important;
    border: 1px solid #30363D !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 0.7rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #F0F6FC !important;
}

/* Accordion */
.cyber-accordion,
details {
    background: #161B22 !important;
    border: 1px solid #262C36 !important;
    border-radius: 6px !important;
    box-shadow: none !important;
}

summary {
    color: #C9D1D9 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
}

/* Compact File Component for Backup & Restore */
.compact-file {
    min-height: 80px !important;
}
.compact-file > .empty,
.compact-file .file-upload,
.compact-file [data-testid="file-upload"] {
    min-height: 75px !important;
    padding: 8px 12px !important;
}
.compact-file svg {
    width: 22px !important;
    height: 22px !important;
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
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate"
).set(
    body_background_fill="#0D1117",
    body_background_fill_dark="#0D1117",
    background_fill_primary="#0D1117",
    background_fill_primary_dark="#0D1117",
    background_fill_secondary="#161B22",
    background_fill_secondary_dark="#161B22",
    block_background_fill="#161B22",
    block_background_fill_dark="#161B22",
    block_border_color="#262C36",
    block_border_color_dark="#262C36",
    block_radius="6px",
    container_radius="6px",
    block_label_background_fill="#161B22",
    block_label_background_fill_dark="#161B22",
    block_label_text_color="#8B949E",
    block_label_text_color_dark="#8B949E",
    block_title_text_color="#F0F6FC",
    block_title_text_color_dark="#F0F6FC",
    body_text_color="#C9D1D9",
    body_text_color_dark="#C9D1D9",
    body_text_color_subdued="#8B949E",
    body_text_color_subdued_dark="#8B949E",
    input_background_fill="#0D1117",
    input_background_fill_dark="#0D1117",
    input_border_color="#262C36",
    input_border_color_dark="#262C36",
    input_placeholder_color="#6E7681",
    input_placeholder_color_dark="#6E7681",
    checkbox_background_color="#21262D",
    checkbox_background_color_selected="#2563EB",
    checkbox_background_color_dark="#21262D",
    checkbox_background_color_selected_dark="#2563EB",
    checkbox_border_color="#30363D",
    checkbox_border_color_selected="#3B82F6",
    checkbox_border_color_dark="#30363D",
    checkbox_border_color_selected_dark="#3B82F6",
    checkbox_label_background_fill="transparent",
    checkbox_label_background_fill_dark="transparent",
    checkbox_label_text_color="#C9D1D9",
    checkbox_label_text_color_dark="#C9D1D9",
    accordion_text_color="#F0F6FC",
    accordion_text_color_dark="#F0F6FC",
    table_even_background_fill="#161B22",
    table_even_background_fill_dark="#161B22",
    table_odd_background_fill="#161B22",
    table_odd_background_fill_dark="#161B22",
    table_text_color="#C9D1D9",
    table_text_color_dark="#C9D1D9",
    button_primary_background_fill="#2563EB",
    button_primary_background_fill_dark="#2563EB",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_secondary_background_fill="#21262D",
    button_secondary_background_fill_dark="#21262D",
    button_secondary_text_color="#C9D1D9",
    button_secondary_text_color_dark="#C9D1D9"
)


def get_header_bar_html():
    hw_name, _ = hardware.get_onnx_acceleration_status()
    db_stats = db.get_stats()
    p_count = db_stats.get("total_persons", 0)
    safe_hw = html.escape(str(hw_name))
    return f"""
    <div class="industrial-header-bar">
        <div class="header-left">
            <span class="header-brand">ArgusFace</span>
            <span class="header-tagline">Industrial VMS</span>
        </div>
        <div class="header-middle">
            <div class="status-pill">
                <span class="status-dot green"></span>
                <span>100% Lokalno</span>
            </div>
            <div class="status-pill">
                <span class="status-dot blue"></span>
                <span>{safe_hw} Aktivno</span>
            </div>
            <div class="status-pill">
                <span class="status-dot slate"></span>
                <span>Baza: {p_count} osoba</span>
            </div>
        </div>
    </div>
    """


