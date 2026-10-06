import os
import time
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
import cv2
import numpy as np

try:
    from . import config
    from . import db
except ImportError:
    import config
    import db

# Global alert cooldown tracker: key -> (event_type, identifier) : float timestamp
_alert_history = {}
_alert_lock = threading.Lock()

def _parse_recipients(recipients_raw) -> list[str]:
    """Splits comma/semicolon/newline-delimited string of emails or chat IDs into clean list."""
    if not recipients_raw:
        return []
    if isinstance(recipients_raw, (list, tuple, set)):
        return [str(x).strip() for x in recipients_raw if str(x).strip()]
    raw_str = str(recipients_raw).replace(";", ",").replace("\n", ",")
    return [r.strip() for r in raw_str.split(",") if r.strip()]

def _encode_image_bytes(image_input) -> bytes | None:
    """Helper to convert numpy array, file path or raw bytes to JPEG byte string."""
    if isinstance(image_input, np.ndarray):
        success, enc = cv2.imencode(".jpg", image_input, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if success:
            return enc.tobytes()
    elif isinstance(image_input, (bytes, bytearray)):
        return bytes(image_input)
    elif isinstance(image_input, str) and os.path.isfile(image_input):
        try:
            with open(image_input, "rb") as f:
                return f.read()
        except Exception:
            return None
    return None

# =========================================================================
# TELEGRAM NOTIFICATIONS
# =========================================================================

def send_telegram_message(text: str, bot_token: str = None, chat_id: str = None) -> tuple[bool, str]:
    """
    Sends a Markdown-formatted text message via Telegram Bot API.
    Supports comma-separated chat IDs.
    Returns: (success: bool, status_message: str)
    """
    if not bot_token or not chat_id:
        cfg = config.get_telegram_config()
        bot_token = bot_token or cfg.get("bot_token")
        chat_id = chat_id or cfg.get("chat_id")

    bot_token = str(bot_token or "").strip()
    chat_ids = _parse_recipients(chat_id)

    if not bot_token or not chat_ids:
        return False, "Telegram Bot Token ili Chat ID nisu postavljeni."

    success_count = 0
    last_err = ""
    for cid in chat_ids:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": cid,
            "text": text,
            "parse_mode": "Markdown"
        }
        try:
            resp = requests.post(url, json=payload, timeout=8)
            data = resp.json()
            if resp.status_code == 200 and data.get("ok"):
                success_count += 1
            else:
                last_err = data.get("description", f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = str(e)

    if success_count > 0:
        return True, f"Poruka poslana na {success_count}/{len(chat_ids)} Telegram primatelja."
    return False, f"Greška Telegram API-ja: {last_err}"

def send_telegram_photo(image_input, caption: str = "", bot_token: str = None, chat_id: str = None) -> tuple[bool, str]:
    """
    Sends a photo with optional Markdown caption via Telegram Bot API.
    Supports multiple/comma-separated chat IDs.
    """
    if not bot_token or not chat_id:
        cfg = config.get_telegram_config()
        bot_token = bot_token or cfg.get("bot_token")
        chat_id = chat_id or cfg.get("chat_id")

    bot_token = str(bot_token or "").strip()
    chat_ids = _parse_recipients(chat_id)

    if not bot_token or not chat_ids:
        return False, "Telegram Bot Token ili Chat ID nisu postavljeni."

    photo_bytes = _encode_image_bytes(image_input)
    if not photo_bytes:
        return send_telegram_message(caption, bot_token=bot_token, chat_id=chat_id)

    success_count = 0
    last_err = ""
    for cid in chat_ids:
        url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
        data = {
            "chat_id": cid,
            "caption": caption,
            "parse_mode": "Markdown"
        }
        files = {
            "photo": ("alert_frame.jpg", photo_bytes, "image/jpeg")
        }
        try:
            resp = requests.post(url, data=data, files=files, timeout=12)
            res_data = resp.json()
            if resp.status_code == 200 and res_data.get("ok"):
                success_count += 1
            else:
                last_err = res_data.get("description", f"HTTP {resp.status_code}")
        except Exception as e:
            last_err = str(e)

    if success_count > 0:
        return True, f"Fotografija poslana na {success_count}/{len(chat_ids)} Telegram primatelja."
    return False, f"Greška pri slanju fotografije: {last_err}"

def test_telegram_connection(bot_token: str = None, chat_id: str = None) -> tuple[bool, str]:
    """
    Sends a test verification message to verify Bot Token and Chat ID.
    """
    msg = (
        "🛡️ *Argusface Biometric Security*\n"
        "━━━━━━━━━━━━━━━━━━━━━━\n"
        "✅ **Uspješna veza s Telegramom!**\n"
        "Sustav je konfiguriran i spreman za slanje trenutnih sigurnosnih obavijesti "
        "za VIP osobe, Crnu listu i pokušaje lažiranja (Anti-Spoof).\n\n"
        f"⏱️ Vrijeme testa: `{time.strftime('%d.%m.%Y. %H:%M:%S')}`"
    )
    return send_telegram_message(msg, bot_token=bot_token, chat_id=chat_id)

# =========================================================================
# E-MAIL NOTIFICATIONS (SMTP)
# =========================================================================

def send_email_alert(
    subject: str,
    html_body: str,
    image_input=None,
    cfg_override: dict = None
) -> tuple[bool, str]:
    """
    Sends a structured HTML email alert with optional embedded JPEG picture using smtplib.
    Returns: (success: bool, status_message: str)
    """
    cfg = cfg_override or config.get_email_config()
    smtp_host = cfg.get("smtp_host", "smtp.gmail.com").strip()
    smtp_port = int(cfg.get("smtp_port", 587))
    use_tls = bool(cfg.get("use_tls", True))
    use_ssl = bool(cfg.get("use_ssl", False))
    sender = cfg.get("sender", "").strip()
    password = cfg.get("password", "").strip()
    recipients = _parse_recipients(cfg.get("recipients", ""))

    if not smtp_host:
        return False, "SMTP poslužitelj (Host) nije konfiguriran."
    if not sender:
        return False, "Pošiljatelj (Sender email) nije postavljen."
    if not recipients:
        return False, "Niti jedan primatelj (Recipients) nije definiran."

    # Build MIME message
    msg = MIMEMultipart("related")
    msg["Subject"] = subject
    msg["From"] = f"Argusface Security <{sender}>"
    msg["To"] = ", ".join(recipients)
    msg["Date"] = smtplib.email.utils.formatdate(localtime=True)

    msg_alt = MIMEMultipart("alternative")
    msg.attach(msg_alt)

    # Attach HTML
    msg_alt.attach(MIMEText(html_body, "html", "utf-8"))

    # Attach photo inline if present
    photo_bytes = _encode_image_bytes(image_input)
    if photo_bytes:
        img_part = MIMEImage(photo_bytes, _subtype="jpeg")
        img_part.add_header("Content-ID", "<alert_photo>")
        img_part.add_header("Content-Disposition", "inline", filename="argusface_incident.jpg")
        msg.attach(img_part)

    try:
        if use_ssl or smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=12)
            if use_tls:
                server.starttls()

        if password:
            server.login(sender, password)

        server.sendmail(sender, recipients, msg.as_string())
        server.quit()
        return True, f"Sigurnosni E-mail uspješno poslan na: {', '.join(recipients)}"
    except Exception as e:
        return False, f"Greška SMTP poslužitelja: {e}"

def test_email_connection(
    smtp_host: str,
    smtp_port: int,
    use_tls: bool,
    use_ssl: bool,
    sender: str,
    password: str,
    recipients: str
) -> tuple[bool, str]:
    """
    Sends a test verification email to ensure SMTP credentials and host work properly.
    """
    subject = "🛡️ Argusface Security: Testna provjera E-mail obavijesti"
    now_str = time.strftime("%d.%m.%Y. u %H:%M:%S")
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b1120; color: #f8fafc; margin: 0; padding: 24px; }}
            .container {{ max-width: 600px; margin: 0 auto; background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
            .header {{ display: flex; align-items: center; border-bottom: 2px solid #38bdf8; padding-bottom: 16px; margin-bottom: 20px; }}
            .title {{ font-size: 20px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px; }}
            .badge-test {{ background: #0284c7; color: #fff; font-size: 12px; font-weight: 600; padding: 4px 10px; border-radius: 20px; text-transform: uppercase; }}
            .content {{ line-height: 1.6; font-size: 14px; color: #cbd5e1; }}
            .card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin: 16px 0; border-left: 4px solid #10b981; }}
            .footer {{ font-size: 12px; color: #64748b; margin-top: 24px; text-align: center; border-top: 1px solid #1e293b; padding-top: 16px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <span class="title">🛡️ Argusface Biometric Studio</span>
            </div>
            <div class="content">
                <p><span class="badge-test">✅ Uspješna veza</span></p>
                <h3>Konfiguracija E-mail obavijesti je ispravna!</h3>
                <p>Ovaj testni e-mail potvrđuje da je Argusface uspješno povezan s Vašim SMTP poslužiteljem te je spreman za slanje trenutnih sigurnosnih upozorenja u slučaju incidenta.</p>
                <div class="card">
                    <strong>Informacije o testu:</strong><br>
                    • Vrijeme: <code>{now_str}</code><br>
                    • SMTP Poslužitelj: <code>{smtp_host}:{smtp_port}</code><br>
                    • Pošiljatelj: <code>{sender}</code>
                </div>
            </div>
            <div class="footer">
                Automatska poruka generirana iz Argusface Biometric Studio sustava.
            </div>
        </div>
    </body>
    </html>
    """
    override = {
        "smtp_host": smtp_host,
        "smtp_port": smtp_port,
        "use_tls": use_tls,
        "use_ssl": use_ssl,
        "sender": sender,
        "password": password,
        "recipients": recipients
    }
    return send_email_alert(subject, html_body, cfg_override=override)

# =========================================================================
# CENTRAL BACKGROUND DISPATCHER (TELEGRAM + EMAIL)
# =========================================================================

def _dispatch_worker(event_type: str, person_name: str, similarity: float, camera_label: str, frame_bgr=None, crop_bgr=None):
    """Background worker that builds and sends alerts via Telegram and E-mail without blocking the camera thread."""
    try:
        now_str = time.strftime("%d.%m.%Y. u %H:%M:%S")
        sim_pct = f"{similarity * 100:.1f}%" if similarity > 0 else "N/A"
        target_img = frame_bgr if frame_bgr is not None else crop_bgr

        tg_cfg = config.get_telegram_config()
        em_cfg = config.get_email_config()

        # Cooldown check (tracks event_type + person_name)
        cooldown_sec = max(60, min(tg_cfg.get("cooldown_min", 5), em_cfg.get("cooldown_min", 10)) * 60)
        cooldown_key = f"{event_type}_{person_name.strip().lower()}"
        now = time.time()

        with _alert_lock:
            last_t = _alert_history.get(cooldown_key, 0.0)
            if now - last_t < cooldown_sec:
                return
            _alert_history[cooldown_key] = now

        # ----------------------------------------------------
        # 1. TELEGRAM DISPATCH
        # ----------------------------------------------------
        if tg_cfg.get("enabled"):
            should_send_tg = False
            if event_type == "blacklist" and tg_cfg.get("notify_blacklist"):
                should_send_tg = True
            elif event_type == "vip" and tg_cfg.get("notify_vip"):
                should_send_tg = True
            elif event_type == "spoof" and tg_cfg.get("notify_spoof"):
                should_send_tg = True
            elif event_type == "unknown" and tg_cfg.get("notify_unknown"):
                should_send_tg = True

            if should_send_tg:
                bot_token = tg_cfg.get("bot_token")
                # Selective routing: Check if dedicated chat_id exists for this event
                if event_type in ("blacklist", "spoof") and tg_cfg.get("chat_id_security"):
                    chat_id_tg = tg_cfg.get("chat_id_security")
                elif event_type == "vip" and tg_cfg.get("chat_id_vip"):
                    chat_id_tg = tg_cfg.get("chat_id_vip")
                else:
                    chat_id_tg = tg_cfg.get("chat_id")

                if bot_token and chat_id_tg:
                    if event_type == "blacklist":
                        caption = (
                            "🚨 *Argusface SIGURNOSNI ALARM — CRNA LISTA*\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"👤 *Nepoželjno lice:* `{person_name}`\n"
                            f"📊 *Pouzdanost:* `{sim_pct}`\n"
                            f"📹 *Izvor / Kamera:* `{camera_label}`\n"
                            f"⏱️ *Vrijeme:* `{now_str}`\n"
                            "⚠️ *Status:* Osoba ima evidentiranu zabranu pristupa!"
                        )
                    elif event_type == "vip":
                        caption = (
                            "⭐ *Argusface OBAVIJEST — VIP DOLAZAK*\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"👤 *VIP Osoba:* `{person_name}`\n"
                            f"📊 *Pouzdanost:* `{sim_pct}`\n"
                            f"📹 *Izvor / Kamera:* `{camera_label}`\n"
                            f"⏱️ *Vrijeme:* `{now_str}`\n"
                            "ℹ️ *Status:* Registriran ulazak VIP osobe u objekt."
                        )
                    elif event_type == "spoof":
                        target_desc = f"`{person_name}`" if person_name and person_name != "Nepoznato" else "Nepoznata osoba"
                        caption = (
                            "🛡️ *Argusface ALARM — POKUŠAJ LAŽIRANJA*\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"⚠️ *Detektirano:* Prezentacijski napad (Ekran mobitela / Slika)\n"
                            f"👤 *Pokušano lice:* {target_desc}\n"
                            f"📹 *Izvor / Kamera:* `{camera_label}`\n"
                            f"⏱️ *Vrijeme:* `{now_str}`\n"
                            "🛑 *Akcija:* Evidencija prolaska i pristup su automatski BLOKIRANI."
                        )
                    else:
                        caption = (
                            "ℹ️ *Argusface Evidencija*\n"
                            "━━━━━━━━━━━━━━━━━━━━━━\n"
                            f"👤 *Osoba:* `{person_name}` ({sim_pct})\n"
                            f"📹 *Kamera:* `{camera_label}`\n"
                            f"⏱️ *Vrijeme:* `{now_str}`"
                        )

                    if target_img is not None:
                        send_telegram_photo(target_img, caption=caption, bot_token=bot_token, chat_id=chat_id_tg)
                    else:
                        send_telegram_message(caption, bot_token=bot_token, chat_id=chat_id_tg)

        # ----------------------------------------------------
        # 2. E-MAIL DISPATCH
        # ----------------------------------------------------
        if em_cfg.get("enabled"):
            should_send_em = False
            if event_type == "blacklist" and em_cfg.get("notify_blacklist"):
                should_send_em = True
            elif event_type == "spoof" and em_cfg.get("notify_spoof"):
                should_send_em = True
            elif event_type == "vip" and em_cfg.get("notify_vip"):
                should_send_em = True

            if should_send_em:
                if event_type == "blacklist":
                    subject = f"🚨 [SIGURNOSNO UPOZORENJE] Detektirana osoba s Crne liste: {person_name}"
                    banner_color = "#ef4444"
                    banner_title = "🚨 DETEKTIRANO LICE S CRNE LISTE (ZABRANA PRISTUPA)"
                    action_txt = "<span style='color: #ef4444; font-weight: bold;'>⚠️ POTREBNA HITNA REAKCIJA SLUŽBE OSIGURANJA. Osoba ima zabranu ulaska!</span>"
                elif event_type == "spoof":
                    subject = "🛡️ [SIGURNOSNI ALARM] Detektiran pokušaj lažiranja lica (Anti-Spoofing)"
                    banner_color = "#f59e0b"
                    banner_title = "🛡️ DETEKTIRAN PREZENTACIJSKI NAPAD (SLIKA / EKRAN)"
                    action_txt = "<span style='color: #f59e0b; font-weight: bold;'>🛑 Pristup je automatski blokiran. Provjerite fizički ulaznu točku.</span>"
                else:
                    subject = f"⭐ [VIP OBAVIJEST] Registriran dolazak: {person_name}"
                    banner_color = "#eab308"
                    banner_title = "⭐ REGISTRIRAN DOLAZAK VIP OSOBE"
                    action_txt = "<span style='color: #eab308; font-weight: bold;'>ℹ️ VIP uzvanik je ušao u objekt. Obavijestite domaćina / recepciju.</span>"

                img_html = ""
                if target_img is not None:
                    img_html = """
                    <div style="margin-top: 16px; text-align: center;">
                        <img src="cid:alert_photo" alt="Fotografija lica" style="max-width: 100%; height: auto; border-radius: 8px; border: 2px solid #334155; box-shadow: 0 4px 12px rgba(0,0,0,0.4);" />
                    </div>
                    """

                html_body = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <style>
                        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0b1120; color: #f8fafc; margin: 0; padding: 24px; }}
                        .container {{ max-width: 640px; margin: 0 auto; background: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 24px; box-shadow: 0 12px 30px rgba(0,0,0,0.6); }}
                        .header {{ border-bottom: 2px solid #38bdf8; padding-bottom: 12px; margin-bottom: 16px; font-size: 18px; font-weight: 700; color: #38bdf8; }}
                        .banner {{ background: {banner_color}; color: #ffffff; padding: 12px 16px; border-radius: 8px; font-weight: 700; font-size: 15px; margin-bottom: 20px; }}
                        .table-card {{ background: #1e293b; border-radius: 8px; padding: 16px; margin: 16px 0; }}
                        .table-card table {{ width: 100%; border-collapse: collapse; }}
                        .table-card td {{ padding: 8px 6px; font-size: 14px; border-bottom: 1px solid #334155; }}
                        .table-card td.lbl {{ color: #94a3b8; width: 35%; font-weight: 600; }}
                        .table-card td.val {{ color: #f8fafc; }}
                        .footer {{ font-size: 11px; color: #64748b; margin-top: 24px; text-align: center; border-top: 1px solid #1e293b; padding-top: 14px; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="header">🛡️ Argusface Biometric Studio — Sigurnosni Sustav</div>
                        <div class="banner">{banner_title}</div>
                        <div class="table-card">
                            <table>
                                <tr>
                                    <td class="lbl">Identificirano lice:</td>
                                    <td class="val"><strong style="font-size: 16px; color: #38bdf8;">{person_name}</strong></td>
                                </tr>
                                <tr>
                                    <td class="lbl">Izvor / Lokacija:</td>
                                    <td class="val"><code>{camera_label}</code></td>
                                </tr>
                                <tr>
                                    <td class="lbl">Vrijeme detekcije:</td>
                                    <td class="val">{now_str}</td>
                                </tr>
                                <tr>
                                    <td class="lbl">Biometrijska sličnost:</td>
                                    <td class="val">{sim_pct}</td>
                                </tr>
                                <tr>
                                    <td class="lbl">Status & Akcija:</td>
                                    <td class="val">{action_txt}</td>
                                </tr>
                            </table>
                        </div>
                        {img_html}
                        <div class="footer">
                            Ova poruka je automatski generirana iz sustava Argusface Biometric Security.<br>
                            Svi biometrijski podaci obrađeni su 100% lokalno u skladu s GDPR i internim sigurnosnim pravilnikom.
                        </div>
                    </div>
                </body>
                </html>
                """
                send_email_alert(subject, html_body, image_input=target_img)

    except Exception as e:
        print(f"[UPOZORENJE] Greška u centralnom notifier dispečeru: {e}")

def trigger_alert_async(event_type: str, person_name: str, similarity: float = 0.0, camera_label: str = "Kamera", frame_bgr=None, crop_bgr=None):
    """
    Non-blocking entry point for firing alerts from live_cam or detection pipelines.
    Spawns a quick background thread so camera frame rate never drops.
    """
    t = threading.Thread(
        target=_dispatch_worker,
        args=(event_type, person_name, similarity, camera_label, frame_bgr, crop_bgr),
        daemon=True
    )
    t.start()
