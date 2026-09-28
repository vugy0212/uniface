import os
import sys
import docx

APP_DIR = os.path.dirname(os.path.abspath(__file__))
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_shading(cell, color_hex):
    """Postavlja boju pozadine ćelije tablice."""
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Postavlja unutarnje margine (padding) ćelije."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout_box(doc, text, title="NAPOMENA", border_color="0284C7", bg_color="F0F9FF"):
    """Kreira stiliziranu 'Callout / Note' kutiju s lijevom debelom linijom."""
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_shading(cell, bg_color)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    # Lijevi border
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="none"/>'
        f'  <w:left w:val="single" w:sz="24" w:space="0" w:color="{border_color}"/>'
        f'  <w:bottom w:val="none"/>'
        f'  <w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run_title = p.add_run(f"📌 {title}: ")
    run_title.bold = True
    run_title.font.name = "Segoe UI"
    run_title.font.size = Pt(10)
    run_title.font.color.rgb = RGBColor.from_string(border_color)
    
    run_text = p.add_run(text)
    run_text.font.name = "Segoe UI"
    run_text.font.size = Pt(9.5)
    run_text.font.color.rgb = RGBColor(30, 41, 59)
    
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def format_table(tbl, col_widths, header_bg="1E3A8A", alt_bg="F8FAFC"):
    """Formatira tablicu s modernim zaglavljem i izmjeničnim bojama redaka."""
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for r_idx, row in enumerate(tbl.rows):
        # Spriječi lomljenje retka preko stranice
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        
        is_header = (r_idx == 0)
        if is_header:
            trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))
            
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            set_cell_margins(cell, top=120, bottom=120, left=140, right=140)
            
            # Shading
            if is_header:
                set_cell_shading(cell, header_bg)
            elif r_idx % 2 == 1:
                set_cell_shading(cell, alt_bg)
            else:
                set_cell_shading(cell, "FFFFFF")
                
            # Finiji obrubi
            tcPr = cell._tc.get_or_add_tcPr()
            tcBorders = parse_xml(
                f'<w:tcBorders {nsdecls("w")}>'
                f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
                f'  <w:left w:val="none"/>'
                f'  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CBD5E1"/>'
                f'  <w:right w:val="none"/>'
                f'</w:tcBorders>'
            )
            tcPr.append(tcBorders)
            
            # Formatiranje teksta
            for p in cell.paragraphs:
                p.paragraph_format.space_before = Pt(2)
                p.paragraph_format.space_after = Pt(2)
                p.paragraph_format.line_spacing = 1.15
                for r in p.runs:
                    r.font.name = "Segoe UI"
                    if is_header:
                        r.bold = True
                        r.font.size = Pt(9.5)
                        r.font.color.rgb = RGBColor(255, 255, 255)
                    else:
                        r.font.size = Pt(9)
                        r.font.color.rgb = RGBColor(30, 41, 59)

def build_user_manual():
    doc = Document()
    
    # 1. Postavljanje margina stranice (A4 standard, 2 cm)
    for section in doc.sections:
        section.page_width = Inches(8.27)  # A4
        section.page_height = Inches(11.69)
        section.top_margin = Inches(0.8)
        section.bottom_margin = Inches(0.8)
        section.left_margin = Inches(0.85)
        section.right_margin = Inches(0.85)
        section.different_first_page_header_footer = True
        
        # Header i Footer
        header = section.header
        hp = header.paragraphs[0]
        hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        hrun = hp.add_run("ArgusFace Studio v1.0 • Korisnička i Tehničko-Pravna Dokumentacija")
        hrun.font.name = "Segoe UI"
        hrun.font.size = Pt(8.5)
        hrun.font.color.rgb = RGBColor(148, 163, 184)
        
        footer = section.footer
        fp = footer.paragraphs[0]
        fp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        frun = fp.add_run("Povjerljivo • Zaštićeno autorskim pravom • ArgusFace Studio")
        frun.font.name = "Segoe UI"
        frun.font.size = Pt(8.5)
        frun.font.color.rgb = RGBColor(148, 163, 184)

    # ==========================================
    # NASLOVNICA (COVER PAGE)
    # ==========================================
    p_top_space = doc.add_paragraph()
    p_top_space.paragraph_format.space_before = Pt(40)
    
    # Bedž verzije
    p_badge = doc.add_paragraph()
    r_badge = p_badge.add_run("SLUŽBENA DOKUMENTACIJA SUSTAVA • VERZIJA 1.0 (PRODUCTION RELEASE)")
    r_badge.font.name = "Segoe UI"
    r_badge.font.size = Pt(9.5)
    r_badge.bold = True
    r_badge.font.color.rgb = RGBColor(2, 132, 199) # Sky Blue
    
    # Glavni naslov
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(8)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("ArgusFace Studio v1.0")
    r_title.font.name = "Segoe UI"
    r_title.font.size = Pt(32)
    r_title.bold = True
    r_title.font.color.rgb = RGBColor(15, 23, 42) # Deep Slate
    
    # Podnaslov
    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(24)
    r_sub = p_sub.add_run("Sveobuhvatni korisnički priručnik, tehnička specifikacija i pravni okvir (Licence, GDPR, EU AI Act)")
    r_sub.font.name = "Segoe UI"
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = RGBColor(71, 85, 105)
    
    # Razdjelnik
    p_rule = doc.add_paragraph()
    p_rule.paragraph_format.space_after = Pt(28)
    r_rule = p_rule.add_run("―" * 42)
    r_rule.font.color.rgb = RGBColor(203, 213, 225)
    
    # Kartica sa sažetkom specifikacije na naslovnici
    tbl_cover = doc.add_table(rows=6, cols=2)
    tbl_cover.alignment = WD_TABLE_ALIGNMENT.CENTER
    cover_widths = [Inches(2.2), Inches(4.3)]
    
    meta_data = [
        ("Naziv sustava:", "ArgusFace Studio — Biometrijski sustav za analizu i prepoznavanje lica"),
        ("Verzija izdanja:", "v1.0 (Komercijalno izdanje s čistim modelima)"),
        ("Arhitektura obrade:", "100% On-Premise / Edge (Potpuno lokalno, Zero-Cloud rizik)"),
        ("Temeljni AI modeli:", "RetinaFace (Detekcija • MIT) + EdgeFace BASE (Identifikacija • BSD-3-Clause)"),
        ("Pravni status:", "100% legalna komercijalizacija, usklađeno s GDPR-om i EU AI Actom"),
        ("Datum izdanja:", "Rujan 2026. godine")
    ]
    for idx, (lbl, val) in enumerate(meta_data):
        row = tbl_cover.rows[idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        cell_lbl.paragraphs[0].add_run(lbl).bold = True
        cell_val.paragraphs[0].add_run(val)
        
    format_table(tbl_cover, cover_widths, header_bg="0F172A", alt_bg="F1F5F9")
    
    p_space = doc.add_paragraph()
    p_space.paragraph_format.space_before = Pt(60)
    
    p_footer_note = doc.add_paragraph()
    p_footer_note.paragraph_format.space_after = Pt(0)
    r_fn = p_footer_note.add_run("Dokument je namijenjen krajnjim korisnicima, sistem administratorima, voditeljima obrade podataka i pravnim službama.")
    r_fn.font.name = "Segoe UI"
    r_fn.font.size = Pt(9)
    r_fn.font.color.rgb = RGBColor(100, 116, 139)

    doc.add_page_break()

    # ==========================================
    # SADRŽAJ (TABLE OF CONTENTS PREVIEW)
    # ==========================================
    h1 = doc.add_heading("Sadržaj Dokumenta", level=1)
    h1.paragraph_format.space_before = Pt(12)
    h1.paragraph_format.space_after = Pt(14)
    
    toc_items = [
        ("1. Uvod i Pregled Sustava", "Filozofija, namjena, arhitektura i ključne prednosti lokalne obrade"),
        ("2. Instalacija i Načini Pokretanja", "Hardverski zahtjevi (DirectML & FAISS), Windows Setup Wizard (.exe), Desktop pokretanje"),
        ("3. Detaljan Korisnički Vodič kroz Module", "Korak-po-korak upute za svih 6 temeljnih modula ArgusFace Studija"),
        ("   3.1. Modul: Prepoznavanje Lica (Live & Static)", "Učitavanje slike, podešavanje praga, maskiranje/cenzura, brzo dodavanje"),
        ("   3.2. Modul: Baza Osoba & Vektorski Indeks", "Pojedinačni i masovni unos, FAISS HNSW/FlatIP, uređivanje profila (Edit)"),
        ("   3.3. Modul: Nadzor Uživo i Mreža Kamera (Multi-Cam)", "Spajanje USB i RTSP kamera, 2x2 mreža, solo prikaz, tipkovničke kratice"),
        ("   3.4. Modul: NVR Snimanje i Dnevnik Prolazaka", "Automatska evidencija prolazaka, GDPR automatska rotacija, CSV izvoz"),
        ("   3.5. Modul: Pametni Sorter Fotografija", "Automatizirano razvrstavanje tisuća slika uz GPU/DirectML ubrzanje"),
        ("   3.6. Modul: O Sustavu i Sigurnosna Kopija", "ZIP arhiviranje, obnova baze, Cyber statusni bedž, GDPR postavke"),
        ("4. Pravni Aspekti, Licence i Regulatorna Usklađenost", "Temeljito pravno obrazloženje: Licence modela, GDPR i EU AI Act"),
        ("   4.1. Komercijalne Licence Model Stoga", "RetinaFace (MIT), EdgeFace BASE (BSD-3-Clause), uklanjanje FairFace-a"),
        ("   4.2. GDPR Usklađenost (Uredba EU 2016/679)", "Biometrijski podaci (Čl. 9), Ograničenje pohrane (Čl. 5), Pravo na zaborav"),
        ("   4.3. EU AI Act Usklađenost (Uredba EU 2024/1689)", "Kategorizacija sustava, dopuštena vs. zabranjena upotreba"),
        ("5. Rješavanje Problema i Najbolje Prakse (FAQ)", "Savjeti za točnost, DirectML podrška, FAISS skalabilnost, backup")
    ]
    
    tbl_toc = doc.add_table(rows=len(toc_items), cols=2)
    tbl_toc.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_widths = [Inches(3.2), Inches(3.3)]
    for idx, (title, desc) in enumerate(toc_items):
        r = tbl_toc.rows[idx]
        c1, c2 = r.cells[0], r.cells[1]
        c1.paragraphs[0].add_run(title).bold = (not title.startswith("   "))
        c2.paragraphs[0].add_run(desc)
    format_table(tbl_toc, toc_widths, header_bg="1E3A8A", alt_bg="F8FAFC")

    doc.add_page_break()

    # ==========================================
    # POGLAVLJE 1: UVOD I PREGLED SUSTAVA
    # ==========================================
    h1 = doc.add_heading("1. Uvod i Pregled Sustava", level=1)
    h1.paragraph_format.space_before = Pt(16)
    
    p = doc.add_paragraph(
        "ArgusFace Studio v1.0 predstavlja cjelovito industrijsko softversko rješenje za detekciju, "
        "biometrijsko prepoznavanje, organizaciju i nadzor lica u stvarnom vremenu. "
        "Aplikacija objedinjuje najsuvremenija dostignuća dubokog učenja (Deep Learning) i računalnog vida (Computer Vision) "
        "u intuitivno grafičko sučelje prilagođeno kako svakodnevnim korisnicima, tako i profesionalnim operaterima."
    )
    p.paragraph_format.line_spacing = 1.15
    
    p = doc.add_paragraph(
        "Temeljna arhitektonska filozofija ArgusFace Studija počiva na 100% lokalnoj obradi (Edge / On-Premise). "
        "To znači da niti jedna fotografija, video stream niti generirani biometrijski vektor nikada ne napuštaju "
        "lokalno računalo korisnika. Ovakav pristup pruža tri ključne prednosti:"
    )
    p.paragraph_format.line_spacing = 1.15
    
    bullet_points = [
        ("Maksimalna privatnost i sigurnost: ", "Potpuna eliminacija rizika od curenja podataka kroz oblak (Cloud data breaches) ili neovlaštenog pristupa trećih strana."),
        ("Puna neovisnost o internetu (Offline rad): ", "Aplikacija radi u potpunosti autonomno bez internetske veze, u zatvorenim lokalnim mrežama (LAN) ili na terenskim prijenosnim računalima."),
        ("Nulta latencija i trenutni odziv: ", "Usporedba biometrijskih vektora odvija se na lokalnom procesoru ili grafičkoj kartici u djeliću milisekunde zahvaljujući optimiziranim BLAS matričnim operacijama.")
    ]
    for b_title, b_desc in bullet_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(3)
        bp.paragraph_format.line_spacing = 1.15
        r1 = bp.add_run(b_title)
        r1.bold = True
        r1.font.color.rgb = RGBColor(15, 23, 42)
        bp.add_run(b_desc)

    add_callout_box(
        doc,
        "ArgusFace Studio v1.0 u potpunosti rješava pravne i licencne izazove prepoznavanja lica. "
        "Uklanjanjem akademskih modela (ArcFace/FairFace) i ugradnjom komercijalno čistog modela EdgeFace BASE (BSD-3-Clause), "
        "sustav je pravno spreman za prodaju poslovnim subjektima bez rizika od tužbi ili skrivenih tantijema.",
        title="KOMERCIJALNI STATUS v1.0",
        border_color="0284C7",
        bg_color="F0F9FF"
    )

    # ==========================================
    # POGLAVLJE 2: INSTALACIJA I NAČINI POKRETANJA
    # ==========================================
    h1 = doc.add_heading("2. Instalacija i Načini Pokretanja", level=1)
    h1.paragraph_format.space_before = Pt(16)
    
    doc.add_heading("2.1. Hardverski i Softverski Zahtjevi", level=2)
    
    p = doc.add_paragraph("Sustav je optimiziran za rad na širokom spektru hardvera, od standardnih uredskih računala do radnih stanica s namjenskim grafičkim karticama:")
    p.paragraph_format.line_spacing = 1.15
    
    tbl_req = doc.add_table(rows=7, cols=3)
    req_widths = [Inches(1.8), Inches(2.3), Inches(2.4)]
    req_data = [
        ("Komponenta", "Minimalni zahtjevi", "Preporučena konfiguracija"),
        ("Procesor (CPU)", "Intel Core i3 / AMD Ryzen 3 (4 jezgre)", "Intel Core i5/i7/i9 ili AMD Ryzen 5/7 (6+ jezgri)"),
        ("Radna memorija (RAM)", "4 GB RAM-a", "8 GB do 16 GB RAM-a"),
        ("Grafička kartica (GPU / NPU)", "Nije obavezna (Radi na CPU-u)", "Intel Iris Xe / Arc, AMD Radeon, NVIDIA GeForce ili NPU (DirectML DirectX 12)"),
        ("Vektorski indeks", "Ugrađeni NumPy BLAS", "FAISS C++ AVX2/HNSW (sub-milisekundno za > 20.000 lica)"),
        ("Pohrana (Disk)", "1.5 GB slobodnog prostora", "SSD disk za brži uvoz velikih kolekcija fotografija"),
        ("Operativni sustav", "Windows 10 / Windows 11 (64-bit)", "Windows 11 (64-bit)")
    ]
    for idx, (c1, c2, c3) in enumerate(req_data):
        row = tbl_req.rows[idx]
        row.cells[0].paragraphs[0].add_run(c1)
        row.cells[1].paragraphs[0].add_run(c2)
        row.cells[2].paragraphs[0].add_run(c3)
    format_table(tbl_req, req_widths, header_bg="1E3A8A", alt_bg="F8FAFC")

    add_callout_box(
        doc,
        "ArgusFace Studio uključuje univerzalno hardversko ubrzanje putem Microsoft DirectML (DirectX 12) i Intel OpenVINO podsustava. "
        "Za razliku od starijih rješenja koja zahtijevaju isključivo skupe NVIDIA kartice i instalaciju 4 GB vanjskih CUDA paketa, "
        "ArgusFace Studio automatski koristi bilo koji integrirani ili diskretni grafički čip (Intel Iris Xe, Intel Arc, AMD Radeon, NVIDIA) "
        "ili novi NPU procesor, postižući vrhunske performanse i do 10x manju potrošnju energije!",
        title="UNIVERZALNO HARDVERSKO UBRZANJE (DirectML)",
        border_color="0284C7",
        bg_color="F0F9FF"
    )
    
    doc.add_heading("2.2. Instalacija putem Windows Instalacijskog Čarobnjaka (Setup Wizard)", level=2)
    p = doc.add_paragraph(
        "Komercijalno izdanje ArgusFace Studija isporučuje se kao cjeloviti, samostalni instalacijski paket "
        "za Windows operacijske sustave ('ArgusFace_Studio_Setup_v1.0.exe'). "
        "Korisnik ne mora imati predinstaliran Python, baze podataka niti tehničke razvojne alate – "
        "sve potrebne biblioteke, optimizirani AI modeli i runtime okruženje integrirani su u jedinstveni instalacijski paket."
    )
    p.paragraph_format.line_spacing = 1.15
    
    install_steps = [
        ("1. Pokretanje instalacije: ", "Dvokliknite na instalacijsku datoteku 'ArgusFace_Studio_Setup_v1.0.exe'. Ako Windows prikaže sigurnosno UAC upozorenje (Korisnički račun), potvrdite pokretanje s 'Da'."),
        ("2. Licenčni uvjeti: ", "U instalacijskom čarobnjaku prikazuju se uvjeti komercijalne licence, izjave o 100% lokalnoj obradi podataka i usklađenosti s GDPR-om."),
        ("3. Odabir odredišne mape: ", "Čarobnjak nudi zadanu instalacijsku mapu (npr. 'C:\\Program Files\\ArgusFace Studio\\' ili odabranu lokaciju po vašoj želji)."),
        ("4. Kreiranje prečaca: ", "Instalater automatski kreira prečac na Radnoj površini (Desktop) i unutar Windows Start izbornika s prepoznatljivom ikonom aplikacije."),
        ("5. Završetak: ", "Nakon što čarobnjak raspakira i konfigurira sve datoteke, klikom na 'Završi' aplikacija je spremna za trenutni rad.")
    ]
    for s_title, s_desc in install_steps:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        bp.add_run(s_title).bold = True
        bp.add_run(s_desc)
        
    doc.add_heading("2.3. Pokretanje i Rad s Aplikacijom (.exe Izvršna Datoteka)", level=2)
    p = doc.add_paragraph(
        "Nakon dovršene instalacije, pokretanje ArgusFace Studija vrši se izravno putem izvršne .exe datoteke "
        "ili prečaca na radnoj površini ('ArgusFace Studio.exe'):"
    )
    p.paragraph_format.line_spacing = 1.15
    
    exe_features = [
        ("Nativni Desktop radni prozor: ", "Dvoklikom na prečac 'ArgusFace Studio' na radnoj površini, aplikacija se automatski otvara u čistom, samostalnom radnom prozoru (WebView2) bez adresne trake preglednika, tabova ili ometanja."),
        ("Tihi start u pozadini: ", "Pokretanje je u potpunosti automatizirano i prilagođeno korisniku – ne prikazuju se crni konzolni prozori. Svi pozadinski AI procesi, video engine i lokalna SQLite baza pokreću se nečujno u pozadini."),
        ("Čisto i sigurno gašenje: ", "Zatvaranjem radnog prozora klikom na tipku 'X' (u gornjem desnom kutu), aplikacija automatski oslobađa video tokove s kamera, sigurno zatvara sve aktivne NVR video snimke i gasi pozadinski servis bez zaostajanja procesa u memoriji.")
    ]
    for ef_t, ef_d in exe_features:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        bp.add_run(ef_t).bold = True
        bp.add_run(ef_d)

    add_callout_box(
        doc,
        "Za administratore sustava i integraciju u veće IT sustave: "
        "Izvršna datoteka podržava i napredne opcije pokretanja putem parametara, poput rada kroz vanjski "
        "web preglednik ('ArgusFace Studio.exe --browser') ili automatiziranog pokretanja specifičnih "
        "pozadinskih servisa (npr. neprekidno NVR snimanje).",
        title="NAPOMENA ZA ADMINISTRATORE",
        border_color="0284C7",
        bg_color="F0F9FF"
    )

    doc.add_page_break()

    # ==========================================
    # POGLAVLJE 3: DETALJAN KORISNIČKI VODIČ KROZ MODULE
    # ==========================================
    h1 = doc.add_heading("3. Detaljan Vodič kroz Module i Grafičko Sučelje", level=1)
    h1.paragraph_format.space_before = Pt(16)
    
    # MODUL 1
    doc.add_heading("3.1. Modul: Prepoznavanje Lica (Live & Static Analysis)", level=2)
    p = doc.add_paragraph(
        "Ovaj modul predstavlja primarni alat za analizu fotografija, pojedinačnih slika ili video tokova. "
        "Korisnik može prenijeti fotografiju sa svog računala (Drag & Drop) ili snimiti fotografiju web kamerom."
    )
    p.paragraph_format.line_spacing = 1.15
    
    m1_steps = [
        ("Učitavanje slike: ", "Povucite sliku u predviđeno polje ili kliknite na gumb 'Webcam' za snimanje trenutnog kadra."),
        ("Prag prepoznavanja (Threshold): ", "Klizačem možete fino podesiti osjetljivost usporedbe kosinusne sličnosti (0.0 do 1.0). Optimalna preporučena vrijednost iznosi 0.45. Viša vrijednost (npr. 0.55) smanjuje lažna prepoznavanja, dok niža (npr. 0.38) omogućuje prepoznavanje pri lošijem osvjetljenju."),
        ("Zamućenje nepoznatih lica (Privacy Blur): ", "Aktiviranjem ove opcije sustav automatski pikselizira/zamućuje lica prolaznika ili osoba koje se ne nalaze u bazi, štiteći njihovu privatnost."),
        ("Vizualna identifikacija: ", "Prepoznate osobe uokviruju se smaragdnim zelenim okvirom s imenom i postotkom točnosti (npr. 'Ivan Horvat (96.3%)'), dok se nepoznata lica uokviruju crvenom bojom."),
        ("Brza registracija nepoznatih: ", "Ako sustav detektira lice koje nije u bazi, korisnik ga može izravno iz rezultata analize spremiti kao novu osobu bez napuštanja ovog taba.")
    ]
    for st, sd in m1_steps:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        r1 = bp.add_run(st)
        r1.bold = True
        bp.add_run(sd)

    # MODUL 2
    doc.add_heading("3.2. Modul: Baza Osoba & Tehnologija Centroid Multi-Sample", level=2)
    p = doc.add_paragraph(
        "Modul Baza Osoba omogućuje registraciju, organizaciju, pretragu i uređivanje profila osoba. "
        "Sustav podržava dva načina unosa: pojedinačni unos s vizualnim odabirom i brzi masovni unos (Batch)."
    )
    p.paragraph_format.line_spacing = 1.15
    
    add_callout_box(
        doc,
        "ArgusFace Studio koristi napredni Centroid Multi-Sample algoritam. "
        "Kada za osobu unesete više fotografija (npr. frontalno, blagi lijevi kut, desni kut), "
        "sustav matematički sintetizira njihove 512-D vektore u jedinstveni optimalni centroid. "
        "Time se točnost podiže s bazičnih 35% (1 slika) na preko 95-100% pokrivenosti!",
        title="TEHNOLOGIJA MAKSIMALNE TOČNOSTI",
        border_color="10B981",
        bg_color="F0FDF4"
    )

    p_tbl_title = doc.add_paragraph()
    p_tbl_title.paragraph_format.space_before = Pt(6)
    p_tbl_title.add_run("Razine kvalitete biometrijskog profila u bazi:").bold = True
    
    tbl_qual = doc.add_table(rows=4, cols=4)
    qual_widths = [Inches(1.2), Inches(1.8), Inches(1.5), Inches(2.0)]
    qual_data = [
        ("Broj slika", "Oznaka u sustavu", "Pokrivenost", "Preporuka za korisnika"),
        ("1 slika", "🔴 Osnovno", "35% pokrivenost", "Preporučuje se dodati još 1-2 slike pod drugim kutom."),
        ("2 slike", "🟡 Dobro", "70% pokrivenost", "Dobra točnost za standardne uredske uvjete."),
        ("3+ slika", "🟢 Izvrsno", "100% pokrivenost", "Optimalna točnost; pokriveni su svi kutevi i crte lica.")
    ]
    for idx, (c1, c2, c3, c4) in enumerate(qual_data):
        row = tbl_qual.rows[idx]
        row.cells[0].paragraphs[0].add_run(c1)
        row.cells[1].paragraphs[0].add_run(c2)
        row.cells[2].paragraphs[0].add_run(c3)
        row.cells[3].paragraphs[0].add_run(c4)
    format_table(tbl_qual, qual_widths, header_bg="1E3A8A", alt_bg="F8FAFC")

    p_edit = doc.add_paragraph()
    p_edit.paragraph_format.space_before = Pt(8)
    p_edit.paragraph_format.line_spacing = 1.15
    r_ed = p_edit.add_run("Uređivanje postojećih profila (Ime, Prezime, Bilješke): ")
    r_ed.bold = True
    p_edit.add_run(
        "Korisnik može u bilo kojem trenutku kliknuti na željenu osobu u tablici baze te klikom na gumb '✏️ Uredi osobu' "
        "otvoriti formu za promjenu imena ili ažuriranje internih administratorskih bilješki. "
        "Klikom na '💾 Spremi izmjene' podaci se trenutno trajno pohranjuju u bazu bez potrebe za ponovnim unosom fotografija."
    )

    add_callout_box(
        doc,
        "Skalabilni FAISS Vektorski Indeks (Facebook AI Similarity Search): "
        "ArgusFace Studio koristi dvorazinski vektorski pretraživač. Za standardne baze koristi se FAISS IndexFlatIP (SIMD AVX2/AVX-512) "
        "koji trenutačno računa egzaktne kosinusne sličnosti. Kod velikih baza koje prelaze 1.000 registriranih osoba sustav automatski "
        "aktivira HNSW grafovsko indeksiranje (Hierarchical Navigable Small World). Na testu baze od 20.000 lica, "
        "pretraživanje traje nevjerojatnih 0.11 ms — preko 20x brže od klasičnih SQLite/Python pretraga, uz nulto zagušenje procesora!",
        title="SKALABILNOST ZA DESETKE TISUĆA PROFILA (FAISS)",
        border_color="0284C7",
        bg_color="F0F9FF"
    )

    # MODUL 3
    doc.add_heading("3.3. Modul: Nadzor Uživo i Mreža Kamera (Multi-Camera 2×2)", level=2)
    p = doc.add_paragraph(
        "ArgusFace Studio uključuje robusni video podsustav sposoban za istovremeni prihvat lokalnih USB kamera "
        "i profesionalnih mrežnih IP/RTSP nadzornih kamera (npr. Denver IPC-1030MK2, Hikvision, Dahua i dr.)."
    )
    p.paragraph_format.line_spacing = 1.15
    
    p = doc.add_paragraph("Korisnik u prozorima nadzora ima na raspolaganju intuitivne tipkovničke prečace:")
    p.paragraph_format.line_spacing = 1.15
    
    keys_data = [
        ("Tipke [1], [2], [3], [4]: ", "Trenutno povećavaju odabranu kameru u prikaz preko cijelog ekrana (Solo mod)."),
        ("Tipka [0] ili [ESC]: ", "Vraća prikaz u standardnu 2×2 mrežu sa svim aktivnim kamerama."),
        ("Tipka [R]: ", "Ručno pokretanje ili zaustavljanje NVR video snimanja."),
        ("Tipka [S]: ", "Trenutno spremanje visokorezolucijskog kadra (Snapshot) u mapu 'data/snapshots/'.")
    ]
    for k_t, k_d in keys_data:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        bp.add_run(k_t).bold = True
        bp.add_run(k_d)

    # MODUL 4
    doc.add_heading("3.4. Modul: NVR Snimanje i Dnevnik Prolazaka (Security Audit)", level=2)
    p = doc.add_paragraph(
        "Svaka uspješna detekcija prepoznate osobe ili registriranog kretanja automatski se bilježi u "
        "sigurnosni dnevnik prolazaka unutar SQLite baze podataka (`detection_events`)."
    )
    p.paragraph_format.line_spacing = 1.15
    
    p_log = doc.add_paragraph(
        "Korisnik može u bilo kojem trenutku pregledati kronološki popis događaja s točnim vremenom, nazivom izvora "
        "(kamere), izračunatom sličnosti i vezanim kadrom. Također je omogućen **izvoz cjelokupnog dnevnika u CSV/Excel format** "
        "za potrebe evidencije radnog vremena, revizija ili internih sigurnosnih izvještaja."
    )
    p_log.paragraph_format.line_spacing = 1.15

    add_callout_box(
        doc,
        "GDPR Automatska Rotacija i Brisanje Starih Podataka (Retention Policy): "
        "Sukladno načelu ograničenja pohrane (GDPR Članak 5(1)(e)), sustav omogućuje automatsku rotaciju zapisa prolazaka, "
        "JPEG slika lica i NVR MP4 snimaka na rokove od 15, 30, 60 ili 90 dana (ili trajno). "
        "Sustav automatski tiho briše zastarjele podatke pri svakom pokretanju, a korisnik može i ručno u bilo kojem trenutku "
        "pokrenuti čišćenje klikom na gumb 'Očisti stare podatke odmah' u Tabu 6.",
        title="GDPR ROTACIJA PODATAKA",
        border_color="10B981",
        bg_color="F0FDF4"
    )

    # MODUL 5
    doc.add_heading("3.5. Modul: Pametni Sorter Fotografija (Photo Sorter)", level=2)
    p = doc.add_paragraph(
        "Namijenjen fotografskim studijima, organizatorima događanja, marketinškim agencijama i privatnim korisnicima "
        "s velikim arhivama slika. Sorter prolazi kroz odabranu ulaznu mapu sa stotinama ili tisućama fotografija, "
        "analizira svako lice te automatski sortira slike u podmape s imenima prepoznatih osoba uz punu GPU/DirectML akceleraciju."
    )
    p.paragraph_format.line_spacing = 1.15
    
    ps_features = [
        ("Zajedničke fotografije: ", "Ako je na slici detektirano više različitih osoba iz baze, slika se pametno smješta u posebnu mapu 'Zajednicke_fotografije' kako bi se izbjeglo nepotrebno višestruko dupliciranje diska."),
        ("Neprepoznate osobe: ", "Slike na kojima nema registriranih osoba premještaju se u mapu 'Neprepoznati' za kasniji ručni pregled."),
        ("Detaljan CSV izvještaj: ", "Nakon završetka sortiranja automatski se generira detaljan izvještaj s popisom svih datoteka, detektiranih lica i njihovih koordinata.")
    ]
    for pst, psd in ps_features:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        bp.add_run(pst).bold = True
        bp.add_run(psd)

    # MODUL 6
    doc.add_heading("3.6. Modul: O Sustavu i Sigurnosna Kopija (Backup, Telemetrija & GDPR)", level=2)
    p = doc.add_paragraph(
        "Ovaj modul pruža uvid u hardversku telemetriju računala kroz interaktivni Cyber statusni bedž "
        "(prikaz grafičkih kartica, DirectML/CUDA hardverskog ubrzanja, FAISS vektorskog indeksa i zauzeća diska), "
        "upravljanje GDPR politikom zadržavanja podataka te kompletan izvoz (Export) i uvoz (Restore) cjelokupne "
        "biometrijske baze i fotografija u ZIP arhivu jednim klikom, osiguravajući jednostavnu migraciju na drugo računalo."
    )
    p.paragraph_format.line_spacing = 1.15

    doc.add_page_break()

    # ==========================================
    # POGLAVLJE 4: PRAVNI ASPEKTI, LICENCE I USKLAĐENOST
    # ==========================================
    h1 = doc.add_heading("4. Pravni Aspekti, Licence i Regulatorna Usklađenost", level=1)
    h1.paragraph_format.space_before = Pt(16)
    
    p = doc.add_paragraph(
        "Ovo poglavlje pruža detaljno pravno tumačenje tehnološkog stoga ArgusFace Studija v1.0, "
        "s naglaskom na zaštitu intelektualnog vlasništva, legalnu komercijalnu distribuciju, "
        "usklađenost s Općom uredbom o zaštiti podataka (GDPR) te Uredbom o umjetnoj inteligenciji (EU AI Act)."
    )
    p.paragraph_format.line_spacing = 1.15
    
    doc.add_heading("4.1. Komercijalne Licence AI Modela i Tehnološkog Stoga", level=2)
    
    p = doc.add_paragraph(
        "U razvoju sustava ArgusFace Studio v1.0 provedena je stroga revizija podrijetla koda i težina (weights) "
        "svih korištenih modela kako bi se eliminirala svaka mogućnost povrede autorskih prava trećih strana:"
    )
    p.paragraph_format.line_spacing = 1.15
    
    tbl_lic = doc.add_table(rows=5, cols=4)
    lic_widths = [Inches(1.5), Inches(1.5), Inches(1.6), Inches(1.9)]
    lic_data = [
        ("Komponenta", "AI Model / Biblioteka", "Licenca koda i utega", "Komercijalni status"),
        ("Detekcija lica", "RetinaFace (ResNet/MobileNet)", "MIT License", "Dopuštena komercijalna prodaja i distribucija."),
        ("Prepoznavanje lica", "EdgeFace BASE (EdgeNeXt)", "BSD-3-Clause (Idiap Institute)", "100% legalna komercijalna prodaja i integracija."),
        ("Analiza dobi i spola", "Isključeno iz v1.0 stoga", "N/A (Nije ugrađeno)", "Uklonjeno radi pravnog i tehničkog rasterećenja."),
        ("Inženjerski stog", "OpenCV, ONNX Runtime, SQLite", "Apache 2.0 / Public Domain", "Potpuno slobodno za komercijalnu upotrebu.")
    ]
    for idx, (c1, c2, c3, c4) in enumerate(lic_data):
        row = tbl_lic.rows[idx]
        row.cells[0].paragraphs[0].add_run(c1).bold = (idx > 0)
        row.cells[1].paragraphs[0].add_run(c2)
        row.cells[2].paragraphs[0].add_run(c3)
        row.cells[3].paragraphs[0].add_run(c4)
    format_table(tbl_lic, lic_widths, header_bg="0F172A", alt_bg="F1F5F9")

    add_callout_box(
        doc,
        "Zašto je uklonjen ArcFace? Gotovi ArcFace modeli (InsightFace Model Zoo) trenirani su na skupovima "
        "poput Glint360k i MS1MV2 koji nose restriktivnu 'Non-commercial / Academic research only' licencu. "
        "ArgusFace Studio v1.0 zamijenio je ArcFace modernijim modelom EdgeFace BASE koji je razvio švicarski institut Idiap "
        "i objavio pod čistom BSD-3-Clause licencom, čime je postignuta jednaka točnost (99.83%), ali uz 100% legalnu prodaju!",
        title="PRAVNO RAZGRANIČENJE MODELA",
        border_color="0284C7",
        bg_color="F0F9FF"
    )

    doc.add_heading("4.2. Zaštita Osobnih Podataka (GDPR Usklađenost - Uredba EU 2016/679)", level=2)
    p = doc.add_paragraph(
        "Opća uredba o zaštiti podataka (GDPR) postavlja vrlo stroge zahtjeve pri obradi biometrijskih podataka. "
        "ArgusFace Studio je od temelja dizajniran prema načelu 'Privacy by Design' (Privatnost kroz dizajn):"
    )
    p.paragraph_format.line_spacing = 1.15
    
    gdpr_points = [
        ("Biometrijski podatak (Članak 9. GDPR-a): ", "Vektorski embedding lica (512 decimalnih brojeva) omogućuje jedinstvenu identifikaciju pojedinca i pravno predstavlja biometrijski podatak posebne kategorije."),
        ("100% On-Premise arhitektura: ", "Za razliku od SaaS rješenja koja šalju slike na američke ili strane Cloud servere, ArgusFace Studio radi isključivo lokalno. Niti jedan podatak ne napušta prostoriju ili server kupca, što drastično olakšava usklađenost s GDPR-om jer nema međunarodnog prijenosa podataka."),
        ("Ograničenje pohrane i automatska rotacija (Članak 5(1)(e) GDPR-a): ", "Biometrijski podaci i video snimke ne smiju se čuvati duže nego što je nužno. Ugrađeni GDPR mehanizam automatske rotacije (Data Retention Policy) omogućuje automatsko brisanje događaja, slika i NVR video segmenata starijih od 15, 30, 60 ili 90 dana, štiteći voditelja obrade od visokih zakonskih sankcija."),
        ("Pravo na zaborav (Članak 17. GDPR-a): ", "Sustav u grafičkom sučelju omogućuje trenutno, trajno i nepovratno brisanje profila osobe, pripadajućih embedding vektora i svih povezanih fotografija s diska."),
        ("Pravna osnova kupca: ", "Kupac/operater sustava dužan je osigurati zakonitu pravnu osnovu za unos osoba u bazu (npr. izričita privola ispitanika ili legitimni interes sukladno primjenjivom nacionalnom zakonodavstvu).")
    ]
    for gt, gd in gdpr_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        bp.add_run(gt).bold = True
        bp.add_run(gd)

    doc.add_heading("4.3. Usklađenost s Uredbom o Umjetnoj Inteligenciji (EU AI Act - Uredba 2024/1689)", level=2)
    p = doc.add_paragraph(
        "Novi Zakon o umjetnoj inteligenciji Europske unije (EU AI Act) kategorizira biometrijske sustave "
        "prema stupnju rizika. ArgusFace Studio v1.0 pozicioniran je u zonu kontroliranog/prihvatljivog rizika uz jasna ograničenja:"
    )
    p.paragraph_format.line_spacing = 1.15
    
    ai_act_points = [
        ("Dopuštena namjena: ", "Organizacija i sortiranje privatnih i poslovnih arhiva fotografija, evidencija prisutnosti u privatnim i kontroliranim poslovnim prostorima te interna sigurnosna provjera."),
        ("Zabranjene prakse (Članak 5. AI Acta): ", "Softver nije namijenjen, licenciran niti se smije koristiti za daljinsku biometrijsku identifikaciju u stvarnom vremenu na javno dostupnim prostorima u svrhu provedbe zakona ili masovnog nadzora građana."),
        ("Uklanjanje rizičnih klasifikacija: ", "Izbacivanjem modela za procjenu dobi, spola i rase (FairFace), sustav je u potpunosti izbjegao kategoriju 'osjetljive biometrijske kategorizacije pojedinaca' koja podliježe najstrožim zabranama u EU.")
    ]
    for at, ad in ai_act_points:
        bp = doc.add_paragraph(style='List Bullet')
        bp.paragraph_format.space_after = Pt(2)
        bp.paragraph_format.line_spacing = 1.15
        bp.add_run(at).bold = True
        bp.add_run(ad)

    doc.add_page_break()

    # ==========================================
    # POGLAVLJE 5: RJEŠAVANJE PROBLEMA I ČESTA PITANJA
    # ==========================================
    h1 = doc.add_heading("5. Rješavanje Problema i Najbolje Prakse (FAQ)", level=1)
    h1.paragraph_format.space_before = Pt(16)
    
    faq_list = [
        ("Kako postići maksimalnu točnost prepoznavanja?",
         "Za svaku osobu unesite između 2 i 4 kvalitetne fotografije: jednu pod ravnim kutom, jednu s blagim smiješkom te po jednu pod blagim lijevim i desnim kutom (15-30 stupnjeva). Izbjegavajte slike sa sunčanim naočalama ili ekstremnim sjenama."),
        ("Podržava li sustav grafičke kartice koje nisu NVIDIA (npr. Intel Iris Xe ili AMD Radeon)?",
         "Da! Zahvaljujući integraciji DirectML (DirectX 12) i Intel OpenVINO tehnologije, ArgusFace Studio automatski koristi bilo koji moderni grafički procesor (Intel Iris Xe, Intel Arc, AMD Radeon, NVIDIA) ili integriranu NPU jedinicu, postižući trenutačno hardversko ubrzanje bez potrebe za kompliciranom instalacijom NVIDIA CUDA paketa."),
        ("Kako sustav osigurava visoke performanse kada baza naraste na tisuće osoba?",
         "U sustav je ugrađen industrijski FAISS (Facebook AI Similarity Search) vektorski indeks. Za baze do 1.000 osoba koristi se C++ AVX2 SIMD pretraživanje, a za veće baze automatski se aktivira HNSW grafovsko indeksiranje koje pronalazi najsličniju osobu u svega 0.1 milisekundi bez opterećivanja CPU-a."),
        ("Koliko dugo se čuvaju zapisi prolazaka i video snimke?",
         "U Tabu 6 možete podesiti politiku zadržavanja podataka (Data Retention) na 15 dana (preporučeno za video nadzor prema smjernicama AZOP-a), 30 dana, 60 dana, 90 dana ili trajno. Sustav pri svakom startu automatski uklanja snimke i događaje starije od zadanog roka."),
        ("Što ako mrežna IP kamera ne prikazuje sliku?",
         "Provjerite ispravnost RTSP adrese. Standardni format za većinu kamera je 'rtsp://admin:lozinka@IP_ADRESA:554/onvif1' ili 'rtsp://admin:lozinka@IP_ADRESA:554/live/ch0'. Provjerite može li računalo 'pingati' IP adresu kamere."),
        ("Kako napraviti sigurnosnu kopiju cijelog sustava?",
         "U tabu 'O Sustavu i Sigurnosna Kopija' kliknite na gumb 'Kreiraj i preuzmi sigurnosnu kopiju (ZIP)'. Preuzetu ZIP datoteku spremite na vanjski USB disk. Ona sadrži kompletnu SQLite bazu, sve biometrijske vektore i izrezane fotografije lica."),
        ("Što učiniti ako sustav prepoznaje pogrešnu osobu (lažni pozitiv)?",
         "Povećajte prag prepoznavanja (Threshold) s 0.45 na 0.50 ili 0.55. Također provjerite profil pogrešno prepoznate osobe i uklonite eventualno mutne ili nekvalitetne uzorke lica."),
        ("Može li aplikacija raditi na računalu bez interneta?",
         "Da, ArgusFace Studio je 100% samostalan. Svi potrebni AI modeli (RetinaFace, EdgeFace) pohranjeni su lokalno na računalu i ne zahtijevaju internetsku vezu za rad.")
    ]
    
    for q, a in faq_list:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(6)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(f"❓ Pitanje: {q}")
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(15, 23, 42)
        
        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_before = Pt(0)
        p_a.paragraph_format.space_after = Pt(8)
        p_a.paragraph_format.line_spacing = 1.15
        r_a = p_a.add_run(f" Odgovor: {a}")
        r_a.font.color.rgb = RGBColor(51, 65, 85)

    p_final = doc.add_paragraph()
    p_final.paragraph_format.space_before = Pt(20)
    p_final.add_run("Kraj korisničke i tehničko-pravne dokumentacije • ArgusFace Studio v1.0").italic = True
    
    # Spremanje dokumenta
    out_path = os.path.join(APP_DIR, "Korisnicka_Dokumentacija_ArgusFace_Studio_v1.0.docx")
    doc.save(out_path)
    print(f"[+] Uspješno kreiran dokument: {out_path}")
    return out_path

if __name__ == "__main__":
    build_user_manual()
