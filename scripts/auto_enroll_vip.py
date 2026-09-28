"""
Automatski unos 50 javnih osoba iz političkog, društvenog i sportskog života Srbije i BiH.
Preuzima portrete visoke kvalitete putem Wikimedia Commons / Wikipedia API-ja,
vrši biometrijsku detekciju lica, sprema 3-4 različita uzorka po osobi i rekalkulira FAISS indeks.
"""

import os
import sys
import time
import json
import uuid
import urllib.request
import urllib.parse
import cv2
import numpy as np

# Ensure src is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(CURRENT_DIR)
SRC_DIR = os.path.join(APP_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import db
import face_engine
from image_utils import imwrite_unicode
from app import save_image_dedup, UPLOADS_DIR, CROPS_DIR

HEADERS = {
    "User-Agent": "ArgusFaceBot/1.0 (internal-demo@argusface.local; python-urllib)"
}

PEOPLE = [
    # --- BiH Politika & Društvo (15) ---
    {
        "name": "Milorad Dodik",
        "notes": "Politika - Predsjednik RS (BiH)",
        "terms": ["Milorad Dodik", "Dodik portrait"]
    },
    {
        "name": "Željka Cvijanović",
        "notes": "Politika - Članica Predsjedništva BiH",
        "terms": ["Zeljka Cvijanovic", "Željka Cvijanović", "Cvijanovic portrait"]
    },
    {
        "name": "Denis Bećirović",
        "notes": "Politika - Član Predsjedništva BiH",
        "terms": ["Denis Becirovic", "Denis Bećirović", "Becirovic portrait"]
    },
    {
        "name": "Željko Komšić",
        "notes": "Politika - Član Predsjedništva BiH",
        "terms": ["Zeljko Komsic", "Željko Komšić", "Komsic portrait"]
    },
    {
        "name": "Dragan Čović",
        "notes": "Politika - Predsjednik HDZ BiH",
        "terms": ["Dragan Covic", "Dragan Čović", "Covic portrait"]
    },
    {
        "name": "Bakir Izetbegović",
        "notes": "Politika - Predsjednik SDA (BiH)",
        "terms": ["Bakir Izetbegovic", "Bakir Izetbegović", "Izetbegovic portrait"]
    },
    {
        "name": "Borjana Krišto",
        "notes": "Politika - Predsjedateljica Vijeća ministara BiH",
        "terms": ["Borjana Kristo", "Borjana Krišto", "Kristo portrait"]
    },
    {
        "name": "Nermin Nikšić",
        "notes": "Politika - Premijer FBiH",
        "terms": ["Nermin Niksic", "Nermin Nikšić", "Niksic portrait"]
    },
    {
        "name": "Elmedin Konaković",
        "notes": "Politika - Ministar vanjskih poslova BiH",
        "terms": ["Elmedin Konakovic", "Dino Konakovic", "Elmedin Konaković"]
    },
    {
        "name": "Draško Stanivuković",
        "notes": "Politika - Gradonačelnik Banja Luke (BiH)",
        "terms": ["Drasko Stanivukovic", "Draško Stanivuković", "Stanivukovic portrait"]
    },
    {
        "name": "Benjamina Karić",
        "notes": "Politika - Političarka (BiH)",
        "terms": ["Benjamina Karic", "Benjamina Karić", "Karic portrait"]
    },
    {
        "name": "Jelena Trivić",
        "notes": "Politika - Političarka RS (BiH)",
        "terms": ["Jelena Trivic", "Jelena Trivić", "Trivic portrait"]
    },
    {
        "name": "Mirko Šarović",
        "notes": "Politika - Političar SDS (BiH)",
        "terms": ["Mirko Sarovic", "Mirko Šarović", "Sarovic portrait"]
    },
    {
        "name": "Edin Forto",
        "notes": "Politika - Ministar prometa i komunikacija BiH",
        "terms": ["Edin Forto", "Forto portrait"]
    },
    {
        "name": "Fahrudin Radončić",
        "notes": "Politika i mediji - Predsjednik SBB (BiH)",
        "terms": ["Fahrudin Radoncic", "Fahrudin Radončić", "Radoncic portrait"]
    },

    # --- Srbija Politika & Društvo (15) ---
    {
        "name": "Miloš Vučević",
        "notes": "Politika - Premijer Srbije",
        "terms": ["Milos Vucevic", "Miloš Vučević", "Vucevic portrait"]
    },
    {
        "name": "Bratislav Gašić",
        "notes": "Politika - Ministar obrane Srbije",
        "terms": ["Bratislav Gasic", "Bratislav Gašić", "Gasic portrait"]
    },
    {
        "name": "Siniša Mali",
        "notes": "Politika - Ministar financija Srbije",
        "terms": ["Sinisa Mali", "Siniša Mali", "Mali portrait"]
    },
    {
        "name": "Marko Đurić",
        "notes": "Politika - Ministar vanjskih poslova Srbije",
        "terms": ["Marko Djuric", "Marko Đurić", "Djuric portrait"]
    },
    {
        "name": "Goran Vesić",
        "notes": "Politika - Političar / Bivši ministar (Srbija)",
        "terms": ["Goran Vesic", "Goran Vesić", "Vesic portrait"]
    },
    {
        "name": "Maja Gojković",
        "notes": "Politika - Predsjednica Vlade Vojvodine",
        "terms": ["Maja Gojkovic", "Maja Gojković", "Gojkovic portrait"]
    },
    {
        "name": "Dubravka Đedović",
        "notes": "Politika - Ministrica rudarstva i energetike Srbije",
        "terms": ["Dubravka Djedovic", "Dubravka Đedović", "Djedovic portrait"]
    },
    {
        "name": "Darija Kisić",
        "notes": "Profesorica i bivša ministrica (Srbija)",
        "terms": ["Darija Kisic", "Darija Kisić", "Kisic portrait"]
    },
    {
        "name": "Aleksandar Šapić",
        "notes": "Politika - Gradonačelnik Beograda",
        "terms": ["Aleksandar Sapic", "Aleksandar Šapić", "Sapic portrait"]
    },
    {
        "name": "Dragan Đilas",
        "notes": "Politika - Predsjednik SSP (Srbija)",
        "terms": ["Dragan Djilas", "Dragan Đilas", "Djilas portrait"]
    },
    {
        "name": "Zdravko Ponoš",
        "notes": "Politika - General i predsjednik SRCE (Srbija)",
        "terms": ["Zdravko Ponos", "Zdravko Ponoš", "Ponos portrait"]
    },
    {
        "name": "Boris Tadić",
        "notes": "Politika - Bivši predsjednik Srbije",
        "terms": ["Boris Tadic", "Boris Tadić", "Tadic portrait"]
    },
    {
        "name": "Vuk Jeremić",
        "notes": "Politika - Diplomat i političar (Srbija)",
        "terms": ["Vuk Jeremic", "Vuk Jeremić", "Jeremic portrait"]
    },
    {
        "name": "Vojislav Šešelj",
        "notes": "Politika - Predsjednik SRS (Srbija)",
        "terms": ["Vojislav Seselj", "Vojislav Šešelj", "Seselj portrait"]
    },
    {
        "name": "Čedomir Jovanović",
        "notes": "Politika - Predsjednik LDP (Srbija)",
        "terms": ["Cedomir Jovanovic", "Čedomir Jovanović", "Jovanovic portrait"]
    },

    # --- BiH Sport (10) ---
    {
        "name": "Edin Džeko",
        "notes": "Sport - Nogometna legenda / Kapetan BiH",
        "terms": ["Edin Dzeko", "Edin Džeko", "Dzeko portrait"]
    },
    {
        "name": "Miralem Pjanić",
        "notes": "Sport - Nogomet (BiH)",
        "terms": ["Miralem Pjanic", "Miralem Pjanić", "Pjanic portrait"]
    },
    {
        "name": "Sead Kolašinac",
        "notes": "Sport - Nogomet (BiH)",
        "terms": ["Sead Kolasinac", "Sead Kolašinac", "Kolasinac portrait"]
    },
    {
        "name": "Ermedin Demirović",
        "notes": "Sport - Nogomet (BiH)",
        "terms": ["Ermedin Demirovic", "Ermedin Demirović", "Demirovic portrait"]
    },
    {
        "name": "Jusuf Nurkić",
        "notes": "Sport - Košarka NBA (BiH)",
        "terms": ["Jusuf Nurkic", "Jusuf Nurkić", "Nurkic portrait"]
    },
    {
        "name": "Džanan Musa",
        "notes": "Sport - Košarka Real Madrid (BiH)",
        "terms": ["Dzanan Musa", "Džanan Musa", "Musa basketball"]
    },
    {
        "name": "Sergej Barbarez",
        "notes": "Sport - Izbornik nogometne reprezentacije BiH",
        "terms": ["Sergej Barbarez", "Barbarez portrait"]
    },
    {
        "name": "Safet Sušić",
        "notes": "Sport - Nogometna legenda (BiH)",
        "terms": ["Safet Susic", "Safet Sušić", "Susic portrait"]
    },
    {
        "name": "Amel Tuka",
        "notes": "Sport - Atletika (BiH)",
        "terms": ["Amel Tuka", "Tuka athletics"]
    },
    {
        "name": "Lana Pudar",
        "notes": "Sport - Plivanje (BiH)",
        "terms": ["Lana Pudar", "Pudar swimming"]
    },

    # --- Srbija Sport (10) ---
    {
        "name": "Novak Đoković",
        "notes": "Sport - Teniski šampion (Srbija)",
        "terms": ["Novak Djokovic", "Novak Đoković", "Djokovic portrait"]
    },
    {
        "name": "Nikola Jokić",
        "notes": "Sport - Košarka NBA MVP (Srbija)",
        "terms": ["Nikola Jokic", "Nikola Jokić", "Jokic portrait"]
    },
    {
        "name": "Bogdan Bogdanović",
        "notes": "Sport - Košarka NBA / Kapetan (Srbija)",
        "terms": ["Bogdan Bogdanovic", "Bogdan Bogdanović", "Bogdanovic basketball"]
    },
    {
        "name": "Dušan Tadić",
        "notes": "Sport - Nogomet (Srbija)",
        "terms": ["Dusan Tadic", "Dušan Tadić", "Tadic football"]
    },
    {
        "name": "Aleksandar Mitrović",
        "notes": "Sport - Nogomet (Srbija)",
        "terms": ["Aleksandar Mitrovic", "Aleksandar Mitrović", "Mitrovic football"]
    },
    {
        "name": "Dušan Vlahović",
        "notes": "Sport - Nogomet Juventus (Srbija)",
        "terms": ["Dusan Vlahovic", "Dušan Vlahović", "Vlahovic portrait"]
    },
    {
        "name": "Nemanja Matić",
        "notes": "Sport - Nogomet (Srbija)",
        "terms": ["Nemanja Matic", "Nemanja Matić", "Matic portrait"]
    },
    {
        "name": "Dragan Stojković Piksi",
        "notes": "Sport - Nogometni izbornik (Srbija)",
        "terms": ["Dragan Stojkovic", "Dragan Stojković", "Piksi Stojkovic"]
    },
    {
        "name": "Željko Obradović",
        "notes": "Sport - Košarkaški trener Partizan (Srbija)",
        "terms": ["Zeljko Obradovic", "Željko Obradović", "Obradovic coach"]
    },
    {
        "name": "Ivana Španović",
        "notes": "Sport - Atletika (Srbija)",
        "terms": ["Ivana Spanovic", "Ivana Španović", "Ivana Vuleta"]
    }
]

SKIP_KEYWORDS = [
    "flag", "coat", "signature", "logo", "map", "icon", "stemma", "herb",
    "grb", "zastava", "potpis", "badge", "diagram", "stadium", "building"
]

def search_wikimedia(query, limit=15):
    """Pronalazi URL-ove slika na Wikimedia Commons za zadani upit."""
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&generator=search"
        f"&gsrnamespace=6&gsrsearch={urllib.parse.quote(query)}"
        f"&gsrlimit={limit}&prop=imageinfo&iiprop=url|size|mime&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    urls = []
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                for ii in p.get("imageinfo", []):
                    mime = ii.get("mime", "")
                    u = ii.get("url", "")
                    if mime in ["image/jpeg", "image/png"]:
                        u_lower = u.lower()
                        if not any(skip in u_lower for skip in SKIP_KEYWORDS):
                            if u not in urls:
                                urls.append(u)
    except Exception:
        pass
    return urls

def search_wikipedia_page_images(title, lang="sr"):
    """Dohvaća glavne slike sa stranice na Wikipediji kao alternativni izvor."""
    url = (
        f"https://{lang}.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(title)}"
        f"&prop=pageimages|images&pithumbsize=1000&format=json"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    urls = []
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, p in pages.items():
                thumb = p.get("thumbnail", {}).get("source")
                if thumb and not any(skip in thumb.lower() for skip in SKIP_KEYWORDS):
                    urls.append(thumb)
    except Exception:
        pass
    return urls

def download_image(url):
    """Preuzima sliku s weba i dekodira je u OpenCV BGR format."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=12) as resp:
            arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img_bgr
    except Exception:
        return None

def process_person(person_info, target_samples=3):
    """Preuzima slike, ekstrahira lica i sprema osobu u bazu."""
    name = person_info["name"]
    notes = person_info["notes"]
    terms = person_info["terms"]

    # 1. Prikupi kandidate za URL-ove
    candidate_urls = []
    for term in terms:
        urls = search_wikimedia(term, limit=12)
        for u in urls:
            if u not in candidate_urls:
                candidate_urls.append(u)
        if len(candidate_urls) >= 15:
            break

    # Dopuna sa srpske/bosanske Wikipedije ako fali
    if len(candidate_urls) < 6:
        for lang in ["sr", "bs", "en"]:
            wiki_urls = search_wikipedia_page_images(name, lang=lang)
            for wu in wiki_urls:
                if wu not in candidate_urls:
                    candidate_urls.append(wu)

    if not candidate_urls:
        return 0, "Nisu pronađene slike"

    # 2. Dohvati ili kreiraj osobu u bazi
    person_id = db.get_or_create_person(name, notes)

    existing_samples = db.get_person_samples(person_id)
    accepted_embeddings = [s["embedding"] for s in existing_samples if s.get("embedding") is not None]

    if len(accepted_embeddings) >= target_samples:
        return len(accepted_embeddings), "Već ima dovoljno uzoraka"

    saved_count = len(accepted_embeddings)

    for u in candidate_urls:
        if saved_count >= target_samples:
            break

        img_bgr = download_image(u)
        if img_bgr is None:
            continue

        h_img, w_img = img_bgr.shape[:2]
        if h_img < 100 or w_img < 100:
            continue

        faces = face_engine.extract_faces_from_image(img_bgr)
        if not faces:
            continue

        # Sortiraj lica po veličini bounding boxa (najveće prvo)
        faces.sort(key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]), reverse=True)
        best_face = faces[0]

        fw = best_face["bbox"][2] - best_face["bbox"][0]
        fh = best_face["bbox"][3] - best_face["bbox"][1]

        # Provjera minimalne rezolucije lica i pouzdanosti modela
        if fw < 50 or fh < 50 or best_face["confidence"] < 0.70:
            continue

        # Ako na slici ima više lica, prihvati samo ako je glavno lice dominantno
        if len(faces) > 1:
            second_area = (faces[1]["bbox"][2] - faces[1]["bbox"][0]) * (faces[1]["bbox"][3] - faces[1]["bbox"][1])
            best_area = fw * fh
            if best_area < 1.3 * second_area and best_face["confidence"] < 0.85:
                continue

        # Deduplikacija: izbjegni spremanje identičnih slika ili istih kadrova
        is_duplicate = False
        emb = best_face["embedding"]
        for prev_emb in accepted_embeddings:
            sim = float(np.dot(emb, prev_emb))
            if sim > 0.95:  # Gotovo identičan kadar
                is_duplicate = True
                break

        if is_duplicate:
            continue

        # Spremi original i crop
        orig_path = save_image_dedup(img_bgr, UPLOADS_DIR, prefix="orig")
        crop_filename = f"crop_{person_id}_{uuid.uuid4().hex[:8]}.jpg"
        crop_path = os.path.join(CROPS_DIR, crop_filename)
        imwrite_unicode(crop_path, best_face["crop_bgr"])

        # Upiši u SQLite bazu
        db.add_face_sample(person_id, orig_path, crop_path, emb, best_face["confidence"])
        accepted_embeddings.append(emb)
        saved_count += 1
        time.sleep(0.15)  # Ljubaznost prema Wikimedia poslužiteljima

    return saved_count, "Uspješno"

def main():
    print("=" * 70)
    print("  ARGUSFACE STUDIO - AUTOMATIZIRANI UNOS 50 JAVNIH OSOBA (BiH & SRB)")
    print("=" * 70)
    print(f"Ciljani broj osoba: {len(PEOPLE)}")
    print(f"Ciljano uzoraka po osobi: 3 do 4 kvalitetne profilne fotografije")
    print("-" * 70)

    start_time = time.time()
    success_count = 0
    total_images_added = 0

    for idx, p in enumerate(PEOPLE, 1):
        name = p["name"]
        print(f"[{idx:02d}/{len(PEOPLE)}] Obrada: {name} ... ", end="", flush=True)

        try:
            samples_cnt, status = process_person(p, target_samples=3)
            if samples_cnt >= 1:
                badge = "Zlatni (3+ slike)" if samples_cnt >= 3 else f"Profil ({samples_cnt} sl.)"
                print(f"✅ {samples_cnt} slika [{badge}]")
                success_count += 1
                total_images_added += samples_cnt
            else:
                print(f"⚠️ {status}")
        except Exception as e:
            print(f"❌ Greška: {e}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"Završena obrada u {elapsed:.1f} sekundi.")
    print(f"Uspješno unesenih osoba: {success_count}/{len(PEOPLE)}")
    print(f"Ukupno novih slika s biometrijom: {total_images_added}")
    print("-" * 70)

    # Rekalkulacija i ažuriranje FAISS vektorskog indeksa
    print("🔄 Obnavljanje FAISS vektorskog indeksa (FlatIP / HNSW) ...")
    try:
        from face_engine import get_face_index
        idx = get_face_index(force_refresh=True)
        print(f"✅ FAISS Vektorski indeks uspješno obnovljen! [{idx.backend}] Ukupno vektora u indeksu: {idx.total_samples}")
    except Exception as e:
        print(f"⚠️ Obnova indeksa: {e}")

    print("=" * 70)
    print("  AUTOMATIZIRANI UNOS JE USPJEŠNO ZAVRŠEN!")
    print("=" * 70)

if __name__ == "__main__":
    main()
