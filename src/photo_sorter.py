import os
import sys
import time
import shutil
import csv
import cv2
import numpy as np
from typing import Callable, Optional, List, Dict, Any, Set

# Ensure src is on path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import db
from face_engine import get_analyzer, FaceIndex
from image_utils import imread_unicode

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}

class PhotoSorter:
    """
    Automatski razvrstava i sortira velike mape fotografija (događaji, konferencije, natjecanja, albumi, arhive)
    prema prepoznatim osobama iz biometrijske baze.
    """
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        target_person_ids: Optional[List[int]] = None,
        threshold: float = 0.48,
        file_action: str = "hardlink",  # "hardlink", "copy", "move"
        group_min_faces: int = 4,
        enable_group_folder: bool = True,
        enable_combo_folder: bool = True,
        enable_no_face_folder: bool = True,
        enable_unregistered_folder: bool = False,
        enable_other_guests_folder: Optional[bool] = None,
        max_det_dim: int = 1600,
        device: str = "CPU"
    ):
        self.input_dir = os.path.abspath(input_dir)
        self.output_dir = os.path.abspath(output_dir)
        self.target_person_ids = target_person_ids
        self.threshold = float(threshold)
        self.file_action = file_action.lower()
        self.group_min_faces = int(group_min_faces)
        self.enable_group_folder = enable_group_folder
        self.enable_combo_folder = enable_combo_folder
        self.enable_no_face_folder = enable_no_face_folder
        if enable_other_guests_folder is not None:
            self.enable_unregistered_folder = enable_other_guests_folder
        else:
            self.enable_unregistered_folder = enable_unregistered_folder
        self.max_det_dim = max_det_dim
        self.device = device

        self.cancel_requested = False
        self.is_running = False

    def scan_files(self) -> List[str]:
        """Pronalazi sve podržane slikovne datoteke u ulaznoj mapi."""
        files = []
        if not os.path.isdir(self.input_dir):
            return files

        for root, _, filenames in os.walk(self.input_dir):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_IMAGE_EXTS:
                    files.append(os.path.join(root, fn))

        files.sort()
        return files

    def _transfer_file(self, src: str, dst_folder: str) -> str:
        """Kopira, premješta ili stvara hardlink datoteke u ciljnu mapu."""
        os.makedirs(dst_folder, exist_ok=True)
        fname = os.path.basename(src)
        dst = os.path.join(dst_folder, fname)

        # Izbjegni prepisivanje ako datoteka s istim imenom već postoji
        if os.path.exists(dst):
            base, ext = os.path.splitext(fname)
            idx = 1
            while os.path.exists(dst):
                dst = os.path.join(dst_folder, f"{base}_{idx}{ext}")
                idx += 1

        if self.file_action == "move":
            shutil.move(src, dst)
        elif self.file_action == "hardlink" and sys.platform == "win32":
            try:
                os.link(src, dst)
            except Exception:
                # Ako hardlink ne uspije (npr. različiti diskovi ili particije), automatski fallback na copy
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

        return dst

    def run(self, progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        Pokreće proces skeniranja, biometrijskog prepoznavanja i sortiranja fotografija.
        """
        self.cancel_requested = False
        self.is_running = True
        t0 = time.time()

        files = self.scan_files()
        total_files = len(files)

        if total_files == 0:
            self.is_running = False
            return {
                "status": "error",
                "message": f"Nema podržanih slika u ulaznoj mapi: {self.input_dir}",
                "total_files": 0,
                "processed": 0
            }

        # 1. Izgradi biometrijski indeks osoba iz baze
        all_samples = db.get_cached_embeddings()
        if not all_samples:
            self.is_running = False
            return {
                "status": "error",
                "message": "Baza osoba je prazna! Molimo unesite barem jednu osobu u bazu prije sortiranja.",
                "total_files": total_files,
                "processed": 0
            }

        # Filtriraj uzorke ako su odabrane samo specifične osobe
        if self.target_person_ids:
            target_ids_set = set(self.target_person_ids)
            filtered_samples = [s for s in all_samples if s["person_id"] in target_ids_set]
            if not filtered_samples:
                filtered_samples = all_samples
        else:
            filtered_samples = all_samples

        face_index = FaceIndex(filtered_samples)
        # Koristi analyzer bez atributa za maksimalnu brzinu obrade
        analyzer = get_analyzer(device=self.device, with_attributes=False)

        # Statistika i rezultati
        stats_by_person: Dict[str, int] = {}
        total_matched_photos = 0
        total_faces_found = 0
        csv_rows = []
        recent_previews = []

        os.makedirs(self.output_dir, exist_ok=True)

        for i, file_path in enumerate(files):
            if self.cancel_requested:
                break

            fname = os.path.basename(file_path)
            t_file_start = time.time()

            try:
                img_bgr, _ = imread_unicode(file_path)
            except Exception:
                img_bgr = None

            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]

            # Skaliranje za visoku brzinu detekcije na velikim rezolucijama (fotoaparati 24-50 MP)
            if max(h, w) > self.max_det_dim:
                scale = self.max_det_dim / max(h, w)
                det_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                det_bgr = img_bgr

            # Detekcija i prepoznavanje lica
            try:
                raw_faces = analyzer.analyze(det_bgr)
            except Exception:
                raw_faces = []

            face_count = len(raw_faces)
            total_faces_found += face_count

            photo_matched_persons: Dict[str, float] = {}  # ime_osobe -> similarity

            for f in raw_faces:
                emb = f.embedding
                if emb is not None:
                    match_res = face_index.match(emb, threshold=self.threshold)
                    if match_res.get("matched"):
                        p_name = match_res["person_name"]
                        sim = match_res["similarity"]
                        if p_name not in photo_matched_persons or sim > photo_matched_persons[p_name]:
                            photo_matched_persons[p_name] = sim

            # Određivanje odredišnih mapa za ovu fotografiju
            assigned_folders: Set[str] = set()

            if photo_matched_persons:
                total_matched_photos += 1
                for p_name in photo_matched_persons.keys():
                    stats_by_person[p_name] = stats_by_person.get(p_name, 0) + 1
                    # Sigurna mapa po osobi
                    safe_pname = "".join(c for c in p_name if c.isalnum() or c in (" ", "_", "-")).strip()
                    assigned_folders.add(os.path.join(self.output_dir, f"Osoba_{safe_pname}"))

                # Ako je prepoznato više ciljanih osoba na istoj slici
                if self.enable_combo_folder and len(photo_matched_persons) >= 2:
                    assigned_folders.add(os.path.join(self.output_dir, "Zajedno_Ciljane_Osobe"))

            else:
                if face_count == 0:
                    if self.enable_no_face_folder:
                        assigned_folders.add(os.path.join(self.output_dir, "Fotografije_Bez_Lica"))
                else:
                    if self.enable_unregistered_folder:
                        assigned_folders.add(os.path.join(self.output_dir, "Neregistrirana_Lica"))

            # Grupne fotografije (ako ima >= N lica)
            if self.enable_group_folder and face_count >= self.group_min_faces:
                assigned_folders.add(os.path.join(self.output_dir, "Grupne_Fotografije"))

            # Izvrši prijenos / povezivanje u dodijeljene mape
            for folder in assigned_folders:
                try:
                    self._transfer_file(file_path, folder)
                except Exception:
                    pass

            t_file_ms = (time.time() - t_file_start) * 1000

            # Zapis za CSV izvještaj
            matched_str = "; ".join([f"{name} ({sim*100:.1f}%)" for name, sim in photo_matched_persons.items()]) if photo_matched_persons else ("Nema" if face_count > 0 else "Nema lica")
            folders_str = "; ".join([os.path.basename(f) for f in assigned_folders]) if assigned_folders else "Nesortirano"

            csv_rows.append([
                fname,
                file_path,
                f"{w}x{h}",
                face_count,
                matched_str,
                folders_str,
                f"{t_file_ms:.1f}"
            ])

            # Ažuriraj thumbnail za pregled u sučelju
            if photo_matched_persons and len(recent_previews) < 8:
                recent_previews.insert(0, {
                    "path": file_path,
                    "filename": fname,
                    "matched": list(photo_matched_persons.keys()),
                    "faces": face_count
                })

            # Slanje napretka
            if progress_callback and (i % 2 == 0 or i == total_files - 1):
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed if elapsed > 0 else 0.0
                rem_files = total_files - (i + 1)
                eta = rem_files / fps if fps > 0 else 0.0

                progress_data = {
                    "current": i + 1,
                    "total": total_files,
                    "percent": round(((i + 1) / total_files) * 100, 1),
                    "current_file": fname,
                    "fps": round(fps, 1),
                    "elapsed_sec": int(elapsed),
                    "eta_sec": int(eta),
                    "matched_photos": total_matched_photos,
                    "total_faces": total_faces_found,
                    "stats_by_person": stats_by_person,
                    "recent_previews": recent_previews[:6]
                }
                progress_callback(progress_data)

        # 2. Generiraj CSV izvještaj
        report_path = os.path.join(self.output_dir, "izvjestaj_sortiranja.csv")
        try:
            with open(report_path, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.writer(f, delimiter=";")
                writer.writerow([
                    "Naziv datoteke",
                    "Puna putanja",
                    "Rezolucija",
                    "Broj detektiranih lica",
                    "Prepoznate osobe",
                    "Dodijeljene mape",
                    "Vrijeme obrade (ms)"
                ])
                writer.writerows(csv_rows)
        except Exception:
            report_path = ""

        total_elapsed = time.time() - t0
        self.is_running = False

        return {
            "status": "cancelled" if self.cancel_requested else "success",
            "total_files": total_files,
            "processed": len(csv_rows),
            "matched_photos": total_matched_photos,
            "total_faces": total_faces_found,
            "stats_by_person": stats_by_person,
            "output_dir": self.output_dir,
            "report_path": report_path,
            "elapsed_sec": round(total_elapsed, 1),
            "avg_fps": round(len(csv_rows) / total_elapsed, 1) if total_elapsed > 0 else 0.0
        }

    def cancel(self):
        """Prekida trenutni proces obrade."""
        self.cancel_requested = True
