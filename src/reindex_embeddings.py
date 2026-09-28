"""
reindex_embeddings.py
Osvježava postojeće embeddinge lica u SQLite bazi pomoću novog komercijalno čistog modela EdgeFace BASE.
"""
import os
import sys
import sqlite3
import cv2
import numpy as np

# Postavljanje putanje
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from src import face_engine, db

def reindex_all_embeddings(db_path: str = None) -> tuple[int, int, int]:
    """
    Prolazi kroz sve uzorke lica u bazi i ponovno računa 512-D vektore pomoću EdgeFace BASE.
    Vraća (uspješno_ažurirano, neuspjelo, ukupno).
    """
    if db_path is None:
        db_path = db.DB_PATH
        
    print(f"[*] Pokrećem reindeksiranje baze: {db_path}")
    print("[*] Koristi se model: EdgeFace BASE (BSD-3-Clause, komercijalno čist)")
    
    # Inicijalizacija analyzera jednom (AUTO omogucuje GPU / DirectML akceleraciju)
    analyzer = face_engine.get_analyzer(device="AUTO", with_attributes=False)
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT id, person_id, image_path, crop_path FROM face_samples ORDER BY id ASC")
    samples = cursor.fetchall()
    total = len(samples)
    print(f"[*] Pronađeno ukupno uzoraka u bazi: {total}")
    
    if total == 0:
        print("[!] Baza je prazna, nema uzoraka za ažuriranje.")
        conn.close()
        return (0, 0, 0)
        
    updated = 0
    failed = 0
    
    for idx, s in enumerate(samples, start=1):
        sample_id = s["id"]
        crop_path = s["crop_path"]
        image_path = s["image_path"]
        
        new_emb = None
        
        # 1. Pokušaj iz isječka lica (crop)
        if crop_path and os.path.exists(crop_path):
            img = cv2.imread(crop_path)
            if img is not None:
                faces = analyzer.analyze(img)
                if len(faces) > 0 and faces[0].embedding is not None:
                    new_emb = faces[0].embedding
                    
        # 2. Ako u cropu nije detektiran, probaj iz originalne slike
        if new_emb is None and image_path and os.path.exists(image_path):
            img = cv2.imread(image_path)
            if img is not None:
                faces = analyzer.analyze(img)
                if len(faces) > 0 and faces[0].embedding is not None:
                    new_emb = faces[0].embedding
                    
        if new_emb is not None:
            # Normalizacija vektora
            norm = np.linalg.norm(new_emb)
            if norm > 0:
                new_emb = new_emb / norm
            emb_bytes = np.ascontiguousarray(new_emb, dtype=np.float32).tobytes()
            cursor.execute("UPDATE face_samples SET embedding = ? WHERE id = ?", (emb_bytes, sample_id))
            updated += 1
            if idx % 10 == 0 or idx == total:
                print(f"    -> Progres: {idx}/{total} ({updated} uspješno)")
        else:
            print(f"    [!] Upozorenje: Nije moguće izvući lice za uzorak ID={sample_id} ({crop_path})")
            failed += 1
            
    conn.commit()
    conn.close()
    
    # Invalidate cached embeddings in memory
    db.invalidate_cache()
    
    print(f"[+] Reindeksiranje završeno! Ažurirano: {updated}/{total}, Neuspjelo: {failed}")
    return (updated, failed, total)

if __name__ == "__main__":
    reindex_all_embeddings()
