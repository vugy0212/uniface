import sqlite3
import os
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "database.db")

def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                notes TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id INTEGER NOT NULL,
                image_path TEXT NOT NULL,
                crop_path TEXT NOT NULL,
                embedding BLOB NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (person_id) REFERENCES persons(id) ON DELETE CASCADE
            );
        """)
    conn.close()

def add_person(name: str, notes: str = "") -> int:
    name = name.strip()
    if not name:
        raise ValueError("Ime osobe ne smije biti prazno.")
    conn = get_connection()
    with conn:
        cur = conn.execute("INSERT INTO persons (name, notes) VALUES (?, ?)", (name, notes.strip()))
        person_id = cur.lastrowid
    conn.close()
    return person_id

def get_or_create_person(name: str, notes: str = "") -> int:
    name = name.strip()
    if not name:
        raise ValueError("Ime osobe ne smije biti prazno.")
    conn = get_connection()
    row = conn.execute("SELECT id FROM persons WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
    if row:
        conn.close()
        return row["id"]
    with conn:
        cur = conn.execute("INSERT INTO persons (name, notes) VALUES (?, ?)", (name, notes.strip()))
        person_id = cur.lastrowid
    conn.close()
    return person_id

def update_person(person_id: int, name: str, notes: str = ""):
    conn = get_connection()
    with conn:
        conn.execute("UPDATE persons SET name = ?, notes = ? WHERE id = ?", (name.strip(), notes.strip(), person_id))
    conn.close()

def add_face_sample(person_id: int, image_path: str, crop_path: str, embedding: np.ndarray, confidence: float = 1.0) -> int:
    embedding_bytes = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
    conn = get_connection()
    with conn:
        cur = conn.execute(
            """INSERT INTO face_samples (person_id, image_path, crop_path, embedding, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (person_id, image_path, crop_path, embedding_bytes, float(confidence))
        )
        sample_id = cur.lastrowid
    conn.close()
    return sample_id

def get_all_persons():
    conn = get_connection()
    rows = conn.execute("""
        SELECT p.id, p.name, p.notes, p.created_at,
               COUNT(s.id) as sample_count,
               MAX(s.created_at) as last_sample_at
        FROM persons p
        LEFT JOIN face_samples s ON p.id = s.person_id
        GROUP BY p.id
        ORDER BY p.name COLLATE NOCASE ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_person(person_id: int):
    conn = get_connection()
    row = conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_person_samples(person_id: int):
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, person_id, image_path, crop_path, confidence, created_at
        FROM face_samples
        WHERE person_id = ?
        ORDER BY created_at DESC
    """, (person_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_person(person_id: int):
    conn = get_connection()
    samples = conn.execute("SELECT image_path, crop_path FROM face_samples WHERE person_id = ?", (person_id,)).fetchall()
    for s in samples:
        for p in [s["image_path"], s["crop_path"]]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    with conn:
        conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    conn.close()

def delete_sample(sample_id: int):
    conn = get_connection()
    row = conn.execute("SELECT image_path, crop_path FROM face_samples WHERE id = ?", (sample_id,)).fetchone()
    if row:
        for p in [row["image_path"], row["crop_path"]]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass
    with conn:
        conn.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
    conn.close()

def get_all_embeddings():
    conn = get_connection()
    rows = conn.execute("""
        SELECT s.id as sample_id, s.person_id, p.name as person_name,
               s.embedding, s.crop_path, s.confidence
        FROM face_samples s
        JOIN persons p ON s.person_id = p.id
    """).fetchall()
    conn.close()
    
    result = []
    for r in rows:
        emb_arr = np.frombuffer(r["embedding"], dtype=np.float32)
        result.append({
            "sample_id": r["sample_id"],
            "person_id": r["person_id"],
            "person_name": r["person_name"],
            "crop_path": r["crop_path"],
            "confidence": r["confidence"],
            "embedding": emb_arr
        })
    return result

def get_stats():
    conn = get_connection()
    person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    sample_count = conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]
    conn.close()
    return {
        "total_persons": person_count,
        "total_samples": sample_count
    }

init_db()
