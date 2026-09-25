import sqlite3
import os
from contextlib import contextmanager
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "database.db")

_cached_embeddings = None
_cache_invalidation_callbacks = []

def register_cache_invalidation_callback(cb):
    """Registers a callable to be notified when database records change."""
    if cb not in _cache_invalidation_callbacks:
        _cache_invalidation_callbacks.append(cb)

def invalidate_cache():
    """Invalidates the in-memory embeddings cache and notifies registered listeners."""
    global _cached_embeddings
    _cached_embeddings = None
    for cb in list(_cache_invalidation_callbacks):
        try:
            cb()
        except Exception:
            pass

@contextmanager
def get_db():
    """Context manager for SQLite connection that guarantees closing."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def get_connection():
    """Returns a standalone connection (caller responsible for closing)."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_name TEXT NOT NULL,
                    similarity REAL NOT NULL,
                    source_label TEXT NOT NULL,
                    crop_path TEXT DEFAULT '',
                    snapshot_path TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            try:
                conn.execute("ALTER TABLE detection_events ADD COLUMN snapshot_path TEXT DEFAULT '';")
            except Exception:
                pass
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_face_samples_person_id ON face_samples(person_id);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detection_events_created ON detection_events(created_at);
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_detection_events_name ON detection_events(person_name);
            """)

def add_person(name: str, notes: str = "") -> int:
    name = name.strip()
    if not name:
        raise ValueError("Ime osobe ne smije biti prazno.")
    with get_db() as conn:
        with conn:
            cur = conn.execute("INSERT INTO persons (name, notes) VALUES (?, ?)", (name, notes.strip()))
            person_id = cur.lastrowid
    invalidate_cache()
    return person_id

def get_or_create_person(name: str, notes: str = "") -> int:
    name = name.strip()
    if not name:
        raise ValueError("Ime osobe ne smije biti prazno.")
    with get_db() as conn:
        row = conn.execute("SELECT id FROM persons WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
        if row:
            return row["id"]
        with conn:
            cur = conn.execute("INSERT INTO persons (name, notes) VALUES (?, ?)", (name, notes.strip()))
            person_id = cur.lastrowid
    invalidate_cache()
    return person_id

def update_person(person_id: int, name: str, notes: str = ""):
    with get_db() as conn:
        with conn:
            conn.execute("UPDATE persons SET name = ?, notes = ? WHERE id = ?", (name.strip(), notes.strip(), person_id))
    invalidate_cache()

def add_face_sample(person_id: int, image_path: str, crop_path: str, embedding: np.ndarray, confidence: float = 1.0) -> int:
    embedding_bytes = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
    with get_db() as conn:
        with conn:
            cur = conn.execute(
                """INSERT INTO face_samples (person_id, image_path, crop_path, embedding, confidence)
                   VALUES (?, ?, ?, ?, ?)""",
                (person_id, image_path, crop_path, embedding_bytes, float(confidence))
            )
            sample_id = cur.lastrowid
    invalidate_cache()
    return sample_id

def get_all_persons():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT p.id, p.name, p.notes, p.created_at,
                   COUNT(s.id) as sample_count,
                   MAX(s.created_at) as last_sample_at
            FROM persons p
            LEFT JOIN face_samples s ON p.id = s.person_id
            GROUP BY p.id
            ORDER BY p.name COLLATE NOCASE ASC
        """).fetchall()
        return [dict(r) for r in rows]

def get_person(person_id: int):
    with get_db() as conn:
        row = conn.execute("""
            SELECT p.id, p.name, p.notes, p.created_at,
                   COUNT(s.id) as sample_count
            FROM persons p
            LEFT JOIN face_samples s ON p.id = s.person_id
            WHERE p.id = ?
            GROUP BY p.id
        """, (person_id,)).fetchone()
        return dict(row) if row else None

def get_person_samples(person_id: int):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT id, person_id, image_path, crop_path, confidence, created_at
            FROM face_samples
            WHERE person_id = ?
            ORDER BY created_at DESC
        """, (person_id,)).fetchall()
        return [dict(r) for r in rows]

def delete_person(person_id: int):
    with get_db() as conn:
        samples = conn.execute("SELECT image_path, crop_path FROM face_samples WHERE person_id = ?", (person_id,)).fetchall()
        for s in samples:
            crop_p = s["crop_path"]
            if crop_p and os.path.exists(crop_p):
                try:
                    os.remove(crop_p)
                except Exception:
                    pass
            # Protect shared original images from accidental deletion if other persons use them
            img_p = s["image_path"]
            if img_p and os.path.exists(img_p):
                other_usage = conn.execute(
                    "SELECT COUNT(*) FROM face_samples WHERE image_path = ? AND person_id != ?",
                    (img_p, person_id)
                ).fetchone()[0]
                if other_usage == 0:
                    try:
                        os.remove(img_p)
                    except Exception:
                        pass
        with conn:
            conn.execute("DELETE FROM persons WHERE id = ?", (person_id,))
    invalidate_cache()

def delete_sample(sample_id: int):
    with get_db() as conn:
        row = conn.execute("SELECT image_path, crop_path FROM face_samples WHERE id = ?", (sample_id,)).fetchone()
        if row:
            crop_p = row["crop_path"]
            if crop_p and os.path.exists(crop_p):
                try:
                    os.remove(crop_p)
                except Exception:
                    pass
            img_p = row["image_path"]
            if img_p and os.path.exists(img_p):
                other_usage = conn.execute(
                    "SELECT COUNT(*) FROM face_samples WHERE image_path = ? AND id != ?",
                    (img_p, sample_id)
                ).fetchone()[0]
                if other_usage == 0:
                    try:
                        os.remove(img_p)
                    except Exception:
                        pass
        with conn:
            conn.execute("DELETE FROM face_samples WHERE id = ?", (sample_id,))
    invalidate_cache()

def get_all_embeddings():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT s.id as sample_id, s.person_id, p.name as person_name,
                   s.embedding, s.crop_path, s.confidence
            FROM face_samples s
            JOIN persons p ON s.person_id = p.id
        """).fetchall()
        
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

def get_cached_embeddings():
    """Returns in-memory cached embeddings list, reloading only if the database changed."""
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = get_all_embeddings()
    return _cached_embeddings

def log_detection_event(person_name: str, similarity: float, source_label: str, crop_path: str = "", snapshot_path: str = "") -> int:
    """Inserts a detection event into the event log."""
    with get_db() as conn:
        with conn:
            cur = conn.execute(
                "INSERT INTO detection_events (person_name, similarity, source_label, crop_path, snapshot_path) VALUES (?, ?, ?, ?, ?)",
                (person_name.strip(), float(similarity), source_label.strip(), crop_path.strip(), snapshot_path.strip())
            )
            return cur.lastrowid

def get_detection_events(limit: int = 300, name_filter: str = ""):
    """Fetches recent detection events, optionally filtered by person name."""
    with get_db() as conn:
        if name_filter and name_filter.strip():
            query = """
                SELECT id, person_name, similarity, source_label, crop_path, snapshot_path,
                       datetime(created_at, 'localtime') as local_time
                FROM detection_events
                WHERE LOWER(person_name) LIKE LOWER(?)
                ORDER BY id DESC LIMIT ?
            """
            rows = conn.execute(query, (f"%{name_filter.strip()}%", limit)).fetchall()
        else:
            query = """
                SELECT id, person_name, similarity, source_label, crop_path, snapshot_path,
                       datetime(created_at, 'localtime') as local_time
                FROM detection_events
                ORDER BY id DESC LIMIT ?
            """
            rows = conn.execute(query, (limit,)).fetchall()
            
        return [dict(r) for r in rows]

def clear_detection_events():
    """Clears all detection events and optionally cleans up event crop and snapshot files."""
    with get_db() as conn:
        with conn:
            rows = conn.execute("SELECT crop_path, snapshot_path FROM detection_events").fetchall()
            for r in rows:
                for col in ("crop_path", "snapshot_path"):
                    cp = r[col]
                    if cp and os.path.exists(cp):
                        try:
                            os.remove(cp)
                        except Exception:
                            pass
            conn.execute("DELETE FROM detection_events;")

def get_detection_stats():
    """Returns total count and unique persons count from detection events."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM detection_events").fetchone()[0]
        unique_p = conn.execute("SELECT COUNT(DISTINCT person_name) FROM detection_events").fetchone()[0]
        latest_row = conn.execute("SELECT person_name, datetime(created_at, 'localtime') as lt FROM detection_events ORDER BY id DESC LIMIT 1").fetchone()
        latest = f"{latest_row['person_name']} ({latest_row['lt']})" if latest_row else "Nema zabilježenih prolazaka"
        return {
            "total_events": total,
            "unique_persons": unique_p,
            "latest_event": latest
        }

def get_stats():
    with get_db() as conn:
        person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        sample_count = conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]
        return {
            "total_persons": person_count,
            "total_samples": sample_count
        }

init_db()
