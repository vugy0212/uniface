import os
import shutil
import zipfile
import datetime
import db

def export_database_zip(data_dir: str, export_dir: str = None) -> str:
    """
    Creates a full backup ZIP of the UniFace database, crops, and uploads.
    Returns the path to the created ZIP file for download.
    """
    if export_dir is None:
        export_dir = os.path.join(data_dir, "backups")
    os.makedirs(export_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"uniface_backup_{timestamp}.zip"
    zip_path = os.path.join(export_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Database
        db_path = os.path.join(data_dir, "database.db")
        if os.path.exists(db_path):
            zf.write(db_path, "database.db")
            
        # 2. Crops
        crops_dir = os.path.join(data_dir, "crops")
        if os.path.exists(crops_dir):
            for root, _, files in os.walk(crops_dir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, data_dir)
                    zf.write(full, rel)
                    
        # 3. Uploads
        uploads_dir = os.path.join(data_dir, "uploads")
        if os.path.exists(uploads_dir):
            for root, _, files in os.walk(uploads_dir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, data_dir)
                    zf.write(full, rel)
                    
    return zip_path

def import_database_zip(zip_file_input, data_dir: str) -> tuple[bool, str]:
    """
    Safely restores database and image files from a backup ZIP.
    Creates a pre-restore backup copy before applying any changes.
    """
    if zip_file_input is None:
        return False, "⚠️ Niste odabrali ZIP datoteku za uvoz."
        
    path = getattr(zip_file_input, "name", None) or (zip_file_input if isinstance(zip_file_input, str) else None)
    if not path or not os.path.exists(path):
        return False, "❌ Datoteka sigurnosne kopije ne postoji ili je nedostupna."
        
    db_file = os.path.join(data_dir, "database.db")
    bak_file = os.path.join(data_dir, "database_pre_restore.bak")
    
    try:
        # Pre-check archive validity
        with zipfile.ZipFile(path, "r") as zf:
            namelist = zf.namelist()
            if "database.db" not in namelist:
                return False, "❌ Neispravan format arhive: 'database.db' nije pronađen u ZIP datoteci."
                
            # Create safety backup of current database
            if os.path.exists(db_file):
                shutil.copy2(db_file, bak_file)
                
            # Extract database.db and image folders
            for item in namelist:
                # Security: prevent directory traversal (Zip Slip)
                norm = os.path.normpath(item)
                if norm.startswith("..") or os.path.isabs(norm):
                    continue
                    
                target = os.path.join(data_dir, norm)
                if item.endswith("/"):
                    os.makedirs(target, exist_ok=True)
                else:
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    with zf.open(item) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                        
        # Ensure schema integrity and flush in-memory caches
        db.init_db()
        db.invalidate_cache()
        stats = db.get_stats()
        
        # Clean up temporary backup on success
        if os.path.exists(bak_file):
            try:
                os.remove(bak_file)
            except Exception:
                pass
                
        return True, f"✅ Uspješan uvoz sigurnosne kopije! Baza sada sadrži **{stats['total_persons']}** osoba i **{stats['total_samples']}** biometrijskih uzoraka."
    except Exception as e:
        # Rollback database from backup if failure occurred
        if os.path.exists(bak_file):
            try:
                shutil.copy2(bak_file, db_file)
                os.remove(bak_file)
            except Exception:
                pass
        return False, f"❌ Greška pri uvozu sigurnosne kopije: {e}"
