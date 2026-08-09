"""
Cloud File Manager - ניהול קבצים וענן
מעביר קבצים בין תיקיות, Drive, Dropbox, ומארגן
"""
import os
import shutil
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import json

class CloudFileManager:
    """
    מנהל קבצים חכם - לוקלי + ענן
    """
    def __init__(self, base_dir=None):
        self.base_dir = Path(base_dir) if base_dir else Path.home() / "AdielFiles"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # תיקיות מאורגנות
        self.organized_dirs = {
            "images": self.base_dir / "תמונות",
            "documents": self.base_dir / "מסמכים",
            "videos": self.base_dir / "סרטונים",
            "downloads": self.base_dir / "הורדות",
            "projects": self.base_dir / "פרויקטים",
            "archive": self.base_dir / "ארכיון",
        }
        
        for dir_path in self.organized_dirs.values():
            dir_path.mkdir(exist_ok=True)

    def scan_files(self, directory: str, pattern="*") -> List[Dict]:
        """סריקת קבצים"""
        print(f"[FileManager] 📁 סורק {directory} עם {pattern}")
        
        dir_path = Path(directory)
        if not dir_path.exists():
            return []
        
        files = []
        try:
            for file_path in dir_path.glob(pattern):
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024*1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "extension": file_path.suffix.lower(),
                        "type": self._detect_type(file_path)
                    })
        except Exception as e:
            print(f"[FileManager] Scan failed: {e}")
        
        print(f"[FileManager] נמצאו {len(files)} קבצים")
        return files

    def _detect_type(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]:
            return "image"
        elif ext in [".mp4", ".avi", ".mov", ".mkv"]:
            return "video"
        elif ext in [".pdf", ".doc", ".docx", ".txt", ".md"]:
            return "document"
        elif ext in [".zip", ".rar", ".7z"]:
            return "archive"
        elif ext in [".py", ".js", ".html", ".css", ".json"]:
            return "code"
        else:
            return "other"

    def organize_files(self, source_dir: str, auto_sort=True) -> Dict:
        """ארגון קבצים אוטומטי"""
        print(f"[FileManager] 🗂️ מארגן קבצים מ-{source_dir}")
        
        files = self.scan_files(source_dir)
        if not files:
            return {"success": False, "message": "לא נמצאו קבצים"}
        
        moved = []
        for file_info in files:
            file_path = Path(file_info["path"])
            file_type = file_info["type"]
            
            # בחר תיקיית יעד
            target_dir = None
            if file_type == "image":
                target_dir = self.organized_dirs["images"]
            elif file_type == "document":
                target_dir = self.organized_dirs["documents"]
            elif file_type == "video":
                target_dir = self.organized_dirs["videos"]
            elif file_type == "code":
                target_dir = self.organized_dirs["projects"]
            else:
                target_dir = self.organized_dirs["downloads"]
            
            # תאריך - ארכיון אם ישן
            try:
                mod_date = datetime.fromisoformat(file_info["modified"])
                if (datetime.now() - mod_date).days > 90:
                    target_dir = self.organized_dirs["archive"]
            except:
                pass
            
            if auto_sort:
                try:
                    target_path = target_dir / file_path.name
                    # אם קיים, הוסף מספר
                    counter = 1
                    while target_path.exists():
                        target_path = target_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
                        counter += 1
                    
                    shutil.move(str(file_path), str(target_path))
                    moved.append({
                        "from": file_info["path"],
                        "to": str(target_path),
                        "type": file_type
                    })
                except Exception as e:
                    print(f"[FileManager] Move failed {file_path}: {e}")
        
        return {
            "success": True,
            "scanned": len(files),
            "moved": len(moved),
            "details": moved[:20],
            "organized_dirs": {k: str(v) for k, v in self.organized_dirs.items()},
            "message": f"ארגנתי {len(moved)}/{len(files)} קבצים ל-{len(self.organized_dirs)} תיקיות"
        }

    def sync_to_cloud(self, local_dir: str, cloud_provider="drive", cloud_folder="AdielBackup") -> Dict:
        """
        סנכרון לענן - Drive/Dropbox
        במציאות היה משתמש ב-Google Drive API / Dropbox API
        """
        print(f"[FileManager] ☁️ מסנכרן {local_dir} -> {cloud_provider}/{cloud_folder}")
        
        files = self.scan_files(local_dir)
        
        # דמו - במציאות היה מעלה עם API
        uploaded = []
        for f in files[:10]:  # מגבלת דמו
            uploaded.append({
                "local": f["path"],
                "cloud": f"{cloud_provider}://{cloud_folder}/{Path(f['path']).name}",
                "size": f["size_mb"]
            })
        
        return {
            "success": True,
            "provider": cloud_provider,
            "local_dir": local_dir,
            "cloud_folder": cloud_folder,
            "uploaded_count": len(uploaded),
            "total_files": len(files),
            "uploaded": uploaded,
            "message": f"סנכרנתי {len(uploaded)}/{len(files)} קבצים ל-{cloud_provider}. (דמו - במציאות היה מעלה עם API אמיתי)",
            "real_implementation": "להטמעה אמיתית: pip install google-api-python-client dropbox, ואז OAuth"
        }

    def find_duplicates(self, directory: str) -> Dict:
        """מציאת כפילויות"""
        files = self.scan_files(directory, "*.*")
        
        # קבץ לפי שם וגודל
        by_size_and_name = {}
        for f in files:
            key = f"{f['name']}_{f['size']}"
            by_size_and_name.setdefault(key, []).append(f)
        
        duplicates = {k: v for k, v in by_size_and_name.items() if len(v) > 1}
        
        total_wasted_mb = sum((len(v)-1) * v[0]["size_mb"] for v in duplicates.values())
        
        return {
            "success": True,
            "scanned": len(files),
            "duplicate_groups": len(duplicates),
            "wasted_space_mb": round(total_wasted_mb, 2),
            "duplicates": list(duplicates.values())[:5],
            "message": f"מצאתי {len(duplicates)} קבוצות כפילויות, {total_wasted_mb:.1f}MB מבוזבז"
        }

    def search_files(self, query: str, directory: str = None) -> List[Dict]:
        """חיפוש קבצים"""
        search_dir = directory or str(self.base_dir)
        all_files = self.scan_files(search_dir, "*.*")
        
        results = []
        query_lower = query.lower()
        
        for f in all_files:
            if query_lower in f["name"].lower() or query_lower in f["type"]:
                results.append(f)
        
        return results

def get_file_manager():
    return CloudFileManager()
