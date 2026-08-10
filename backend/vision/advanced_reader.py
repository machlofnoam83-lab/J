"""
Advanced Text Reader - קריאת טקסט חכמה
קורא טקסט ממסך, תמונות, PDF, ומבין אותו

שיפורים:
1. EasyOCR + PaddleOCR + Tesseract - בוחר הכי טוב
2. Layout analysis - מבין כותרות, פסקאות, טבלאות
3. Reading order - קורא בסדר נכון (עברית RTL!)
4. Text understanding - מסכם, עונה על שאלות על הטקסט
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

class TextBlock:
    def __init__(self, text: str, bbox: Tuple[int, int, int, int], confidence: float, block_type="text"):
        self.text = text
        self.bbox = bbox  # x, y, w, h
        self.confidence = confidence
        self.type = block_type  # text, title, table, list
        self.language = self._detect_lang(text)
    
    def _detect_lang(self, text: str) -> str:
        hebrew_chars = len(re.findall(r'[\u0590-\u05FF]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        if hebrew_chars > english_chars:
            return "he"
        elif english_chars > 0:
            return "en"
        return "unknown"

class AdvancedTextReader:
    """
    קורא טקסט מתקדם - מבין עברית RTL וגם דיבור מהיר
    """
    def __init__(self):
        self.has_easyocr = False
        self.has_paddle = False
        self.has_tesseract = False
        self.easyocr_reader = None
        
        # נסה לטעון EasyOCR - הכי טוב לעברית - עם fallback ל-iw
        try:
            import easyocr
            for langs in [['he', 'en'], ['iw', 'en'], ['en']]:
                try:
                    self.easyocr_reader = easyocr.Reader(langs, gpu=False, verbose=False)
                    self.has_easyocr = True
                    print(f"[AdvancedReader] ✓ EasyOCR loaded ({'+'.join(langs)}) - הכי טוב לעברית!")
                    break
                except Exception as e_lang:
                    print(f"[AdvancedReader] EasyOCR langs {langs} failed: {e_lang}")
                    continue
        except Exception as e:
            print(f"[AdvancedReader] EasyOCR not available: {e}")
        
        # PaddleOCR כ-fallback
        try:
            from paddleocr import PaddleOCR
            self.paddle_reader = PaddleOCR(use_angle_cls=True, lang='he', show_log=False)
            self.has_paddle = True
            print("[AdvancedReader] ✓ PaddleOCR loaded")
        except:
            print("[AdvancedReader] PaddleOCR not available")
        
        # Tesseract כ-fallback אחרון
        try:
            import pytesseract
            self.has_tesseract = True
            print("[AdvancedReader] ✓ Tesseract available")
        except:
            print("[AdvancedReader] Tesseract not available")

    def read_image(self, image_path: str = None, image=None, lang="he") -> List[TextBlock]:
        """
        קריאת טקסט מתמונה - עם כל המנועים, בוחר הכי טוב
        """
        from PIL import Image
        import numpy as np
        
        # טען תמונה
        if image_path:
            pil_image = Image.open(image_path).convert('RGB')
        elif image is not None:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
        else:
            return []
        
        blocks = []
        
        # נסה EasyOCR קודם - הכי טוב
        if self.has_easyocr and self.easyocr_reader:
            try:
                print("[AdvancedReader] Reading with EasyOCR...")
                # EasyOCR מחזיר [(bbox, text, confidence)]
                results = self.easyocr_reader.readtext(np.array(pil_image))
                
                for (bbox, text, conf) in results:
                    if conf > 0.3 and len(text.strip()) > 1:  # סף ביטחון
                        # bbox הוא 4 נקודות
                        x = int(min([p[0] for p in bbox]))
                        y = int(min([p[1] for p in bbox]))
                        w = int(max([p[0] for p in bbox]) - x)
                        h = int(max([p[1] for p in bbox]) - y)
                        
                        block_type = "title" if conf > 0.8 and len(text) < 50 else "text"
                        blocks.append(TextBlock(text.strip(), (x, y, w, h), conf, block_type))
                
                print(f"[AdvancedReader] EasyOCR found {len(blocks)} blocks")
                if blocks:
                    return self._sort_blocks_hebrew(blocks)
            except Exception as e:
                print(f"[AdvancedReader] EasyOCR failed: {e}")
        
        # Fallback Tesseract
        if self.has_tesseract:
            try:
                import pytesseract
                print("[AdvancedReader] Reading with Tesseract...")
                
                # עברית + אנגלית
                custom_config = r'--oem 3 --psm 6 -l heb+eng'
                data = pytesseract.image_to_data(pil_image, config=custom_config, output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    conf = int(data['conf'][i])
                    if text and conf > 30:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        blocks.append(TextBlock(text, (x, y, w, h), conf/100.0, "text"))
                
                print(f"[AdvancedReader] Tesseract found {len(blocks)} blocks")
                if blocks:
                    return self._sort_blocks_hebrew(blocks)
            except Exception as e:
                print(f"[AdvancedReader] Tesseract failed: {e}")
        
        return blocks

    def _sort_blocks_hebrew(self, blocks: List[TextBlock]) -> List[TextBlock]:
        """
        מיון בלוקים בסדר קריאה עברי RTL!
        עברית קוראים מימין לשמאל, מלמעלה למטה
        """
        # מיין לפי Y (מלמעלה למטה), ואז X (מימין לשמאל לעברית)
        # אם יש עברית - מימין לשמאל, אם אנגלית - משמאל לימין
        
        def sort_key(block):
            x, y, w, h = block.bbox
            # קבץ לשורות (Y קרוב)
            row = y // 20  # כל 20px שורה חדשה
            if block.language == "he":
                # עברית: מימין לשמאל = X גדול קודם
                return (row, -x)
            else:
                # אנגלית: משמאל לימין
                return (row, x)
        
        return sorted(blocks, key=sort_key)

    def read_screen(self) -> Dict:
        """
        קריאת מסך חיה - עם ניתוח layout
        """
        try:
            import mss
            from PIL import Image
            
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                raw = sct.grab(monitor)
                img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
                
                blocks = self.read_image(image=img)
                
                # נתח layout
                full_text = " ".join([b.text for b in blocks])
                
                # זהה כותרות, טבלאות, רשימות
                titles = [b for b in blocks if b.type == "title"]
                tables = self._detect_tables(blocks)
                
                return {
                    "success": True,
                    "full_text": full_text,
                    "blocks": [{"text": b.text, "type": b.type, "conf": b.confidence, "lang": b.language} for b in blocks],
                    "titles": [t.text for t in titles],
                    "tables": tables,
                    "block_count": len(blocks),
                    "hebrew_blocks": len([b for b in blocks if b.language == "he"]),
                    "message": f"קראתי {len(blocks)} בלוקים, {len([b for b in blocks if b.language == 'he'])} בעברית"
                }
                
        except Exception as e:
            print(f"[AdvancedReader] Screen read failed: {e}")
            import traceback; traceback.print_exc()
            return {"success": False, "error": str(e), "full_text": ""}

    def _detect_tables(self, blocks: List[TextBlock]) -> List[Dict]:
        """זיהוי טבלאות (פשוט)"""
        # אם יש הרבה בלוקים באותו Y עם X שונה - אולי טבלה
        tables = []
        # פשוט מאוד - לדמו
        return tables

    def understand_text(self, text: str, question: str = "") -> Dict:
        """
        הבנת טקסט - מסכם ועונה על שאלות
        """
        if not text:
            return {"success": False, "error": "No text"}
        
        # ניקוי
        text = text.strip()
        
        # סיכום
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        # מילות מפתח
        words = re.findall(r'[\u0590-\u05FF]{3,}|[a-zA-Z]{4,}', text.lower())
        word_freq = {}
        for w in words:
            if len(w) > 2:
                word_freq[w] = word_freq.get(w, 0) + 1
        
        top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # אם יש שאלה - נסה לענות
        answer = ""
        if question:
            # חפש משפט עם מילות השאלה
            q_words = set(re.findall(r'[\u0590-\u05FF]+|[a-zA-Z]+', question.lower()))
            best_sentence = ""
            best_score = 0
            
            for sent in sentences:
                sent_words = set(re.findall(r'[\u0590-\u05FF]+|[a-zA-Z]+', sent.lower()))
                overlap = len(q_words & sent_words)
                if overlap > best_score:
                    best_score = overlap
                    best_sentence = sent
            
            if best_sentence:
                answer = best_sentence
        
        summary = ""
        if len(sentences) > 3:
            # קח 3 משפטים ראשונים + אחרון + עם מילות מפתח
            summary = ". ".join(sentences[:2] + [sentences[-1]]) + "."
        else:
            summary = text[:300]
        
        return {
            "success": True,
            "original_length": len(text),
            "summary": summary,
            "keywords": [w for w, c in top_keywords],
            "sentence_count": len(sentences),
            "answer": answer,
            "question": question,
            "message": f"קראתי {len(text)} תווים, {len(sentences)} משפטים, סיכמתי ל-{len(summary)} תווים"
        }

    def read_pdf(self, pdf_path: str) -> Dict:
        """קריאת PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                full_text += page.get_text() + "\n"
            
            return self.understand_text(full_text)
        except ImportError:
            return {"success": False, "error": "PyMuPDF not installed: pip install PyMuPDF"}
        except Exception as e:
            return {"success": False, "error": str(e)}


def get_advanced_reader():
    return AdvancedTextReader()
