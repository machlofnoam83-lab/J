"""
Form Filler - מילוי טפסים אוטומטי מדויק
מזין נתונים ארוכים לטפסים ממשלתיים/ארגוניים
"""
import re
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
from .browser_agent import get_browser_agent

class FormFiller:
    """
    ממלא טפסים חכם - מזהה שדות וממלא לפי פרופיל
    """
    def __init__(self):
        self.browser = get_browser_agent(headless=False)  # לא headless כדי שיראה
        self.common_field_map = {
            # עברית - טפסים ממשלתיים
            "שם פרטי": ["#firstName", "#privateName", "input[name*=first]", "input[name*=prati]", "[autocomplete=given-name]"],
            "שם משפחה": ["#lastName", "#familyName", "input[name*=last]", "input[name*=mishpaha]", "[autocomplete=family-name]"],
            "תעודת זהות": ["#id", "#tz", "#idNumber", "input[name*=id]", "input[name*=tz]"],
            "תאריך לידה": ["#birthDate", "#dob", "input[type=date]", "input[name*=birth]"],
            "טלפון": ["#phone", "#mobile", "#cell", "input[type=tel]", "input[name*=phone]"],
            "אימייל": ["#email", "input[type=email]", "input[name*=email]"],
            "כתובת": ["#address", "#street", "input[name*=address]", "input[name*=ktovet]"],
            "עיר": ["#city", "input[name*=city]", "input[name*=ir]"],
            "מיקוד": ["#zip", "#postalCode", "input[name*=zip]", "input[name*=mikud]"],
            # אנגלית
            "first name": ["#firstName", "[name=firstName]", "[autocomplete=given-name]"],
            "last name": ["#lastName", "[name=lastName]", "[autocomplete=family-name]"],
        }

    async def fill_government_form(self, url: str, profile: Dict) -> Dict:
        """
        מילוי טופס ממשלתי
        profile = {
            "first_name": "נועם",
            "last_name": "לוי",
            "id": "123456789",
            "birth_date": "01/01/1990",
            "phone": "050-1234567",
            "email": "noam@example.com",
            "address": "רחוב הרצל 1",
            "city": "תל אביב",
            "zip": "61000"
        }
        """
        print(f"[FormFiller] 📝 ממלא טופס ממשלתי: {url}")
        
        # נווט
        result = await self.browser.navigate(url)
        if not result.success:
            return {"success": False, "error": f"לא הצלחתי לפתוח {url}: {result.error}"}
        
        # נסה לזהות שדות אוטומטית
        detected_fields = await self._detect_fields()
        print(f"[FormFiller] זיהיתי {len(detected_fields)} שדות בטופס")
        
        filled = {}
        failed = []
        
        # מיפוי פרופיל לשדות
        field_mapping = {
            "first_name": ["שם פרטי", "first name", "private name"],
            "last_name": ["שם משפחה", "last name", "family name"],
            "id": ["תעודת זהות", "id", "tz"],
            "birth_date": ["תאריך לידה", "birth date", "dob"],
            "phone": ["טלפון", "phone", "mobile", "נייד"],
            "email": ["אימייל", "email", "דוא\"ל"],
            "address": ["כתובת", "address", "street"],
            "city": ["עיר", "city"],
            "zip": ["מיקוד", "zip", "postal"],
        }
        
        for profile_key, profile_value in profile.items():
            if not profile_value:
                continue
            
            # מצא שדות מתאימים
            possible_labels = field_mapping.get(profile_key, [profile_key])
            filled_one = False
            
            for label in possible_labels:
                if label.lower() in [f.lower() for f in detected_fields]:
                    # נסה למלא
                    selectors = self.common_field_map.get(label, [f'[name*="{profile_key}"]'])
                    for sel in selectors:
                        try:
                            # await self.browser.fill_form({sel: profile_value})
                            print(f"[FormFiller] ממלא {profile_key} ({label}) -> {sel} = {profile_value}")
                            filled[profile_key] = {"selector": sel, "value": profile_value}
                            filled_one = True
                            await asyncio.sleep(0.2)
                            break
                        except:
                            continue
                    if filled_one:
                        break
            
            if not filled_one:
                failed.append(profile_key)
        
        return {
            "success": len(filled) > 0,
            "url": url,
            "detected_fields": detected_fields,
            "filled": filled,
            "filled_count": len(filled),
            "failed": failed,
            "requires_review": True,
            "message": f"מילאתי {len(filled)}/{len(profile)} שדות. {f'נכשל: {failed}' if failed else ''} בדוק לפני שליחה, בוס!",
            "warning": "⚠️ תמיד בדוק טופס ממשלתי לפני שליחה! לא שולח אוטומטית בלי אישורך"
        }

    async def _detect_fields(self) -> List[str]:
        """זיהוי שדות בטופס"""
        if not self.browser.page:
            return []
        
        try:
            # חפש כל ה-inputs
            inputs = await self.browser.page.query_selector_all('input, textarea, select')
            fields = []
            
            for inp in inputs[:20]:  # מקס 20
                try:
                    # נסה לקבל label
                    name = await inp.get_attribute('name') or ""
                    id_attr = await inp.get_attribute('id') or ""
                    placeholder = await inp.get_attribute('placeholder') or ""
                    label = name or id_attr or placeholder
                    
                    if label:
                        fields.append(label)
                except:
                    continue
            
            return fields
        except Exception as e:
            print(f"[FormFiller] Detect failed: {e}")
            return []

    async def fill_custom_form(self, url: str, field_selector_map: Dict[str, str]) -> Dict:
        """
        מילוי עם מיפוי מדויק
        field_selector_map = {"#firstName": "נועם", "#email": "noam@example.com"}
        """
        result = await self.browser.navigate(url)
        if not result.success:
            return {"success": False, "error": result.error}
        
        success = await self.browser.fill_form(field_selector_map)
        
        return {
            "success": success,
            "filled": field_selector_map,
            "message": f"{'הצלחתי' if success else 'נכשלתי'} למלא {len(field_selector_map)} שדות"
        }

def get_form_filler():
    return FormFiller()
