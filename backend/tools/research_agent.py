"""
Research Agent - מחקר ואיסוף מידע
סורק עשרות אתרים במקביל ומסכם לקובץ אחד
"""
import asyncio
import re
from typing import List, Dict
from pathlib import Path
from datetime import datetime
from .browser_agent import get_browser_agent

class ResearchAgent:
    """
    סוכן מחקר - סורק, מסנן, מסכם
    """
    def __init__(self):
        self.browser = get_browser_agent(headless=True)
    
    async def research_topic(self, topic: str, depth=10, language="he") -> Dict:
        """
        מחקר מעמיק על נושא
        depth = כמה אתרים לסרוק
        """
        print(f"[Research] 🔬 חוקר נושא: '{topic}' בעומק {depth} אתרים")
        
        # 1. חיפוש ראשוני
        search_queries = [
            topic,
            f"{topic} מדריך",
            f"{topic} הסבר",
            f"{topic} 2024",
            f"{topic} best practices"
        ]
        
        all_urls = []
        for query in search_queries[:3]:
            results = await self.browser.search_google(query, num_results=5)
            all_urls.extend([r["url"] for r in results if r.get("url")])
        
        # הסר כפילויות
        unique_urls = list(dict.fromkeys(all_urls))[:depth]
        print(f"[Research] נמצאו {len(unique_urls)} אתרים לסריקה")
        
        # 2. סריקה מקבילה
        tasks = [self._scrape_and_summarize(url, topic) for url in unique_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, dict) and r.get("success")]
        print(f"[Research] נסרקו בהצלחה {len(valid_results)}/{len(unique_urls)} אתרים")
        
        # 3. סיכום
        summary = self._create_summary(topic, valid_results)
        
        # 4. שמירה לקובץ
        file_path = await self._save_research(topic, summary, valid_results)
        
        return {
            "success": True,
            "topic": topic,
            "scanned_sites": len(unique_urls),
            "successful_scrapes": len(valid_results),
            "summary": summary,
            "sources": [{"url": r["url"], "title": r["title"], "key_points": r["key_points"][:3]} for r in valid_results],
            "file_path": file_path,
            "file_url": f"file://{file_path}",
            "message": f"סיימתי מחקר על '{topic}'! סרקתי {len(valid_results)} אתרים, סיכמתי ל-{len(summary)} תווים ושמרתי ב-{file_path}"
        }

    async def _scrape_and_summarize(self, url: str, topic: str) -> Dict:
        """גורד ומסכם אתר אחד"""
        try:
            result = await self.browser.navigate(url)
            if not result.success:
                return {"success": False, "url": url, "error": result.error}
            
            text = result.content or await self.browser.extract_text()
            
            # נקה טקסט
            text = re.sub(r'\s+', ' ', text)[:8000]
            
            # חלץ נקודות מפתח רלוונטיות לנושא
            key_points = self._extract_key_points(text, topic)
            
            return {
                "success": True,
                "url": url,
                "title": result.title,
                "content": text[:2000],
                "key_points": key_points,
                "relevance_score": len(key_points) * 10
            }
        except Exception as e:
            return {"success": False, "url": url, "error": str(e)}

    def _extract_key_points(self, text: str, topic: str) -> List[str]:
        """חילוץ נקודות מפתח"""
        sentences = re.split(r'[.!?]\s+', text)
        topic_words = set(re.findall(r'[\u0590-\u05FF]+|[a-zA-Z]+', topic.lower()))
        
        scored = []
        for sent in sentences:
            if len(sent) < 20 or len(sent) > 300:
                continue
            words = set(re.findall(r'[\u0590-\u05FF]+|[a-zA-Z]+', sent.lower()))
            overlap = len(words & topic_words)
            if overlap > 0 or any(w in sent.lower() for w in topic.lower().split()):
                scored.append((overlap, sent.strip()))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for score, s in scored[:5]]

    def _create_summary(self, topic: str, results: List[Dict]) -> str:
        """יצירת סיכום מאוחד"""
        if not results:
            return f"לא נמצא מידע על {topic}"
        
        summary_parts = [f"# מחקר: {topic}", f"תאריך: {datetime.now().strftime('%d/%m/%Y %H:%M')}", f"מקורות: {len(results)} אתרים", ""]
        
        # אסוף כל נקודות המפתח
        all_points = []
        for r in results:
            all_points.extend(r.get("key_points", []))
        
        # הסר כפילויות
        unique_points = list(dict.fromkeys(all_points))[:15]
        
        summary_parts.append("## נקודות עיקריות:")
        for i, point in enumerate(unique_points, 1):
            summary_parts.append(f"{i}. {point}")
        
        summary_parts.append("")
        summary_parts.append("## מקורות:")
        for r in results:
            summary_parts.append(f"- {r['title']}: {r['url']}")
        
        return "\n".join(summary_parts)

    async def _save_research(self, topic: str, summary: str, results: List[Dict]) -> str:
        """שמירה לקובץ"""
        try:
            safe_topic = re.sub(r'[^\w\u0590-\u05FF\-_ ]', '_', topic)[:30]
            filename = f"research_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            dir_path = Path(__file__).parent.parent / "data" / "research"
            dir_path.mkdir(parents=True, exist_ok=True)
            file_path = dir_path / filename
            
            # הוסף גם raw data
            full_content = summary + "\n\n---\n\n## נתונים מלאים:\n"
            for r in results:
                full_content += f"\n### {r['title']}\nURL: {r['url']}\n{ r['content'][:1000]}\n"
            
            file_path.write_text(full_content, encoding='utf-8')
            print(f"[Research] נשמר ב-{file_path}")
            return str(file_path)
        except Exception as e:
            print(f"[Research] Save failed: {e}")
            return ""

def get_research_agent():
    return ResearchAgent()
