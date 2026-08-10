"""
Browser Agent - ליבה של אוטומציית דפדפן
מנוע גלישה חכם שמבצע משימות מורכבות
"""
import os
import re
import time
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

@dataclass
class BrowseResult:
    success: bool
    url: str
    title: str
    content: str
    data: Dict
    error: Optional[str] = None

class BrowserAgent:
    """
    סוכן דפדפן - יודע לגלוש, לחפש, למלא טפסים, להשוות מחירים
    תומך ב-Playwright (מומלץ) + Selenium + Requests כ-fallback
    """
    def __init__(self, headless=True):
        self.headless = headless
        self.playwright = None
        self.browser = None
        self.page = None
        self.has_playwright = False
        self.has_selenium = False
        
        try:
            from playwright.async_api import async_playwright
            self.has_playwright = True
            print("[BrowserAgent] ✓ Playwright available")
        except:
            print("[BrowserAgent] Playwright not available, trying Selenium")
            try:
                from selenium import webdriver
                self.has_selenium = True
                print("[BrowserAgent] ✓ Selenium available")
            except:
                print("[BrowserAgent] No browser automation lib, using requests")
    
    async def start(self):
        """מתחיל דפדפן - עם auto-install אם חסר"""
        if self.has_playwright:
            try:
                from playwright.async_api import async_playwright
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.launch(headless=self.headless)
                self.page = await self.browser.new_page()
                await self.page.set_extra_http_headers({"Accept-Language": "he-IL,he;q=0.9,en;q=0.8"})
                print("[BrowserAgent] Browser started")
                return True
            except Exception as e:
                err_msg = str(e)
                if "Executable doesn't exist" in err_msg or "playwright install" in err_msg.lower():
                    print(f"[BrowserAgent] Browsers missing, installing... (first time)")
                    try:
                        import subprocess, sys
                        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False, timeout=120)
                        from playwright.async_api import async_playwright
                        self.playwright = await async_playwright().start()
                        self.browser = await self.playwright.chromium.launch(headless=self.headless)
                        self.page = await self.browser.new_page()
                        print("[BrowserAgent] Browser started after install")
                        return True
                    except Exception as e2:
                        print(f"[BrowserAgent] Install failed: {e2}, fallback to requests")
                print(f"[BrowserAgent] Playwright start failed: {e}")
        return False

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def navigate(self, url: str) -> BrowseResult:
        """נווט לאתר"""
        if not self.page:
            await self.start()
        
        if self.page:
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                title = await self.page.title()
                content = await self.page.content()
                return BrowseResult(True, url, title, content[:5000], {}, None)
            except Exception as e:
                return BrowseResult(False, url, "", "", {}, str(e))
        else:
            # Fallback requests
            try:
                import requests
                from bs4 import BeautifulSoup
                resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}, timeout=10)
                soup = BeautifulSoup(resp.text, 'html.parser')
                title = soup.title.string if soup.title else url
                text = soup.get_text()[:5000]
                return BrowseResult(True, url, title, text, {}, None)
            except Exception as e:
                return BrowseResult(False, url, "", "", {}, str(e))

    async def search_google(self, query: str, num_results=5) -> List[Dict]:
        """חיפוש גוגל והחזרת תוצאות"""
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&hl=he"
        result = await self.navigate(url)
        
        if not result.success:
            return []
        
        # אם יש Playwright, נגרד תוצאות
        if self.page:
            try:
                # Google results
                results = []
                elements = await self.page.query_selector_all('div.g')
                for el in elements[:num_results]:
                    try:
                        title_el = await el.query_selector('h3')
                        link_el = await el.query_selector('a')
                        if title_el and link_el:
                            title = await title_el.inner_text()
                            link = await link_el.get_attribute('href')
                            results.append({"title": title, "url": link, "query": query})
                    except:
                        continue
                if results:
                    return results
            except Exception as e:
                print(f"[BrowserAgent] Google scrape failed: {e}")
        
        # Fallback - החזר לפחות את שאילתת החיפוש
        return [{"title": f"תוצאות חיפוש ל-{query}", "url": url, "query": query}]

    async def extract_text(self, selector: str = "body") -> str:
        """חילוץ טקסט מהדף"""
        if self.page:
            try:
                if selector == "body":
                    return await self.page.inner_text('body')
                else:
                    el = await self.page.query_selector(selector)
                    if el:
                        return await el.inner_text()
            except:
                pass
        return ""

    async def fill_form(self, form_data: Dict[str, str]) -> bool:
        """
        מילוי טופס אוטומטי
        form_data = {"selector_or_name": "value", ...}
        לדוגמה: {"#firstName": "נועם", "input[name=email]": "noam@example.com"}
        """
        if not self.page:
            print("[BrowserAgent] No page for form fill")
            return False
        
        try:
            for selector, value in form_data.items():
                try:
                    await self.page.fill(selector, value, timeout=5000)
                    print(f"[BrowserAgent] Filled {selector} = {value}")
                    await asyncio.sleep(0.3)
                except:
                    # נסה לפי label או name
                    try:
                        await self.page.fill(f'[name="{selector}"]', value, timeout=2000)
                    except:
                        print(f"[BrowserAgent] Failed to fill {selector}")
            return True
        except Exception as e:
            print(f"[BrowserAgent] Form fill failed: {e}")
            return False

    async def click(self, selector: str) -> bool:
        if self.page:
            try:
                await self.page.click(selector, timeout=5000)
                return True
            except Exception as e:
                print(f"[BrowserAgent] Click {selector} failed: {e}")
                return False
        return False

    async def screenshot(self, path: str = None) -> Optional[str]:
        """צילום מסך של הדפדפן"""
        if self.page:
            try:
                if not path:
                    path = f"/tmp/browser_{int(time.time())}.png"
                await self.page.screenshot(path=path)
                return path
            except Exception as e:
                print(f"[BrowserAgent] Screenshot failed: {e}")
        return None

# Singleton
_global_browser = None

def get_browser_agent(headless=True) -> BrowserAgent:
    global _global_browser
    if _global_browser is None:
        _global_browser = BrowserAgent(headless=headless)
    return _global_browser
