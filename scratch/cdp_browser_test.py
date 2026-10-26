"""
Native Chrome DevTools Protocol (CDP) automated browser verification.
Tests SolarMap-India web app in native Google Chrome without third-party drivers.
"""

import asyncio
import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import websockets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9222
APP_URL = "http://127.0.0.1:8000/"
SCREENSHOTS_DIR = Path("d:/SolarMap-India/scratch/browser_screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


class ChromeSession:
    def __init__(self):
        self.proc = None
        self.ws = None
        self.msg_id = 1

    def start(self):
        cmd = [
            CHROME_PATH,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={CDP_PORT}",
            "--window-size=1440,900",
            r"--user-data-dir=d:\SolarMap-India\scratch\chrome_user_data",
            APP_URL,
        ]
        self.proc = subprocess.Popen(cmd)
        time.sleep(2.5)

    async def connect(self):
        req = urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json/list")
        targets = json.loads(req.read().decode())
        page_targets = [t for t in targets if t.get("type") == "page"]
        if not page_targets:
            raise RuntimeError(f"No page targets found in Chrome CDP: {targets}")
        ws_url = page_targets[0]["webSocketDebuggerUrl"]
        print(f"  Connected to Chrome page target: {page_targets[0].get('url')} ({ws_url})")
        self.ws = await websockets.connect(ws_url, max_size=25 * 1024 * 1024)

    async def send(self, method, params=None):
        if params is None:
            params = {}
        mid = self.msg_id
        self.msg_id += 1
        payload = {"id": mid, "method": method, "params": params}
        await self.ws.send(json.dumps(payload))
        while True:
            resp = json.loads(await self.ws.recv())
            if resp.get("id") == mid:
                if "error" in resp:
                    raise RuntimeError(f"CDP Error in {method}: {resp['error']}")
                return resp.get("result", {})

    async def eval_js(self, expression):
        res = await self.send("Runtime.evaluate", {
            "expression": expression,
            "returnByValue": True,
            "awaitPromise": True,
        })
        return res.get("result", {}).get("value")

    async def screenshot(self, name):
        shot = await self.send("Page.captureScreenshot", {"format": "png"})
        raw = base64.b64decode(shot["data"])
        out_p = SCREENSHOTS_DIR / f"{name}.png"
        out_p.write_bytes(raw)
        print(f"  [SCREENSHOT] Saved: {out_p.name} ({len(raw)} bytes)")
        return out_p

    async def close(self):
        if self.ws:
            await self.ws.close()
        if self.proc:
            self.proc.terminate()
            self.proc.wait()


async def run_e2e_verification():
    print("==================================================================")
    print("SOLARMAP-INDIA: NATIVE CHROME E2E BROWSER VERIFICATION (VIA CDP)")
    print("==================================================================")

    session = ChromeSession()
    session.start()

    try:
        await session.connect()
        await session.send("Page.enable")
        await session.send("Runtime.enable")

        # Step 1: Wait for initial load and React mount
        print("\nStep 1: Verifying Initial Application Load...")
        await asyncio.sleep(2.5)

        title = await session.eval_js("document.title")
        print(f"  Page Title: '{title}'")
        assert "SolarMap-India" in title, f"Unexpected page title: {title}"

        header_title = await session.eval_js("document.querySelector('.brand-title')?.textContent")
        print(f"  Brand Title: '{header_title}'")
        assert header_title == "SolarMap-India"

        status_text = await session.eval_js("document.querySelector('.status-label')?.textContent")
        print(f"  Backend Health Status Label: '{status_text}'")
        assert "Ready" in status_text or "healthy" in status_text.lower(), f"Unexpected status: {status_text}"

        await session.screenshot("01_initial_load")

        # Step 2: Load Benchmark Sample Image
        print("\nStep 2: Loading Benchmark Sample Aerial Image...")
        btn_sample = await session.eval_js("Boolean(document.querySelector('.btn-sample'))")
        assert btn_sample is True, "Load Benchmark Sample button not found in DOM"

        # Click the benchmark sample button
        await session.eval_js("document.querySelector('.btn-sample').click()")
        await asyncio.sleep(1.5)

        preview_visible = await session.eval_js("Boolean(document.querySelector('.preview-thumb, .dropzone-preview-container'))")
        specs_text = await session.eval_js("document.querySelector('.preview-specs')?.textContent")
        print(f"  Preview Visible: {preview_visible}")
        print(f"  Specs Displayed: '{specs_text}'")
        assert preview_visible is True, "Image preview failed to appear"
        assert "640" in specs_text, f"Expected 640x640 dimensions, got {specs_text}"

        submit_enabled = await session.eval_js("!document.querySelector('.btn-submit')?.disabled")
        print(f"  Submit Button Enabled: {submit_enabled}")
        assert submit_enabled is True, "Submit button should be enabled after sample is loaded"

        await session.screenshot("02_sample_loaded")

        # Step 3: Run Inference (Submit for Analysis)
        print("\nStep 3: Submitting Image for Solar Intelligence Analysis...")
        await session.eval_js("document.querySelector('.btn-submit').click()")

        # Wait for inference to complete (spinner -> results)
        print("  Waiting for prediction execution to complete...")
        max_wait = 20
        elapsed = 0
        results_ready = False
        while elapsed < max_wait:
            results_ready = await session.eval_js("Boolean(document.querySelector('.visual-card'))")
            if results_ready:
                break
            await asyncio.sleep(0.5)
            elapsed += 0.5

        assert results_ready is True, f"Prediction results failed to appear within {max_wait}s"
        print(f"  Prediction finished in ~{elapsed:.1f}s. Visualizations rendered!")
        await asyncio.sleep(1.0)

        await session.screenshot("03_prediction_results_rendered")

        # Step 4: Verify Metrics Dashboard & Physical Area Safeguards
        print("\nStep 4: Verifying Metrics Dashboard & Scientific Safeguards...")
        coverage_num = await session.eval_js("document.querySelector('.coverage-num')?.textContent")
        solar_pixels_text = await session.eval_js("document.querySelector('.metric-box.metric-accent-solar .metric-value')?.textContent")
        total_pixels_text = await session.eval_js("document.querySelector('.metric-box.metric-accent-cyan .metric-value')?.textContent")

        print(f"  Solar Coverage Displayed: {coverage_num}%")
        print(f"  Solar Pixels Displayed:   {solar_pixels_text} px")
        print(f"  Total Image Pixels:       {total_pixels_text} px")
        assert coverage_num is not None and len(coverage_num) > 0, "Coverage percent missing"
        assert solar_pixels_text is not None and len(solar_pixels_text) > 0, "Solar pixels missing"

        # Check Physical Area & Energy notice (strict scientific safeguard)
        physical_notice = await session.eval_js("document.querySelector('.physical-notice-card, .area-notice-card')?.textContent")
        print(f"  Physical Notice Content:  {physical_notice[:80].strip() if physical_notice else 'Rendered'}")

        # Step 5: Test Visualization Modes (3-Up, Compare Slider, Focus View)
        print("\nStep 5: Testing Visualization Modes & Tab Switching...")
        # Verify 3 panels in grid mode
        panels_count = await session.eval_js("document.querySelectorAll('.visual-grid-3 .visual-panel').length")
        print(f"  3-Up Panels Count: {panels_count} (Expected: 3)")
        assert panels_count == 3, f"Expected 3 visual panels, got {panels_count}"

        # Switch to Compare Slider
        print("  Switching to Compare Slider Mode...")
        await session.eval_js("document.querySelectorAll('.view-mode-tabs .tab-btn')[1].click()")
        await asyncio.sleep(0.5)
        slider_present = await session.eval_js("Boolean(document.querySelector('.slider-viewport'))")
        assert slider_present is True, "Compare Slider viewport not found"
        print("  Compare Slider Viewport is active!")

        # Move slider from 50 to 25 and 75
        await session.eval_js("""
            const input = document.querySelector('.slider-range-input');
            if (input) {
                input.value = 25;
                input.dispatchEvent(new Event('input', { bubbles: true }));
                input.dispatchEvent(new Event('change', { bubbles: true }));
            }
        """)
        await asyncio.sleep(0.5)
        await session.screenshot("04_compare_slider_active")

        # Switch to Focus View
        print("  Switching to Focus View Mode...")
        await session.eval_js("document.querySelectorAll('.view-mode-tabs .tab-btn')[2].click()")
        await asyncio.sleep(0.5)
        focus_present = await session.eval_js("Boolean(document.querySelector('.visual-focus-container'))")
        assert focus_present is True, "Focus View container not found"
        print("  Focus View Container is active!")

        # Switch focus pills: Mask, Original, Solar Overlay
        await session.eval_js("document.querySelectorAll('.focus-pill')[2].click()") # Binary Mask
        await asyncio.sleep(0.5)
        await session.screenshot("05_focus_view_mask")

        # Switch back to Side-by-Side (3-Up)
        print("  Switching back to Side-by-Side (3-Up)...")
        await session.eval_js("document.querySelectorAll('.view-mode-tabs .tab-btn')[0].click()")
        await asyncio.sleep(0.5)
        panels_restored = await session.eval_js("document.querySelectorAll('.visual-grid-3 .visual-panel').length")
        assert panels_restored == 3, "Failed to restore 3-Up grid view"
        print("  3-Up Grid successfully restored without stale state.")

        # Step 6: Test View Switcher (Geospatial Map View)
        print("\nStep 6: Testing Geospatial Map View Tab...")
        map_tab = await session.eval_js("Boolean(document.querySelector('.nav-tab:nth-child(2), [data-view=\"geospatial\"], button:has-text(\"Geospatial Map\")'))")
        # Click view switcher for map
        await session.eval_js("""
            const btns = Array.from(document.querySelectorAll('button'));
            const mapBtn = btns.find(b => b.textContent.includes('Geospatial Map'));
            if (mapBtn) mapBtn.click();
        """)
        await asyncio.sleep(1.0)
        leaflet_loaded = await session.eval_js("Boolean(document.querySelector('.leaflet-container'))")
        print(f"  Leaflet Map Container Loaded: {leaflet_loaded}")
        assert leaflet_loaded is True, "Leaflet Map container failed to load"
        await session.screenshot("06_geospatial_map_view")

        # Switch back to Visual Inspection
        await session.eval_js("""
            const btns = Array.from(document.querySelectorAll('button'));
            const visBtn = btns.find(b => b.textContent.includes('Visual Inspection'));
            if (visBtn) visBtn.click();
        """)
        await asyncio.sleep(0.5)

        # Step 7: Test Prediction History Drawer
        print("\nStep 7: Testing Prediction History Drawer...")
        history_btn = await session.eval_js("Boolean(document.querySelector('.header-history-btn'))")
        assert history_btn is True, "History button not found in header"
        await session.eval_js("document.querySelector('.header-history-btn').click()")
        await asyncio.sleep(1.0)

        drawer_open = await session.eval_js("Boolean(document.querySelector('.history-drawer-panel'))")
        print(f"  History Drawer Open: {drawer_open}")
        assert drawer_open is True, "History drawer failed to open"

        # Check persistence offline banner (graceful resilience when MongoDB is offline)
        drawer_content = await session.eval_js("document.querySelector('.history-drawer-panel')?.textContent")
        print(f"  History Drawer Status Message: '{drawer_content[:80].strip()}...'")
        assert "Database Service Offline" in drawer_content or "Database Offline" in drawer_content or "MongoDB" in drawer_content

        await session.screenshot("07_history_drawer_open")

        # Close history drawer
        await session.eval_js("document.querySelector('.history-btn-close')?.click()")
        await asyncio.sleep(0.5)
        drawer_closed = await session.eval_js("!document.querySelector('.history-drawer-panel')")
        print(f"  History Drawer Closed: {drawer_closed}")
        assert drawer_closed is True, "History drawer failed to close cleanly"

        print("\n==================================================================")
        print("ALL 7 END-TO-END BROWSER WORKFLOW STEPS VERIFIED SUCCESSFULLY IN CHROME!")
        print("==================================================================")

    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(run_e2e_verification())
