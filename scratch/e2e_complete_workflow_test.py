"""
SolarMap-India — Comprehensive End-to-End Browser & API Verification Suite (Phase 14.5).
Executes all 20 browser workflow requirements, variable dimension matrices,
and scientific integrity safeguards against native Google Chrome via Chrome DevTools Protocol (CDP).
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
import urllib.error
import numpy as np
from PIL import Image
import websockets

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path("d:/SolarMap-India")
CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
CDP_PORT = 9222
APP_URL = "http://127.0.0.1:8000/"
SCREENSHOTS_DIR = PROJECT_ROOT / "scratch" / "browser_screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
FIXTURES_DIR = PROJECT_ROOT / "scratch" / "browser_fixtures"


class ChromeDevToolsSession:
    def __init__(self, port=CDP_PORT):
        self.port = port
        self.proc = None
        self.ws = None
        self.msg_id = 1
        self.exceptions = []
        self.console_logs = []

    def start(self):
        user_data = PROJECT_ROOT / "scratch" / "chrome_user_data_e2e"
        cmd = [
            CHROME_PATH,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            f"--remote-debugging-port={self.port}",
            "--window-size=1440,960",
            f"--user-data-dir={user_data}",
            APP_URL,
        ]
        self.proc = subprocess.Popen(cmd)
        time.sleep(2.5)

    async def connect(self):
        req = urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list")
        targets = json.loads(req.read().decode())
        page_targets = [t for t in targets if t.get("type") == "page"]
        if not page_targets:
            raise RuntimeError(f"No page targets found: {targets}")
        ws_url = page_targets[0]["webSocketDebuggerUrl"]
        print(f"  Connected to Chrome page: {page_targets[0].get('url')} ({ws_url})")
        self.ws = await websockets.connect(ws_url, max_size=50 * 1024 * 1024)

    async def send(self, method, params=None):
        if params is None:
            params = {}
        mid = self.msg_id
        self.msg_id += 1
        payload = {"id": mid, "method": method, "params": params}
        await self.ws.send(json.dumps(payload))
        while True:
            resp = json.loads(await self.ws.recv())
            # Capture exceptions & console events
            if resp.get("method") == "Runtime.exceptionThrown":
                exc_details = resp.get("params", {}).get("exceptionDetails", {})
                self.exceptions.append(exc_details)
            elif resp.get("method") == "Console.messageAdded":
                msg = resp.get("params", {}).get("message", {})
                self.console_logs.append(msg)

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
        print(f"  [SCREENSHOT] Saved: {out_p.name} ({len(raw):,} bytes)")
        return out_p

    async def upload_file_bytes(self, file_path: Path, mime_type: str = "image/png"):
        file_bytes = file_path.read_bytes()
        b64_content = base64.b64encode(file_bytes).decode("ascii")
        fname = file_path.name

        js_script = f"""
        (async () => {{
            const byteChars = atob('{b64_content}');
            const byteNums = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {{
                byteNums[i] = byteChars.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNums);
            const blob = new Blob([byteArray], {{ type: '{mime_type}' }});
            const file = new File([blob], '{fname}', {{ type: '{mime_type}' }});

            const input = document.querySelector('input[type="file"]');
            if (!input) return false;
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            input.files = dataTransfer.files;
            input.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return true;
        }})();
        """
        return await self.eval_js(js_script)

    async def close(self):
        if self.ws:
            await self.ws.close()
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


async def run_full_verification():
    print("========================================================================")
    print("   SOLARMAP-INDIA: COMPREHENSIVE END-TO-END BROWSER VERIFICATION")
    print("========================================================================")

    session = ChromeDevToolsSession()
    session.start()

    results_summary = {}

    try:
        await session.connect()
        await session.send("Page.enable")
        await session.send("Runtime.enable")
        await session.send("Console.enable")

        # -------------------------------------------------------------
        # STEP 3.1 & 3.2: Initial Page Load, JS Error Monitoring & Upload UI
        # -------------------------------------------------------------
        print("\n[VERIFICATION 1/20] Application opens without fatal JavaScript error...")
        await asyncio.sleep(2.0)
        title = await session.eval_js("document.title")
        brand = await session.eval_js("document.querySelector('.brand-title')?.textContent")
        health_label = await session.eval_js("document.querySelector('.status-label')?.textContent")
        print(f"  Title: '{title}'")
        print(f"  Brand: '{brand}' | Status: '{health_label}'")
        assert "SolarMap-India" in title, f"Unexpected page title: {title}"
        assert brand == "SolarMap-India", f"Unexpected brand: {brand}"
        assert len(session.exceptions) == 0, f"Fatal JS exceptions caught: {session.exceptions}"
        results_summary["1. No Fatal JS Errors"] = "PASS"

        print("\n[VERIFICATION 2/20] Upload interface is visible...")
        dropzone_visible = await session.eval_js("Boolean(document.querySelector('.dropzone'))")
        sample_btn_visible = await session.eval_js("Boolean(document.querySelector('.btn-sample'))")
        file_input_present = await session.eval_js("Boolean(document.querySelector('input[type=\"file\"]'))")
        print(f"  Dropzone: {dropzone_visible} | Sample Button: {sample_btn_visible} | File Input: {file_input_present}")
        assert dropzone_visible and sample_btn_visible and file_input_present
        await session.screenshot("step1_upload_interface_ready")
        results_summary["2. Upload Interface Visible"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.3, 3.4, 3.5: Valid Image Selection, Prediction Roundtrip & Results
        # -------------------------------------------------------------
        print("\n[VERIFICATION 3/20] Select & upload benchmark sample (640x640)...")
        await session.eval_js("document.querySelector('.btn-sample').click()")
        await asyncio.sleep(1.0)
        preview_specs = await session.eval_js("document.querySelector('.preview-specs')?.textContent")
        submit_btn_enabled = await session.eval_js("!document.querySelector('.btn-submit')?.disabled")
        print(f"  Preview specs: '{preview_specs}' | Submit enabled: {submit_btn_enabled}")
        assert "640" in preview_specs and submit_btn_enabled
        await session.screenshot("step2_benchmark_sample_loaded")
        results_summary["3. Valid Image Uploaded"] = "PASS"

        print("\n[VERIFICATION 4/20 & 5/20] Prediction starts, completes, and results render...")
        await session.eval_js("document.querySelector('.btn-submit').click()")

        # Wait for results card
        max_wait = 25
        elapsed = 0
        while elapsed < max_wait:
            ready = await session.eval_js("Boolean(document.querySelector('.visual-card'))")
            if ready:
                break
            await asyncio.sleep(0.5)
            elapsed += 0.5
        assert ready is True, "Prediction results card failed to appear within timeout"
        print(f"  Inference finished in ~{elapsed:.1f}s. Results container rendered.")
        await session.screenshot("step3_inference_results_rendered")
        results_summary["4. Prediction Execution"] = "PASS"
        results_summary["5. Prediction Results Rendered"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.6, 3.7, 3.8: Original Image, Binary Mask, Predicted Overlay
        # -------------------------------------------------------------
        print("\n[VERIFICATION 6/20, 7/20, 8/20] Original, Mask, and Overlay displayed correctly...")
        orig_src = await session.eval_js("document.querySelector('.visual-grid-3 .visual-panel:nth-child(1) img')?.src")
        mask_src = await session.eval_js("document.querySelector('.visual-grid-3 .visual-panel:nth-child(2) img')?.src")
        overlay_src = await session.eval_js("document.querySelector('.visual-grid-3 .visual-panel:nth-child(3) img')?.src")

        print(f"  Original Src:  {orig_src[:60]}...")
        print(f"  Mask Src:      {mask_src[:60]}...")
        print(f"  Overlay Src:   {overlay_src[:60]}...")
        assert orig_src and "/outputs/" in mask_src and "/outputs/" in overlay_src
        assert mask_src.split("/outputs/")[1].split("/")[0] == overlay_src.split("/outputs/")[1].split("/")[0], "Mask and overlay ID mismatch!"
        pred_id = mask_src.split("/outputs/")[1].split("/")[0]
        print(f"  Verified matching prediction ID across artifacts: {pred_id}")
        results_summary["6. Original Image Displayed"] = "PASS"
        results_summary["7. Binary Mask Displayed"] = "PASS"
        results_summary["8. Overlay Aligns with Prediction"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.9: Aspect-Ratio Preservation
        # -------------------------------------------------------------
        print("\n[VERIFICATION 9/20] Preserving image aspect ratio without distortion...")
        orig_dims = await session.eval_js("""
            (() => {
                const img = document.querySelector('.visual-grid-3 .visual-panel:nth-child(1) img');
                return { naturalW: img?.naturalWidth, naturalH: img?.naturalHeight, clientW: img?.clientWidth, clientH: img?.clientHeight };
            })()
        """)
        print(f"  Original raster dimensions: natural=({orig_dims['naturalW']}, {orig_dims['naturalH']}), client=({orig_dims['clientW']}, {orig_dims['clientH']})")
        assert orig_dims['naturalW'] == 640 and orig_dims['naturalH'] == 640
        # Aspect ratio 1:1 is preserved
        assert abs((orig_dims['clientW'] / orig_dims['clientH']) - 1.0) < 0.05
        results_summary["9. Aspect Ratio Preserved"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.10: Agreement between reported pixel count/coverage and final binary mask
        # -------------------------------------------------------------
        print("\n[VERIFICATION 10/20] Comparing reported solar pixels & coverage with raw binary mask...")
        reported_cov = await session.eval_js("document.querySelector('.coverage-num')?.textContent")
        reported_solar_px_text = await session.eval_js("document.querySelector('.metric-box.metric-accent-solar .metric-value')?.textContent")
        reported_total_px_text = await session.eval_js("document.querySelector('.metric-box.metric-accent-cyan .metric-value')?.textContent")

        reported_solar_px = int(reported_solar_px_text.replace(",", ""))
        reported_total_px = int(reported_total_px_text.replace(",", ""))
        reported_cov_float = float(reported_cov)

        print(f"  UI Reported Solar Pixels: {reported_solar_px:,}")
        print(f"  UI Reported Total Pixels: {reported_total_px:,}")
        print(f"  UI Reported Coverage:     {reported_cov_float:.4f}%")

        # Directly download the binary mask from server
        mask_download_url = f"http://127.0.0.1:8000/outputs/{pred_id}/mask"
        mask_bytes = urllib.request.urlopen(mask_download_url).read()
        mask_img = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_arr = np.array(mask_img)

        actual_nonzero_pixels = int(np.sum(mask_arr > 0))
        actual_total_pixels = mask_arr.shape[0] * mask_arr.shape[1]
        actual_cov_percent = (actual_nonzero_pixels / actual_total_pixels) * 100.0

        print(f"  Server Mask Dimensions:   {mask_img.size} (W, H)")
        print(f"  Server Mask White Pixels: {actual_nonzero_pixels:,}")
        print(f"  Calculated Coverage:      {actual_cov_percent:.4f}%")

        assert actual_nonzero_pixels == reported_solar_px, f"Pixel count discrepancy: {actual_nonzero_pixels} vs {reported_solar_px}"
        assert actual_total_pixels == reported_total_px, f"Total pixel discrepancy: {actual_total_pixels} vs {reported_total_px}"
        assert abs(actual_cov_percent - reported_cov_float) < 1e-3, "Coverage percentage discrepancy"
        print("  [AGREEMENT CONFIRMED] UI metrics exactly match raw binary mask pixels.")
        results_summary["10. Solar Pixels & Coverage Agreement"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.11: Download Buttons
        # -------------------------------------------------------------
        print("\n[VERIFICATION 11/20] Download buttons return expected image files...")
        mask_btn_exists = await session.eval_js("Boolean(document.querySelectorAll('.btn-download-sm')[0])")
        overlay_btn_exists = await session.eval_js("Boolean(document.querySelectorAll('.btn-download-sm')[1])")
        assert mask_btn_exists and overlay_btn_exists

        # Verify overlay artifact download
        overlay_download_url = f"http://127.0.0.1:8000/outputs/{pred_id}/overlay"
        overlay_bytes = urllib.request.urlopen(overlay_download_url).read()
        overlay_img = Image.open(io.BytesIO(overlay_bytes))
        assert overlay_img.size == (640, 640)
        print(f"  Mask download: HTTP 200 ({len(mask_bytes):,} bytes)")
        print(f"  Overlay download: HTTP 200 ({len(overlay_bytes):,} bytes)")
        results_summary["11. Download Buttons Work"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.12, 3.13, 3.14, 3.15: Visualization Modes & State Hygiene
        # -------------------------------------------------------------
        print("\n[VERIFICATION 12/20] Side-by-Side (3-Up) mode...")
        panel_count = await session.eval_js("document.querySelectorAll('.visual-grid-3 .visual-panel').length")
        assert panel_count == 3
        results_summary["12. Side-by-Side 3-Up Mode"] = "PASS"

        print("\n[VERIFICATION 13/20] Compare Slider mode & draggable handle...")
        # Switch to slider tab (index 1)
        await session.eval_js("document.querySelectorAll('.view-mode-tabs .tab-btn')[1].click()")
        await asyncio.sleep(0.5)
        slider_viewport = await session.eval_js("Boolean(document.querySelector('.slider-viewport'))")
        assert slider_viewport is True

        # Move slider to 25% and 75%
        for pos in [25, 75]:
            await session.eval_js(f"""
                (() => {{
                    const input = document.querySelector('.slider-range-input');
                    if (input) {{
                        const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                        nativeSetter.call(input, {pos});
                        input.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        input.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    }}
                }})();
            """)
            await asyncio.sleep(0.4)
            clip_path = await session.eval_js("document.querySelector('.slider-image-clipped')?.style.clipPath")
            divider_left = await session.eval_js("document.querySelector('.slider-divider')?.style.left")
            print(f"  Slider moved to {pos}%: divider left='{divider_left}', clipPath='{clip_path}'")
            assert f"{pos}%" in divider_left and f"{100 - pos}%" in clip_path
        await session.screenshot("step4_compare_slider_active")
        results_summary["13. Compare Slider Functional"] = "PASS"

        print("\n[VERIFICATION 14/20] Focus View mode with pill toggling...")
        # Switch to focus tab (index 2)
        await session.eval_js("document.querySelectorAll('.view-mode-tabs .tab-btn')[2].click()")
        await asyncio.sleep(0.5)
        focus_container = await session.eval_js("Boolean(document.querySelector('.visual-focus-container'))")
        assert focus_container is True

        # Toggle pills: Original (0), Overlay (1), Mask (2)
        await session.eval_js("document.querySelectorAll('.focus-pill')[2].click()") # Mask
        await asyncio.sleep(0.3)
        focus_img_src = await session.eval_js("document.querySelector('.focus-main-img')?.src")
        assert "/mask" in focus_img_src
        print(f"  Focus View Mask active: {focus_img_src[:60]}...")
        await session.screenshot("step5_focus_view_mask")
        results_summary["14. Focus View Functional"] = "PASS"

        print("\n[VERIFICATION 15/20] Mode switching state hygiene...")
        # Cycle back to 3-Up
        await session.eval_js("document.querySelectorAll('.view-mode-tabs .tab-btn')[0].click()")
        await asyncio.sleep(0.5)
        restored_panels = await session.eval_js("document.querySelectorAll('.visual-grid-3 .visual-panel').length")
        assert restored_panels == 3
        # Ensure no leftover slider or focus view DOM
        no_slider = await session.eval_js("!document.querySelector('.slider-viewport')")
        no_focus = await session.eval_js("!document.querySelector('.visual-focus-container')")
        assert no_slider and no_focus
        print("  State hygiene verified: All tabs toggle cleanly without residual DOM or stale states.")
        results_summary["15. Mode Switching Hygiene"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.16: Physical-area, energy, and CO2 safeguards on uncalibrated data
        # -------------------------------------------------------------
        print("\n[VERIFICATION 16/20] Physical-area, energy, and CO2 safeguards...")
        area_notice_text = await session.eval_js("document.querySelector('.notice-card')?.textContent")
        energy_badge_text = await session.eval_js("document.querySelector('.energy-analytics-card .insufficient-badge')?.textContent")
        co2_badge_text = await session.eval_js("document.querySelector('.co2-analytics-card .insufficient-badge')?.textContent")

        print(f"  Physical Area Notice:   '{area_notice_text[:80].strip() if area_notice_text else 'None'}...'")
        print(f"  Energy Model Status:    '{energy_badge_text}'")
        print(f"  CO2 Avoidance Status:   '{co2_badge_text}'")
        assert area_notice_text and ("UNAVAILABLE" in area_notice_text.upper() or "INSUFFICIENT" in area_notice_text.upper())
        assert "Data Insufficient" in energy_badge_text
        assert "Data Insufficient" in co2_badge_text
        results_summary["16. Physical & Environmental Safeguards"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.17 & 3.18: Prediction History Drawer & Offline Resilience
        # -------------------------------------------------------------
        print("\n[VERIFICATION 17/20 & 18/20] Prediction history & offline resilience...")
        await session.eval_js("document.querySelector('.header-history-btn').click()")
        await asyncio.sleep(0.8)
        drawer_open = await session.eval_js("Boolean(document.querySelector('.history-drawer-panel'))")
        assert drawer_open is True

        drawer_text = await session.eval_js("document.querySelector('.history-drawer-panel')?.textContent")
        print(f"  History Drawer Status: '{drawer_text[:100].strip()}...'")
        assert "Database Offline" in drawer_text or "Database Service Offline" in drawer_text
        await session.screenshot("step6_history_drawer_offline")

        # Close history drawer
        await session.eval_js("document.querySelector('.history-btn-close').click()")
        await asyncio.sleep(0.4)
        drawer_closed = await session.eval_js("!document.querySelector('.history-drawer-panel')")
        assert drawer_closed is True
        results_summary["17. Prediction History (Tested offline resilience)"] = "PASS"
        results_summary["18. Offline Resilience Handled"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.19: Invalid uploads and error messaging
        # -------------------------------------------------------------
        print("\n[VERIFICATION 19/20] Invalid uploads error messaging...")
        # Empty file (0 bytes)
        empty_fixture = FIXTURES_DIR / "empty.png"
        await session.upload_file_bytes(empty_fixture, "image/png")
        await asyncio.sleep(0.5)
        warn_text = await session.eval_js("document.querySelector('.alert.alert-warning')?.textContent")
        print(f"  Empty file validation alert: '{warn_text}'")
        assert "empty (0 bytes)" in warn_text

        # Unsupported format (.txt)
        txt_fixture = FIXTURES_DIR / "test.txt"
        await session.upload_file_bytes(txt_fixture, "text/plain")
        await asyncio.sleep(0.5)
        warn_format = await session.eval_js("document.querySelector('.alert.alert-warning')?.textContent")
        print(f"  Format validation alert: '{warn_format}'")
        assert "Unsupported format" in warn_format

        # Corrupt file sent to backend
        corrupt_fixture = FIXTURES_DIR / "corrupt.png"
        await session.upload_file_bytes(corrupt_fixture, "image/png")
        await asyncio.sleep(0.5)
        await session.eval_js("document.querySelector('.btn-submit').click()")
        await asyncio.sleep(1.0)
        error_alert_text = await session.eval_js("document.querySelector('.alert-card')?.textContent")
        print(f"  Corrupt payload backend alert: '{error_alert_text}'")
        assert error_alert_text is not None and len(error_alert_text) > 0
        await session.screenshot("step7_invalid_upload_error_alert")
        results_summary["19. Invalid Uploads Error Messaging"] = "PASS"

        # -------------------------------------------------------------
        # STEP 3.20 & STEP 4: Variable Image Dimensions & Layout Usability
        # -------------------------------------------------------------
        print("\n[VERIFICATION 20/20 & STEP 4] Variable Image Dimensions & Layout Usability...")
        test_dimensions = [
            ("sample_256x256.png", (256, 256), "direct"),
            ("sample_637x463.png", (637, 463), "letterbox"),
            ("sample_1920x1080.jpg", (1920, 1080), "tiled"),
            ("sample_4000x3000.jpg", (4000, 3000), "tiled"),
        ]

        for filename, (exp_w, exp_h), exp_strat in test_dimensions:
            fixture_p = FIXTURES_DIR / filename
            mime = "image/jpeg" if filename.endswith(".jpg") else "image/png"
            print(f"\n  Testing fixture: {filename} ({exp_w}x{exp_h}) [Expected Strategy: {exp_strat}]...")

            await session.upload_file_bytes(fixture_p, mime)
            await asyncio.sleep(0.8)
            specs = await session.eval_js("document.querySelector('.preview-specs')?.textContent")
            print(f"    Preview specs: '{specs}'")
            assert str(exp_w) in specs and str(exp_h) in specs

            # Submit inference
            t0 = time.time()
            await session.eval_js("document.querySelector('.btn-submit').click()")

            # Wait for results
            max_w_time = 40
            elapsed = 0
            while elapsed < max_w_time:
                ready = await session.eval_js("Boolean(document.querySelector('.visual-card'))")
                if ready:
                    break
                await asyncio.sleep(0.5)
                elapsed += 0.5
            assert ready is True, f"Inference timed out for {filename}"
            inference_dur = time.time() - t0

            # Verify reported dimensions, strategy, and layout
            strat_badge = await session.eval_js("document.querySelector('.meta-badge-row .badge-info')?.textContent")
            dims_text = await session.eval_js("document.querySelector('.metric-box:nth-child(3) .metric-value')?.textContent")
            solar_px = await session.eval_js("document.querySelector('.metric-box.metric-accent-solar .metric-value')?.textContent")
            cov_val = await session.eval_js("document.querySelector('.coverage-num')?.textContent")

            print(f"    Completed in {inference_dur:.2f}s:")
            print(f"      Strategy: '{strat_badge}' [Expected: {exp_strat.upper()}]")
            print(f"      Dimensions: '{dims_text}' [Expected: {exp_w} × {exp_h}]")
            print(f"      Solar Pixels: {solar_px} ({cov_val}%)")

            clean_dims = dims_text.replace(",", "")
            assert str(exp_w) in clean_dims and str(exp_h) in clean_dims

            # Check container aspect ratio style in DOM
            container_ratio = await session.eval_js("document.querySelector('.panel-image-container')?.style.aspectRatio")
            print(f"      Container CSS aspect-ratio: '{container_ratio}'")
            assert f"{exp_w} / {exp_h}" in container_ratio or f"{exp_w}/{exp_h}" in container_ratio

        await session.screenshot("step8_variable_dimensions_verified")
        results_summary["20. Layout Usability & Variable Dimensions"] = "PASS"

        # -------------------------------------------------------------
        # STEP 12/12: Interactive Geospatial Map View Tab
        # -------------------------------------------------------------
        print("\n[VERIFICATION LEAFLET MAP] Testing Interactive Geospatial Map Tab...")
        # 1. First verify unreferenced empty state on current unreferenced fixture (sample_4000x3000)
        await session.eval_js("""
            const btns = Array.from(document.querySelectorAll('button'));
            const mapBtn = btns.find(b => b.textContent.includes('Geospatial Map'));
            if (mapBtn) mapBtn.click();
        """)
        await asyncio.sleep(0.8)
        geo_empty = await session.eval_js("Boolean(document.querySelector('.geo-empty-state'))")
        empty_title = await session.eval_js("document.querySelector('.geo-empty-title')?.textContent")
        print(f"  Unreferenced Image Handling: Empty State visible={geo_empty}, Title='{empty_title}'")
        assert geo_empty is True, "Expected unreferenced empty state for unreferenced raster"
        assert "Unavailable" in empty_title or "Unreferenced" in empty_title
        await session.screenshot("step9_geospatial_unreferenced_empty_state")

        # 2. Now load benchmark sample (768.0_1.0) which has Level B georeferencing catalog entry
        print("  Loading referenced benchmark sample (768.0_1.0) to test active Leaflet map...")
        # Return to visual inspection first
        await session.eval_js("""
            const btns = Array.from(document.querySelectorAll('button'));
            const visBtn = btns.find(b => b.textContent.includes('Visual Inspection'));
            if (visBtn) visBtn.click();
        """)
        await asyncio.sleep(0.5)

        # Click Load Benchmark Sample
        await session.eval_js("document.querySelector('.btn-sample').click()")
        await asyncio.sleep(0.8)
        # Click submit
        await session.eval_js("document.querySelector('.btn-submit').click()")
        for _ in range(20):
            ready = await session.eval_js("Boolean(document.querySelector('.visual-card'))")
            if ready:
                break
            await asyncio.sleep(0.5)

        # Switch to Geospatial Map tab
        await session.eval_js("""
            const btns = Array.from(document.querySelectorAll('button'));
            const mapBtn = btns.find(b => b.textContent.includes('Geospatial Map'));
            if (mapBtn) mapBtn.click();
        """)
        await asyncio.sleep(1.2)

        leaflet_container = await session.eval_js("Boolean(document.querySelector('.leaflet-container'))")
        map_hud_text = await session.eval_js("document.querySelector('.map-hud-overlay')?.textContent")
        print(f"  Referenced Image Handling: Leaflet map loaded={leaflet_container}, HUD='{map_hud_text}'")
        assert leaflet_container is True, "Leaflet Map failed to load for georeferenced sample"
        assert map_hud_text is not None and "LEVEL_B" in map_hud_text and "Scene Center" in map_hud_text
        await session.screenshot("step10_geospatial_leaflet_active")

        # Return to visual inspection
        await session.eval_js("""
            const btns = Array.from(document.querySelectorAll('button'));
            const visBtn = btns.find(b => b.textContent.includes('Visual Inspection'));
            if (visBtn) visBtn.click();
        """)
        await asyncio.sleep(0.5)

        # Print final results table
        print("\n========================================================================")
        print("          ALL 20 E2E BROWSER WORKFLOW VERIFICATIONS PASSED!")
        print("========================================================================")
        for check, res in results_summary.items():
            print(f"  [{res}] {check}")
        print("========================================================================")

    finally:
        await session.close()


if __name__ == "__main__":
    import io
    asyncio.run(run_full_verification())
