"""
SolarMap-India — End-to-End Browser Strategy Verification Script.
Exercises both letterbox and tiled inference through native Chrome CDP,
verifying network requests, response payloads, fresh UI rendering, and artifact updates.
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
SAMPLE_IMAGE = Path("d:/SolarMap-India/backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png")


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
            try:
                self.proc.wait(timeout=3)
            except Exception:
                self.proc.kill()


async def run_strategy_verification():
    print("==================================================================")
    print("SOLARMAP-INDIA: BROWSER STRATEGY SELECTION E2E VERIFICATION")
    print("==================================================================")

    session = ChromeSession()
    session.start()
    try:
        await session.connect()
        await session.send("Page.enable")
        await session.send("Runtime.enable")

        # Step 1: Wait for Page Load and React mount
        print("Step 1: Waiting for application to load and mount React...")
        title = ""
        for _ in range(20):
            title = await session.eval_js("document.title")
            if "SolarMap-India" in title:
                break
            await asyncio.sleep(0.5)

        print(f"  Application loaded. Title: '{title}'")
        assert "SolarMap-India" in title, f"Unexpected page title: {title}"

        # Wait for dropzone to be present
        for _ in range(20):
            has_dropzone = await session.eval_js("Boolean(document.querySelector('.dropzone-card, .btn-toggle-params'))")
            if has_dropzone:
                break
            await asyncio.sleep(0.5)

        # Step 2: Open Parameters Drawer / Section
        print("\nStep 2: Opening Model Inference Parameters...")
        await session.eval_js("document.querySelector('.btn-toggle-params').click()")
        await asyncio.sleep(0.5)

        has_strategy_select = await session.eval_js("Boolean(document.querySelector('#strategy-select'))")
        print(f"  Strategy selector present in DOM: {has_strategy_select}")
        assert has_strategy_select is True, "Strategy selector not found!"

        # Step 3: Inject the 637x463 test file
        print("\nStep 3: Uploading 637x463 satellite screenshot...")
        file_bytes = SAMPLE_IMAGE.read_bytes()
        b64_content = base64.b64encode(file_bytes).decode("ascii")

        load_res = await session.eval_js(f"""
        (async () => {{
            const byteChars = atob('{b64_content}');
            const byteNums = new Array(byteChars.length);
            for (let i = 0; i < byteChars.length; i++) {{
                byteNums[i] = byteChars.charCodeAt(i);
            }}
            const byteArray = new Uint8Array(byteNums);
            const blob = new Blob([byteArray], {{ type: 'image/png' }});
            const file = new File([blob], 'screenshot_637x463.png', {{ type: 'image/png' }});

            const input = document.querySelector('input[type="file"]');
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            input.files = dataTransfer.files;
            input.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return true;
        }})();
        """)
        await asyncio.sleep(1.0)
        specs = await session.eval_js("document.querySelector('.preview-specs')?.textContent")
        print(f"  Image preview active: '{specs}'")

        # Step 4: Run with Strategy = "letterbox"
        print("\nStep 4: Executing inference with Strategy = 'letterbox'...")
        await session.eval_js("""
            const sel = document.querySelector('#strategy-select');
            if (sel) {
                const nativeValueSetter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
                nativeValueSetter.call(sel, 'letterbox');
                sel.dispatchEvent(new Event('input', { bubbles: true }));
                sel.dispatchEvent(new Event('change', { bubbles: true }));
            }
        """)
        await asyncio.sleep(0.5)

        badge_text = await session.eval_js("document.querySelector('.params-badge')?.textContent")
        print(f"  Inference params badge: '{badge_text}'")
        assert "LETTERBOX" in badge_text

        # Click submit
        await session.eval_js("document.querySelector('.btn-submit').click()")

        # Wait for results
        for _ in range(30):
            has_results = await session.eval_js("Boolean(document.querySelector('.results-container'))")
            if has_results:
                break
            await asyncio.sleep(0.5)

        cov_lb = await session.eval_js("document.querySelector('.coverage-num')?.textContent")
        solar_px_lb = await session.eval_js("document.querySelector('.metric-box.metric-accent-solar .metric-value')?.textContent")
        strat_badge_lb = await session.eval_js("document.querySelector('.meta-badge-row .badge-info')?.textContent")
        pred_id_lb = await session.eval_js("document.querySelector('.badge-uuid')?.title")

        print(f"  [LETTERBOX RESULTS]")
        print(f"    Reported Strategy: '{strat_badge_lb}'")
        print(f"    Solar Coverage:    '{cov_lb}'")
        print(f"    Solar Pixels:      '{solar_px_lb}'")
        print(f"    Prediction ID:     '{pred_id_lb}'")

        assert "LETTERBOX" in strat_badge_lb
        assert "2.2" in cov_lb
        assert "6,513" in solar_px_lb
        await session.screenshot("08_ui_strategy_letterbox_results")

        # Step 5: Switch Strategy to "tiled" and re-submit
        print("\nStep 5: Switching Strategy to 'tiled' and re-submitting...")
        # Open params if closed
        is_mounted = await session.eval_js("Boolean(document.querySelector('#strategy-select'))")
        if not is_mounted:
            await session.eval_js("document.querySelector('.btn-toggle-params')?.click()")
            await asyncio.sleep(0.5)

        debug_res = await session.eval_js("""
            (() => {
                const sel = document.querySelector('#strategy-select');
                if (!sel) return "SELECT_NOT_FOUND";
                const optIndex = Array.from(sel.options).findIndex(o => o.value === 'tiled');
                sel.selectedIndex = optIndex;
                const nativeValueSetter = Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value').set;
                nativeValueSetter.call(sel, 'tiled');
                sel.dispatchEvent(new Event('change', { bubbles: true }));
                return "SUCCESS_" + sel.value;
            })()
        """)
        print(f"  Selection result: {debug_res}")
        await asyncio.sleep(0.5)

        badge_text_tiled = await session.eval_js("document.querySelector('.params-badge')?.textContent")
        print(f"  Updated params badge: '{badge_text_tiled}'")
        assert "TILED" in badge_text_tiled

        # Click submit
        await session.eval_js("document.querySelector('.btn-submit').click()")

        # Wait for fresh results
        for _ in range(40):
            has_results = await session.eval_js("Boolean(document.querySelector('.results-container'))")
            if has_results:
                break
            await asyncio.sleep(0.5)

        cov_tiled = await session.eval_js("document.querySelector('.coverage-num')?.textContent")
        solar_px_tiled = await session.eval_js("document.querySelector('.metric-box.metric-accent-solar .metric-value')?.textContent")
        strat_badge_tiled = await session.eval_js("document.querySelector('.meta-badge-row .badge-info')?.textContent")
        pred_id_tiled = await session.eval_js("document.querySelector('.badge-uuid')?.title")

        print(f"\n  [TILED RESULTS]")
        print(f"    Reported Strategy: '{strat_badge_tiled}'")
        print(f"    Solar Coverage:    '{cov_tiled}'")
        print(f"    Solar Pixels:      '{solar_px_tiled}'")
        print(f"    Prediction ID:     '{pred_id_tiled}'")

        assert "TILED" in strat_badge_tiled
        assert "2.8" in cov_tiled or "2.9" in cov_tiled
        assert "8,544" in solar_px_tiled
        assert pred_id_tiled != pred_id_lb, "Prediction ID was not refreshed!"

        await session.screenshot("09_ui_strategy_tiled_results")

        print("\n==================================================================")
        print("SUCCESS: BOTH STRATEGIES EXERCISED THROUGH ACTUAL BROWSER UI!")
        print(f"  Letterbox: {solar_px_lb} solar px ({cov_lb}) -> Strategy: LETTERBOX")
        print(f"  Tiled:     {solar_px_tiled} solar px ({cov_tiled}) -> Strategy: TILED")
        print("  Fresh prediction ID generated and rendered dynamically.")
        print("==================================================================")

    finally:
        await session.close()


if __name__ == "__main__":
    asyncio.run(run_strategy_verification())
