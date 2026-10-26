"""
SolarMap-India — Live Server & API Workflow Verification Script.
Exercises the running server on http://127.0.0.1:8000.
"""

import json
import mimetypes
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"


def post_multipart(url, file_path, params=None):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    filename = os.path.basename(file_path)
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode("utf-8"))
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    query = "&".join([f"{k}={v}" for k, v in params.items()]) if params else ""
    full_url = f"{url}?{query}" if query else url

    req = urllib.request.Request(
        full_url,
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
        method="POST",
    )
    res = urllib.request.urlopen(req)
    return res.status, json.loads(res.read().decode("utf-8"))


def test_workflows():
    print("==================================================================")
    print("SOLARMAP-INDIA LIVE SERVER WORKFLOW & VARIABLE-DIMENSION VERIFICATION")
    print("==================================================================")

    # 1. Health check
    health_req = urllib.request.urlopen(f"{BASE_URL}/health")
    health_data = json.loads(health_req.read().decode("utf-8"))
    print(f"\n1. Backend Health: {health_data['status']}")
    print(f"   Model:         {health_data.get('model')}")
    print(f"   Device:        {health_data.get('device')}")
    print(f"   Checkpoint:    {health_data.get('checkpoint_path')}")
    assert health_data["status"] == "healthy"
    assert health_data["checkpoint_available"] is True

    # 2. Database Health
    db_req = urllib.request.urlopen(f"{BASE_URL}/health/db")
    db_data = json.loads(db_req.read().decode("utf-8"))
    print(f"\n2. Database Health: {db_data['status']}")
    print(f"   Connected:     {db_data.get('connected')}")
    print(f"   Message:       {db_data.get('message')}")

    # 3. Static frontend assets serving from root
    spa_req = urllib.request.Request(f"{BASE_URL}/", headers={"Accept": "text/html"})
    spa_res = urllib.request.urlopen(spa_req)
    spa_html = spa_res.read().decode("utf-8")
    print(f"\n3. Static Frontend Serving: HTTP {spa_res.status}")
    print(f"   Title in HTML: {'SolarMap-India' in spa_html}")
    assert spa_res.status == 200
    assert "SolarMap-India" in spa_html

    # Bundled sample asset
    sample_res = urllib.request.urlopen(f"{BASE_URL}/sample_aerial.png")
    sample_bytes = sample_res.read()
    print(f"   Bundled Sample (sample_aerial.png): HTTP {sample_res.status}, {len(sample_bytes)} bytes")
    assert sample_res.status == 200 and len(sample_bytes) > 0

    # 4. Variable dimensions
    fixtures = [
        ("scratch/browser_fixtures/sample_256x256.png", (256, 256), "direct"),
        ("scratch/browser_fixtures/sample_640x640.png", (640, 640), "direct"),
        ("scratch/browser_fixtures/sample_637x463.png", (637, 463), "letterbox"),
        ("scratch/browser_fixtures/sample_1920x1080.jpg", (1920, 1080), "tiled"),
        ("scratch/browser_fixtures/sample_4000x3000.jpg", (4000, 3000), "tiled"),
    ]

    print("\n4. Testing Variable Image Dimensions & Inference Strategies:")
    for fp, expected_size, expected_strat in fixtures:
        t0 = time.time()
        status, data = post_multipart(f"{BASE_URL}/predict", fp)
        dur = time.time() - t0
        w = data["image_width"]
        h = data["image_height"]
        strat = data.get("processing_strategy")
        solar_px = data["solar_area_pixels"]
        cov = data["solar_coverage_percent"]
        mask_url = data["mask_url"]
        overlay_url = data["overlay_url"]

        print(f"\n   Fixture: {os.path.basename(fp)} in {dur:.2f}s")
        print(f"     HTTP Status:         {status}")
        print(f"     Dimensions:          ({w}, {h}) [Expected: {expected_size}]")
        print(f"     Strategy:            {strat} [Expected: {expected_strat}]")
        print(f"     Solar Pixels:        {solar_px:,} ({cov:.2f}%)")
        print(f"     Connected Regions:   {data.get('detected_region_count')}")
        print(f"     Saved to History:    {data.get('saved_to_history')}")
        print(f"     GSD Status:          {data.get('gsd_status')}")
        print(f"     Physical Area:       {data.get('physical_area_m2')} ({data.get('physical_area_status')})")
        print(f"     Annual Energy (kWh): {data.get('estimated_energy_kwh_year')} ({data.get('energy_status')})")
        print(f"     CO2 Avoidance (t):   {data.get('co2_avoided_tonnes_year')} ({data.get('co2_status')})")

        assert (w, h) == expected_size, f"Dimension mismatch: expected {expected_size}, got ({w}, {h})"
        assert strat == expected_strat, f"Strategy mismatch: expected {expected_strat}, got {strat}"

        # Verify mask artifact
        m_req = urllib.request.urlopen(f"{BASE_URL}{mask_url}")
        m_img = Image.open(m_req)
        assert m_img.size == (w, h), f"Mask dimension mismatch: {m_img.size} vs ({w}, {h})"

        # Verify overlay artifact
        o_req = urllib.request.urlopen(f"{BASE_URL}{overlay_url}")
        o_img = Image.open(o_req)
        assert o_img.size == (w, h), f"Overlay dimension mismatch: {o_img.size} vs ({w}, {h})"

        # Verify strict physical safeguards on uncalibrated images
        assert data.get("physical_area_status") == "insufficient_data"
        assert data.get("physical_area_m2") is None
        assert data.get("energy_status") == "insufficient_data"
        assert data.get("estimated_energy_kwh_year") is None
        assert data.get("co2_status") == "insufficient_data"
        assert data.get("co2_avoided_tonnes_year") is None

    # 5. Error handling tests
    print("\n5. Testing Error Handling & Safeguards:")
    # Empty file
    empty_file = "scratch/browser_fixtures/empty.png"
    Path(empty_file).write_bytes(b"")
    try:
        post_multipart(f"{BASE_URL}/predict", empty_file)
        raise AssertionError("Should have raised error for empty file")
    except urllib.error.HTTPError as e:
        print(f"   Empty file rejection: HTTP {e.code} (detail: {json.loads(e.read().decode())['detail']})")
        assert e.code == 400

    # Corrupted image file
    corrupt_file = "scratch/browser_fixtures/corrupt.png"
    Path(corrupt_file).write_bytes(b"This is not a real PNG image content.")
    try:
        post_multipart(f"{BASE_URL}/predict", corrupt_file)
        raise AssertionError("Should have raised error for corrupt file")
    except urllib.error.HTTPError as e:
        print(f"   Corrupt file rejection: HTTP {e.code} (detail: {json.loads(e.read().decode())['detail']})")
        assert e.code == 400

    # Unsupported format
    unsupported_file = "scratch/browser_fixtures/test.txt"
    Path(unsupported_file).write_text("plain text")
    try:
        post_multipart(f"{BASE_URL}/predict", unsupported_file)
        raise AssertionError("Should have raised error for unsupported format")
    except urllib.error.HTTPError as e:
        print(f"   Unsupported format rejection: HTTP {e.code} (detail: {json.loads(e.read().decode())['detail']})")
        assert e.code == 400

    # 6. Prediction history endpoint behavior
    print("\n6. Testing History Endpoint Behavior:")
    try:
        h_res = urllib.request.urlopen(f"{BASE_URL}/api/predictions/history")
        h_data = json.loads(h_res.read().decode())
        print(f"   History query: HTTP {h_res.status}, records: {len(h_data.get('records', []))}")
    except urllib.error.HTTPError as e:
        # If MongoDB is offline, it should gracefully return 503
        h_err = json.loads(e.read().decode())
        print(f"   History when unconfigured: HTTP {e.code} (detail: {h_err.get('detail')})")
        assert e.code in (200, 503)

    print("\n==================================================================")
    print("ALL LIVE WORKFLOW AND VARIABLE-DIMENSION TESTS PASSED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    test_workflows()
