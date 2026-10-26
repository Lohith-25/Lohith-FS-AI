import json
import urllib.request
from pathlib import Path

def test_api(strat):
    url = f"http://127.0.0.1:8000/predict?threshold=0.5&min_component_area=10&strategy={strat}"
    boundary = "----Boundary12345"
    file_bytes = Path("backend/uploads/bbb3a42a-29c3-429c-b061-8f21bb6efba0/input.png").read_bytes()
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="input.png"\r\n')
    body.extend(b"Content-Type: image/png\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    req = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
        method="POST"
    )
    res = urllib.request.urlopen(req)
    data = json.loads(res.read().decode("utf-8"))
    print(f"API requested strategy={strat:10s} -> returned strategy={data.get('processing_strategy'):10s}, solar_px={data['solar_area_pixels']:5d}, cov={data['solar_coverage_percent']:5.2f}%, regions={data.get('detected_region_count', 0):2d}, pred_id={data.get('prediction_id')}")

if __name__ == "__main__":
    test_api("auto")
    test_api("letterbox")
    test_api("tiled")
