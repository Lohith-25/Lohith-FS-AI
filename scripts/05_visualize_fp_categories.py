import os
import zipfile
import pandas as pd
import numpy as np

from PIL import Image, ImageDraw


# ============================================================
# SOLARMAP — STEP 5 VISUALIZE FP CATEGORIES
# ============================================================

BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images.zip"
)

DIAGNOSIS_CSV = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_diagnosis.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "verification",
    "fp_category_visuals"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(DIAGNOSIS_CSV):
    raise FileNotFoundError(f"Diagnosis CSV not found:\n{DIAGNOSIS_CSV}")

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"Dataset ZIP not found:\n{ZIP_PATH}")

df = pd.read_csv(DIAGNOSIS_CSV)

print("Total diagnosed candidates:", len(df))
print("\nCategories:")
print(df["diagnostic_category"].value_counts())

zf = zipfile.ZipFile(ZIP_PATH, "r")
zip_names = set(zf.namelist())

def find_member(filename):
    matches = [
        n for n in zip_names
        if n.endswith("/" + filename) or n == filename
    ]
    return matches[0] if matches else None

def create_sheet(category, records, output_name):
    records = records.head(6)

    tile_width = 320
    tile_height = 350

    sheet = Image.new(
        "RGB",
        (tile_width * 3, tile_height * 2),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for idx, row in enumerate(records.itertuples()):
        filename = row.filename
        member = find_member(filename)

        if member is None:
            print(f"WARNING: image not found in ZIP: {filename}")
            continue

        with zf.open(member) as f:
            image = Image.open(f).convert("RGB")

        # PIL -> NumPy array so NumPy slicing works
        image_np = np.asarray(image)

        h, w = image_np.shape[:2]

        x1 = max(0, min(int(row.bbox_x1), w - 1))
        y1 = max(0, min(int(row.bbox_y1), h - 1))
        x2 = max(x1 + 1, min(int(row.bbox_x2), w))
        y2 = max(y1 + 1, min(int(row.bbox_y2), h))

        bw = max(x2 - x1, 1)
        bh = max(y2 - y1, 1)

        pad_x = int(bw * 2.0)
        pad_y = int(bh * 2.0)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)

        crop = image_np[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            print(f"WARNING: empty crop for {filename}")
            continue

        crop_img = Image.fromarray(crop)
        crop_img.thumbnail((tile_width - 20, tile_height - 70))

        tile = Image.new(
            "RGB",
            (tile_width, tile_height),
            "white"
        )

        ox = (tile_width - crop_img.width) // 2
        oy = 10

        tile.paste(crop_img, (ox, oy))

        scale_x = crop_img.width / max(crop.shape[1], 1)
        scale_y = crop_img.height / max(crop.shape[0], 1)

        rx1 = ox + int((x1 - cx1) * scale_x)
        ry1 = oy + int((y1 - cy1) * scale_y)
        rx2 = ox + int((x2 - cx1) * scale_x)
        ry2 = oy + int((y2 - cy1) * scale_y)

        draw_tile = ImageDraw.Draw(tile)

        draw_tile.rectangle(
            [rx1, ry1, rx2, ry2],
            outline="red",
            width=3
        )

        text = (
            f"{idx + 1}. {filename}\n"
            f"score={row.score:.3f}\n"
            f"GT IoU={row.max_iou_with_GT:.3f}\n"
            f"matched-pred IoU="
            f"{row.max_iou_with_matched_prediction:.3f}"
        )

        draw_tile.text(
            (8, tile_height - 62),
            text,
            fill="black"
        )

        sheet_x = (idx % 3) * tile_width
        sheet_y = (idx // 3) * tile_height

        sheet.paste(tile, (sheet_x, sheet_y))

    title = f"{category} — Top {min(6, len(records))} examples"
    draw.text((10, 5), title, fill="black")

    output_path = os.path.join(OUTPUT_DIR, output_name)
    sheet.save(output_path, quality=95)

    print("Saved:", output_path)

for category in [
    "BACKGROUND_LIKE",
    "AMBIGUOUS",
    "DUPLICATE_LIKE"
]:
    category_df = (
        df[df["diagnostic_category"] == category]
        .sort_values("score", ascending=False)
        .copy()
    )

    create_sheet(
        category,
        category_df,
        category.lower() + ".jpg"
    )

zf.close()

print("\n✅ CATEGORY VISUALIZATION COMPLETE")
