import os
import io
import zipfile

import pandas as pd
import numpy as np

from PIL import Image, ImageDraw, ImageFont


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\SolarMap-India"

ZIP_PATH = os.path.join(
    BASE,
    "dataset",
    "Solar Images.zip"
)

CSV_PATH = os.path.join(
    BASE,
    "results",
    "verification",
    "pilot_candidates.csv"
)

OUTPUT_DIR = os.path.join(
    BASE,
    "results",
    "verification",
    "pilot_visuals"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Candidates:", len(df))

# Highest-confidence examples
tp_df = (
    df[df["label"] == "TP"]
    .sort_values("score", ascending=False)
    .head(10)
)

fp_df = (
    df[df["label"] == "FP"]
    .sort_values("score", ascending=False)
    .head(10)
)

print("TP examples:", len(tp_df))
print("FP examples:", len(fp_df))


# ============================================================
# ZIP IMAGE LOOKUP
# ============================================================

zf = zipfile.ZipFile(
    ZIP_PATH,
    "r"
)

zip_names = set(
    zf.namelist()
)


def find_member(filename):

    matches = [
        n
        for n in zip_names
        if n.endswith("/" + filename)
        or n == filename
    ]

    return matches[0] if matches else None


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    records,
    title,
    output_name
):

    tile_size = 300
    label_height = 55

    sheet_width = tile_size * 5
    sheet_height = (
        (tile_size + label_height) * 2
    )

    sheet = Image.new(
        "RGB",
        (sheet_width, sheet_height),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(
        records.itertuples()
    ):

        filename = row.filename
        score = float(row.score)

        member = find_member(
            filename
        )

        if member is None:
            continue

        with zf.open(member) as f:

            image = Image.open(f).convert(
                "RGB"
            )

        image = np.array(image)

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1 = int(row.bbox_x1)
        y1 = int(row.bbox_y1)
        x2 = int(row.bbox_x2)
        y2 = int(row.bbox_y2)

        h, w = image.shape[:2]

        # ----------------------------------------------------
        # Add context around candidate
        # ----------------------------------------------------

        bw = max(
            x2 - x1,
            1
        )

        bh = max(
            y2 - y1,
            1
        )

        pad_x = int(
            bw * 1.5
        )

        pad_y = int(
            bh * 1.5
        )

        cx1 = max(
            0,
            x1 - pad_x
        )

        cy1 = max(
            0,
            y1 - pad_y
        )

        cx2 = min(
            w,
            x2 + pad_x
        )

        cy2 = min(
            h,
            y2 + pad_y
        )

        crop = image[
            cy1:cy2,
            cx1:cx2
        ]

        crop_img = Image.fromarray(
            crop
        )

        crop_img.thumbnail(
            (tile_size, tile_size)
        )

        # ----------------------------------------------------
        # Center crop inside tile
        # ----------------------------------------------------

        tile = Image.new(
            "RGB",
            (tile_size, tile_size),
            "white"
        )

        offset_x = (
            tile_size -
            crop_img.width
        ) // 2

        offset_y = (
            tile_size -
            crop_img.height
        ) // 2

        tile.paste(
            crop_img,
            (
                offset_x,
                offset_y
            )
        )

        # ----------------------------------------------------
        # Draw candidate rectangle
        # ----------------------------------------------------

        local_x1 = (
            x1 -
            cx1
        )

        local_y1 = (
            y1 -
            cy1
        )

        local_x2 = (
            x2 -
            cx1
        )

        local_y2 = (
            y2 -
            cy1
        )

        scale_x = (
            crop_img.width /
            max(crop.shape[1], 1)
        )

        scale_y = (
            crop_img.height /
            max(crop.shape[0], 1)
        )

        draw_x1 = (
            offset_x +
            int(local_x1 * scale_x)
        )

        draw_y1 = (
            offset_y +
            int(local_y1 * scale_y)
        )

        draw_x2 = (
            offset_x +
            int(local_x2 * scale_x)
        )

        draw_y2 = (
            offset_y +
            int(local_y2 * scale_y)
        )

        draw.rectangle(
            [
                draw_x1,
                draw_y1,
                draw_x2,
                draw_y2
            ],
            outline="red",
            width=3
        )

        # ----------------------------------------------------
        # Position in sheet
        # ----------------------------------------------------

        row_number = index // 5
        col_number = index % 5

        px = (
            col_number *
            tile_size
        )

        py = (
            row_number *
            (tile_size + label_height)
        )

        sheet.paste(
            tile,
            (
                px,
                py
            )
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{index + 1}. "
            f"{filename}\n"
            f"score={score:.3f}"
        )

        draw.text(
            (
                px + 5,
                py + tile_size + 5
            ),
            label,
            fill="black"
        )

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    draw.text(
        (10, 5),
        title,
        fill="black"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    sheet.save(
        output_path
    )

    print(
        "Saved:",
        output_path
    )


# ============================================================
# CREATE SHEETS
# ============================================================

create_contact_sheet(
    tp_df,
    "TOP 10 TRUE POSITIVE CANDIDATES",
    "pilot_TP_examples.jpg"
)

create_contact_sheet(
    fp_df,
    "TOP 10 FALSE POSITIVE CANDIDATES",
    "pilot_FP_examples.jpg"
)

zf.close()

print("\n✅ VISUALIZATION COMPLETE")