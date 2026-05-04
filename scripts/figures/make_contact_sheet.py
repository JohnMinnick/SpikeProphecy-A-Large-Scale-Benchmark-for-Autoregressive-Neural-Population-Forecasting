"""Build contact sheet of all figure candidates for morning review.

Outputs to docs/neurips_neurocog/figure_candidates/CONTACT_SHEET.png.
Keeps dimensions under 2000px wide to stay under the image-input limit.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from figures.style import apply_style


CANDIDATES_DIR = (
    PROJECT_ROOT / "docs" / "neurips_neurocog" / "figure_candidates"
)

SECTIONS = [
    ("F1_hero", "F1 — Hero / page-1 visual"),
    ("F2_cross_arch", "F2 — Cross-architecture decodability (main result)"),
    ("F3_arch_compare", "F3 — Architecture-specific comparison"),
    ("F4_region", "F4 — Brain-region predictability"),
    ("F5_synthetic", "F5 — Synthetic-population validation"),
]


def build():
    apply_style()

    # Collect PNGs per section
    section_images = []
    for folder, title in SECTIONS:
        folder_path = CANDIDATES_DIR / folder
        pngs = sorted(folder_path.glob("*.png"))
        if not pngs:
            continue
        section_images.append((title, pngs))

    # Max 5 items per row (max width); wrap if more
    # Thumbnail max size
    thumb_w = 560  # px
    pad_x = 16
    pad_y = 40
    header_h = 44
    section_gap = 12

    # Build per-section rows then stack vertically
    section_panels = []
    total_h = 0
    max_w = 0
    for title, pngs in section_images:
        row_w, row_h = 0, 0
        thumbs = []
        for p in pngs:
            img = Image.open(p)
            aspect = img.height / img.width
            th_h = int(thumb_w * aspect)
            img_th = img.resize((thumb_w, th_h))
            thumbs.append((img_th, p.stem))
            row_w += thumb_w + pad_x
            row_h = max(row_h, th_h)
        section_panels.append((title, thumbs, row_w, row_h))
        total_h += header_h + row_h + pad_y + section_gap
        max_w = max(max_w, row_w)

    # Clamp width to max 1920 — downscale if needed
    target_w = min(max_w, 1920)
    scale = target_w / max_w if max_w > target_w else 1.0

    # Render
    fig_w = target_w + 2 * pad_x
    fig_h = int(total_h * scale)

    sheet = Image.new("RGB", (fig_w, fig_h), "white")
    from PIL import ImageDraw, ImageFont

    try:
        font_title = ImageFont.truetype("arial.ttf", 22)
        font_label = ImageFont.truetype("arial.ttf", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_label = ImageFont.load_default()

    draw = ImageDraw.Draw(sheet)
    y = pad_y // 2
    for title, thumbs, row_w, row_h in section_panels:
        scaled_row_h = int(row_h * scale)
        draw.text((pad_x, y), title, fill="#222", font=font_title)
        y += header_h
        x = pad_x
        for img, name in thumbs:
            new_w = int(thumb_w * scale)
            new_h = int(img.height * scale)
            img_sm = img.resize((new_w, new_h))
            sheet.paste(img_sm, (x, y))
            draw.text(
                (x, y + new_h + 3), name, fill="#444", font=font_label
            )
            x += new_w + int(pad_x * scale)
        y += scaled_row_h + pad_y + section_gap

    out = CANDIDATES_DIR / "CONTACT_SHEET.png"
    sheet.save(out, optimize=True)
    print(
        f"Wrote {out} ({sheet.width}\u00d7{sheet.height} px, "
        f"{out.stat().st_size/1024:.0f} KB)"
    )


if __name__ == "__main__":
    build()
