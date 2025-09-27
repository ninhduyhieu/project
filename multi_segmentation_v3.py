
"""
multi_segmentation_v2.py
------------------------
Segment WATER, PARK/GREEN, and ROADS from a map/satellite image with
priority painting to avoid class overwrite (WATER > PARKS > ROADS)
and improved WATER detection.

Key fixes vs v1:
- Paint order fixed: roads -> parks -> water (so water never turns gray).
- Roads explicitly excluded by union mask (water|parks) before coloring.
- Improved water mask:
  * union of 2 HSV blue/cyan bands
  * blue dominance vs BOTH green & red channels
  * optional ratio rule b/(max(g,r)+1) >= b_ratio
  * slightly stronger closing to keep thin rivers continuous
- Exports a single colored mask: <base>_seg_colored.png
  (water blue, parks green, roads gray on white background).

Run:
python multi_segmentation_v2.py --input map.png --out_dir out
"""

import argparse
import os
import cv2
import numpy as np
import pandas as pd

# ---------------- Utilities ----------------
def connected_region_stats(mask):
    """Return (DataFrame, labels) for connected components on a binary mask (255=FG)."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    rows = []
    for label in range(1, num_labels):  # skip background
        x, y, w, h, area = stats[label]
        cx, cy = centroids[label]
        rows.append({
            "region_id": int(label),
            "pixel_area": int(area),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_w": int(w),
            "bbox_h": int(h),
            "centroid_x": float(cx),
            "centroid_y": float(cy),
        })
    return pd.DataFrame(rows), labels

def save_regions_csv(mask, path):
    df, _ = connected_region_stats(mask)
    df.to_csv(path, index=False)

# --------- Icon filtering ----------
def build_icon_mask(img_bgr, s_min=120, v_min=120, area_max=2500, include_hues=True):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = []
    m_sv = cv2.inRange(hsv, (0, s_min, v_min), (179, 255, 255))
    masks.append(m_sv)
    if include_hues:
        ranges = [
            ((0,   s_min, v_min), (10,  255, 255)),
            ((170, s_min, v_min), (179, 255, 255)),
            ((11,  s_min, v_min), (30,  255, 255)),
            ((135, s_min, v_min), (165, 255, 255)),
        ]
        for lo, hi in ranges:
            masks.append(cv2.inRange(hsv, lo, hi))
    m = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for mm in masks:
        m = cv2.bitwise_or(m, mm)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if 5 <= area <= area_max:
            out[labels == label] = 255
    out = cv2.morphologyEx(out, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    return out

# ---------------- Segmentation routines ----------------
def segment_water(
    img_bgr,
    # two HSV bands to catch blue + cyan (tuned for Google/OSM styles)
    h1_low=95,  h1_high=145,  # deep blue
    h2_low=80,  h2_high=94,   # cyan/teal
    s_low=20, v_low=15, s_high=255, v_high=255,
    veg_h_low=35, veg_h_high=90, veg_s_low=40, veg_v_low=40,
    # channel dominance
    b_over_g=8, b_over_r=8, b_ratio=1.05,
    # morphology
    k_open=3, k_close=11,
    min_area=40,
    # icon handling
    icons_mode="auto",
    icon_s_min=120, icon_v_min=120, icon_area_max=2500,
):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Base masks (two hue bands)
    m1 = cv2.inRange(hsv, (h1_low, s_low, v_low), (h1_high, s_high, v_high))
    m2 = cv2.inRange(hsv, (h2_low, s_low, v_low), (h2_high, s_high, v_high))
    m_water = cv2.bitwise_or(m1, m2)

    # Remove vegetation greens
    m_veg = cv2.inRange(hsv, (veg_h_low, veg_s_low, veg_v_low), (veg_h_high, 255, 255))
    m = cv2.bitwise_and(m_water, cv2.bitwise_not(m_veg))

    # Blue dominance (vs G and R) + ratio
    b, g, r = cv2.split(img_bgr)
    dom = (
        (b.astype(np.int16) - g.astype(np.int16) >= b_over_g) &
        (b.astype(np.int16) - r.astype(np.int16) >= b_over_r) &
        (b.astype(np.float32) / (np.maximum(g, r).astype(np.float32) + 1.0) >= b_ratio)
    )
    m = cv2.bitwise_and(m, (dom.astype(np.uint8) * 255))

    # Optional: remove icon overlaps
    if icons_mode != "off":
        icon_mask = build_icon_mask(img_bgr, s_min=icon_s_min, v_min=icon_v_min, area_max=icon_area_max, include_hues=True)
        m = cv2.bitwise_and(m, cv2.bitwise_not(icon_mask))

    # Morphology (stronger close to connect rivers)
    if k_open > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))

    # Area filter
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    m_filtered = np.zeros_like(m)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            m_filtered[labels == label] = 255
    return m_filtered

def segment_parks(
    img_bgr,
    park_h_low=35, park_h_high=90, park_s_low=40, park_v_low=40,
    k_open=5, k_close=7, min_area=120,
):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (park_h_low, park_s_low, park_v_low), (park_h_high, 255, 255))
    if k_open > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    m_filtered = np.zeros_like(m)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            m_filtered[labels == label] = 255
    return m_filtered

def segment_roads(
    img_bgr,
    s_max=40, v_min=190, k_open=3, k_close=3, min_area=80, exclude_mask=None,
):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, v_min), (179, s_max, 255))
    if exclude_mask is not None:
        m = cv2.bitwise_and(m, cv2.bitwise_not(exclude_mask))
    if k_open > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))
    if k_close > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k_close, k_close), np.uint8))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    m_filtered = np.zeros_like(m)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            m_filtered[labels == label] = 255
    return m_filtered

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)

    # --- WATER params ---
    ap.add_argument("--h1_low", type=int, default=95)
    ap.add_argument("--h1_high", type=int, default=145)
    ap.add_argument("--h2_low", type=int, default=80)
    ap.add_argument("--h2_high", type=int, default=94)
    ap.add_argument("--s_low", type=int, default=20)
    ap.add_argument("--v_low", type=int, default=15)
    ap.add_argument("--s_high", type=int, default=255)
    ap.add_argument("--v_high", type=int, default=255)
    ap.add_argument("--veg_h_low", type=int, default=35)
    ap.add_argument("--veg_h_high", type=int, default=90)
    ap.add_argument("--veg_s_low", type=int, default=40)
    ap.add_argument("--veg_v_low", type=int, default=40)
    ap.add_argument("--b_over_g", type=int, default=8)
    ap.add_argument("--b_over_r", type=int, default=8)
    ap.add_argument("--b_ratio", type=float, default=1.05)
    ap.add_argument("--w_open", type=int, default=3)
    ap.add_argument("--w_close", type=int, default=11)
    ap.add_argument("--w_min_area", type=int, default=40)
    ap.add_argument("--icons", choices=["auto", "off"], default="auto")
    ap.add_argument("--icon_s_min", type=int, default=120)
    ap.add_argument("--icon_v_min", type=int, default=120)
    ap.add_argument("--icon_area_max", type=int, default=2500)

    # --- PARK params ---
    ap.add_argument("--park_h_low", type=int, default=35)
    ap.add_argument("--park_h_high", type=int, default=90)
    ap.add_argument("--park_s_low", type=int, default=40)
    ap.add_argument("--park_v_low", type=int, default=40)
    ap.add_argument("--park_open", type=int, default=5)
    ap.add_argument("--park_close", type=int, default=7)
    ap.add_argument("--park_min_area", type=int, default=120)

    # --- ROAD params ---
    ap.add_argument("--road_s_max", type=int, default=40)
    ap.add_argument("--road_v_min", type=int, default=190)
    ap.add_argument("--road_open", type=int, default=3)
    ap.add_argument("--road_close", type=int, default=3)
    ap.add_argument("--road_min_area", type=int, default=80)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]

    img_bgr = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read {args.input}")

    # Denoise
    img_bgr = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)

    # --- Segment classes ---
    water = segment_water(
        img_bgr,
        h1_low=args.h1_low, h1_high=args.h1_high,
        h2_low=args.h2_low, h2_high=args.h2_high,
        s_low=args.s_low, v_low=args.v_low, s_high=args.s_high, v_high=args.v_high,
        veg_h_low=args.veg_h_low, veg_h_high=args.veg_h_high, veg_s_low=args.veg_s_low, veg_v_low=args.veg_v_low,
        b_over_g=args.b_over_g, b_over_r=args.b_over_r, b_ratio=args.b_ratio,
        k_open=args.w_open, k_close=args.w_close, min_area=args.w_min_area,
        icons_mode=args.icons, icon_s_min=args.icon_s_min, icon_v_min=args.icon_v_min, icon_area_max=args.icon_area_max,
    )

    parks = segment_parks(
        img_bgr,
        park_h_low=args.park_h_low, park_h_high=args.park_h_high,
        park_s_low=args.park_s_low, park_v_low=args.park_v_low,
        k_open=args.park_open, k_close=args.park_close, min_area=args.park_min_area,
    )

    # Exclude water/parks from roads
    exclude = cv2.bitwise_or(water, parks)
    roads = segment_roads(
        img_bgr,
        s_max=args.road_s_max, v_min=args.road_v_min,
        k_open=args.road_open, k_close=args.road_close, min_area=args.road_min_area,
        exclude_mask=exclude,
    )

    # --- Compose colored mask with priority (roads -> parks -> water) ---
    BLUE  = (255, 0, 0)     # water
    GREEN = (0, 200, 0)     # parks
    GRAY  = (160, 160, 160) # roads

    colored = np.full_like(img_bgr, 255)
    colored[roads > 0] = GRAY
    colored[parks > 0] = GREEN
    colored[water > 0] = BLUE  # highest priority

    # Overlay (optional visual on original)
    overlay = img_bgr.copy()
    overlay[roads > 0] = GRAY
    overlay[parks > 0] = GREEN
    overlay[water > 0] = BLUE

    # --- Save files ---
    def p(name): return os.path.join(args.out_dir, f"{base}_{name}")
    cv2.imwrite(p("mask_water.png"), water)
    cv2.imwrite(p("mask_parks.png"), parks)
    cv2.imwrite(p("mask_roads.png"), roads)
    cv2.imwrite(p("overlay.png"), overlay)
    cv2.imwrite(p("seg_colored.png"), colored)

    save_regions_csv(water, p("regions_water.csv"))
    save_regions_csv(parks, p("regions_parks.csv"))
    save_regions_csv(roads, p("regions_roads.csv"))

    print("Saved outputs to:", args.out_dir)

if __name__ == "__main__":
    main()
