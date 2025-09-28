#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remove_black_circles_auto.py
----------------------------
Tự động xóa các chấm tròn đen trong ảnh trắng-đen.
Nếu không truyền --output thì kết quả sẽ được lưu trong thư mục 'filtered_output/'.
"""

import argparse
import os
import cv2
import numpy as np
import math

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--input", "-i", required=True, help="Path tới ảnh input (đen-trắng)")
    p.add_argument("--output", "-o", default=None, help="Path để lưu ảnh output (mặc định: filtered_output/)")
    p.add_argument("--min_area", type=float, default=50, help="Diện tích nhỏ nhất của vòng tròn cần xóa (px^2)")
    p.add_argument("--max_area", type=float, default=3000, help="Diện tích lớn nhất của vòng tròn cần xóa (px^2)")
    p.add_argument("--circ_low", type=float, default=0.70, help="Ngưỡng circularity thấp")
    p.add_argument("--circ_high", type=float, default=1.20, help="Ngưỡng circularity cao")
    p.add_argument("--invert", action="store_true", help="Đảo ngược ảnh trước khi xử lý (nếu nền đen)")
    return p.parse_args()

def ensure_binary(gray: np.ndarray) -> np.ndarray:
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return bw

def remove_black_circles(img_path, out_path,
                         min_area, max_area,
                         circ_low, circ_high,
                         invert=False):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise RuntimeError(f"Không đọc được ảnh: {img_path}")
    if invert:
        gray = 255 - gray
    bw = ensure_binary(gray)

    contours, _ = cv2.findContours(255 - bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = bw.copy()
    removed = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perim = cv2.arcLength(cnt, True)
        if perim == 0:
            continue
        circularity = 4 * math.pi * area / (perim * perim)
        if circ_low <= circularity <= circ_high and min_area <= area <= max_area:
            cv2.drawContours(result, [cnt], -1, 255, -1)
            removed += 1

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, result)
    return removed, out_path

def main():
    args = parse_args()
    # Nếu không có output -> tự tạo thư mục filtered_output
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = "filtered_output"
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, base + "_no_circles.png")

    removed, out_file = remove_black_circles(
        img_path=args.input,
        out_path=args.output,
        min_area=args.min_area,
        max_area=args.max_area,
        circ_low=args.circ_low,
        circ_high=args.circ_high,
        invert=args.invert
    )
    print(f"✅ Đã lưu kết quả tại: {out_file}")
    print(f"🟢 Đã xóa {removed} hình tròn đen.")

if __name__ == "__main__":
    main()
