"""
Debug blue color detection for player HP bars.
"""

import cv2
import numpy as np
from ocr_reader import ClashOCRReader
import time

reader = ClashOCRReader()

print("[INFO] Waiting 5 seconds before calibration...")
for i in range(5, 0, -1):
    print(f"  Starting calibration in {i}...")
    time.sleep(1)

# Force calibration
while not reader.calibrated:
    if reader.calibrate_hp_bars():
        break
    time.sleep(1)

print("\n[INFO] Testing blue color detection...")
frame = reader.capture_screen()

# Test different blue HSV ranges
blue_ranges = [
    ("Current (100-130)", np.array([100, 120, 70]), np.array([130, 255, 255])),
    ("Wider hue (90-140)", np.array([90, 120, 70]), np.array([140, 255, 255])),
    ("Lower saturation (90-140, sat 80)", np.array([90, 80, 70]), np.array([140, 255, 255])),
    ("Lower value (90-140, val 50)", np.array([90, 120, 50]), np.array([140, 255, 255])),
    ("Very permissive", np.array([90, 50, 50]), np.array([140, 255, 255])),
]

for label, lower, upper in blue_ranges:
    print(f"\n{label}:")
    
    for bar_name, bar_info in [("Player Left", reader.player_left_bar), ("Player Right", reader.player_right_bar)]:
        x, y, w, h, baseline = bar_info
        bar_region = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        
        mask = cv2.inRange(hsv, lower, upper)
        pixel_count = np.count_nonzero(mask)
        percentage = (pixel_count / baseline * 100.0) if baseline > 0 else 0
        
        print(f"  {bar_name}: {pixel_count}/{baseline} pixels = {percentage:.1f}%")
        
        # Save sample for visual inspection
        if label == "Current (100-130)":
            # Sample the center pixel
            center_bgr = bar_region[h//2, w//2]
            center_hsv = hsv[h//2, w//2]
            print(f"    Center pixel - BGR: {center_bgr}, HSV: {center_hsv}")

print("\n[INFO] Check which range gives better detection for player bars")
