"""
Debug HP bar width detection - save images to see what's being detected.
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

print("\n[INFO] Starting calibration attempts...")
max_attempts = 10
for attempt in range(1, max_attempts + 1):
    print(f"\n[ATTEMPT {attempt}] Searching for HP bars...")
    
    frame = reader.capture_screen()
    
    # Try to find bars
    opponent_bars = reader.find_hp_bars_by_color(frame, 'red', search_top=True)
    player_bars = reader.find_hp_bars_by_color(frame, 'blue', search_top=False)
    
    print(f"  Found {len(opponent_bars)} opponent (red) bars")
    print(f"  Found {len(player_bars)} player (blue) bars")
    
    if len(opponent_bars) >= 2 and len(player_bars) >= 2:
        print("  SUCCESS! All bars detected, calibrating...")
        reader.calibrate_hp_bars()
        break
    else:
        print("  Not enough bars detected, retrying in 1 second...")
        time.sleep(1)
else:
    print("\n[ERROR] Could not detect all bars after 10 attempts")
    print("Make sure:")
    print("  - Battle has started")
    print("  - All 4 towers are visible")
    print("  - Towers are at full HP (bars visible)")
    exit(1)

if reader.calibrated:
    print("\n[SUCCESS] Calibration complete!")
    
    frame = reader.capture_screen()
    
    # Process each bar at initial state
    bars = [
        (reader.opponent_left_bar, "OPP_LEFT", 'red'),
        (reader.opponent_right_bar, "OPP_RIGHT", 'red'),
        (reader.player_left_bar, "PLAYER_LEFT", 'blue'),
        (reader.player_right_bar, "PLAYER_RIGHT", 'blue')
    ]
    
    def save_bar_debug(frame, bar_info, label, color_type, suffix=""):
        x, y, w, h, baseline_width = bar_info
        
        # Extract the bar region
        bar_region = frame[y:y+h, x:x+w]
        
        # Convert to HSV and create mask
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        
        if color_type == 'red':
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # blue
            lower_blue = np.array([100, 120, 70])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find colored columns
        column_sums = np.sum(mask, axis=0)
        colored_columns = np.where(column_sums > 0)[0]
        
        if len(colored_columns) > 0:
            leftmost = colored_columns[0]
            rightmost = colored_columns[-1]
            detected_width = rightmost - leftmost + 1
        else:
            leftmost = rightmost = detected_width = 0
        
        # Save images
        cv2.imwrite(f"debug_{label}{suffix}_original.png", bar_region)
        cv2.imwrite(f"debug_{label}{suffix}_mask.png", mask)
        
        # Create visualization with vertical lines at leftmost/rightmost
        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if len(colored_columns) > 0:
            cv2.line(vis, (leftmost, 0), (leftmost, h), (0, 255, 0), 2)  # Green = leftmost
            cv2.line(vis, (rightmost, 0), (rightmost, h), (0, 0, 255), 2)  # Red = rightmost
        cv2.imwrite(f"debug_{label}{suffix}_width_lines.png", vis)
        
        return leftmost, rightmost, detected_width, w
    
    # Save initial state
    print("\n[INITIAL STATE]")
    for bar_info, label, color_type in bars:
        leftmost, rightmost, detected_width, region_width = save_bar_debug(frame, bar_info, label, color_type, "_initial")
        baseline_width = bar_info[4]
        print(f"{label}: width={detected_width}px, baseline={baseline_width}px, HP={detected_width/baseline_width*100:.1f}%")
    
    # Wait 20 seconds
    print("\n[INFO] Waiting 20 seconds for damage to occur...")
    for i in range(20, 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    print("\n")
    
    # Capture again after damage
    frame2 = reader.capture_screen()
    
    print("\n[AFTER 20 SECONDS]")
    for bar_info, label, color_type in bars:
        leftmost, rightmost, detected_width, region_width = save_bar_debug(frame2, bar_info, label, color_type, "_after20s")
        baseline_width = bar_info[4]
        hp_percent = (detected_width / baseline_width * 100.0) if baseline_width > 0 else 0
        print(f"{label}: width={detected_width}px, baseline={baseline_width}px, HP={hp_percent:.1f}%")
    
    print("\n[INFO] Debug images saved:")
    print("  *_initial_* = state at calibration")
    print("  *_after20s_* = state after 20 seconds")
    print("  Check if the width decreased as HP decreased!")
else:
    print("[ERROR] Calibration failed!")
