"""
Show which bars are detected as left/right during calibration.
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

print("\n[INFO] Calibrating...")
if reader.calibrate_hp_bars():
    print("\n[SUCCESS] Calibration complete!")
    
    # Capture frame and draw boxes
    frame = reader.capture_screen()
    
    # Draw and label each detected bar
    bars = [
        (reader.opponent_left_bar, "OPPONENT_LEFT", (0, 0, 255)),
        (reader.opponent_right_bar, "OPPONENT_RIGHT", (0, 100, 255)),
        (reader.player_left_bar, "PLAYER_LEFT", (255, 0, 0)),
        (reader.player_right_bar, "PLAYER_RIGHT", (255, 100, 0))
    ]
    
    for bar_info, label, color in bars:
        x, y, w, h, baseline_px = bar_info
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, color, 2)
        cv2.putText(frame, f"{baseline_px}px", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, color, 1)
        
        print(f"\n{label}:")
        print(f"  Position: ({x}, {y})")
        print(f"  Size: {w}x{h}")
        print(f"  Baseline: {baseline_px} pixels")
    
    # Save image
    cv2.imwrite("calibration_visualization.png", frame)
    print("\n[INFO] Saved visualization to: calibration_visualization.png")
    print("[INFO] Check the image to verify left/right are correct!")
    print("\nColor key:")
    print("  Dark Red = OPPONENT_LEFT")
    print("  Light Red = OPPONENT_RIGHT") 
    print("  Dark Blue = PLAYER_LEFT")
    print("  Light Blue = PLAYER_RIGHT")
else:
    print("[ERROR] Calibration failed!")
