"""
Test HP bar calibration to debug pixel counting.
"""

from ocr_reader import ClashOCRReader
import time

reader = ClashOCRReader()

print("[INFO] Attempting calibration...")
if reader.calibrate_hp_bars():
    print("\n[SUCCESS] Calibration complete!")
    print("\nWaiting 2 seconds, then reading HP bars...")
    time.sleep(2)
    
    # Read HP bars
    hp_data = reader.read_tower_hp_bars()
    
    print("\n[RESULTS]")
    print(f"  Opponent Left:  {hp_data['opponent_left_hp']:.1f}%")
    print(f"  Opponent Right: {hp_data['opponent_right_hp']:.1f}%")
    print(f"  Player Left:    {hp_data['player_left_hp']:.1f}%")
    print(f"  Player Right:   {hp_data['player_right_hp']:.1f}%")
    
    # Now manually check the pixel counts
    print("\n[DETAILED PIXEL COUNTS]")
    frame = reader.capture_screen()
    
    for name, bar_info, color in [
        ("Opponent Left", reader.opponent_left_bar, "red"),
        ("Opponent Right", reader.opponent_right_bar, "red"),
        ("Player Left", reader.player_left_bar, "blue"),
        ("Player Right", reader.player_right_bar, "blue")
    ]:
        import cv2
        import numpy as np
        x, y, w, h, baseline = bar_info
        bar_region = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        
        if color == 'red':
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            lower_blue = np.array([100, 120, 70])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        current = np.count_nonzero(mask)
        total = mask.size
        print(f"  {name}: baseline={baseline}, current={current}, total_pixels={total}, %={current/baseline*100:.1f}%")
else:
    print("[ERROR] Calibration failed!")
