"""
Visual tool to show automatically detected HP bar positions.
Captures the screen and draws boxes around detected HP bars.
"""

import cv2
import numpy as np
from ocr_reader import ClashOCRReader

reader = ClashOCRReader()

# Capture full screen area
frame = reader.capture_screen()

# Use automatic detection to find bars
print("[INFO] Detecting HP bars automatically...")

opponent_bars = reader.find_hp_bars_by_color(frame, 'red', search_top=True)
player_bars = reader.find_hp_bars_by_color(frame, 'blue', search_top=False)

print(f"[INFO] Found {len(opponent_bars)} opponent (red) bars")
print(f"[INFO] Found {len(player_bars)} player (blue) bars")

# Draw detected bars
vis_frame = frame.copy()

# Draw opponent bars in red
for i, (x, y, w, h, fill_percent) in enumerate(opponent_bars):
    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
    label = f"OPP_{i+1}: {fill_percent:.1f}%"
    cv2.putText(vis_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (0, 0, 255), 2)
    print(f"  Opponent bar {i+1}: pos=({x},{y}) size=({w}x{h}) fill={fill_percent:.1f}%")

# Draw player bars in blue
for i, (x, y, w, h, fill_percent) in enumerate(player_bars):
    cv2.rectangle(vis_frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
    label = f"PLAYER_{i+1}: {fill_percent:.1f}%"
    cv2.putText(vis_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
               0.7, (255, 0, 0), 2)
    print(f"  Player bar {i+1}: pos=({x},{y}) size=({w}x{h}) fill={fill_percent:.1f}%")

# Save the visualization
cv2.imwrite("hp_bar_auto_detected.png", vis_frame)

print("\n[INFO] Visualization saved to: hp_bar_auto_detected.png")
print("Open this image to see if the detected bars are correct.")
print("Red boxes = opponent (top), Blue boxes = player (bottom)")
