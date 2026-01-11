"""
Debug HP bar detection by saving captured regions and color masks.
"""

import cv2
import numpy as np
from ocr_reader import ClashOCRReader

reader = ClashOCRReader()

# Capture all HP bar regions
print("[INFO] Capturing HP bar regions...")
opp_left = reader.capture_hp_bar_region(reader.opponent_left_hp_region)
opp_right = reader.capture_hp_bar_region(reader.opponent_right_hp_region)
player_left = reader.capture_hp_bar_region(reader.player_left_hp_region)
player_right = reader.capture_hp_bar_region(reader.player_right_hp_region)

# Save raw captures
cv2.imwrite("debug_opp_left_raw.png", opp_left)
cv2.imwrite("debug_opp_right_raw.png", opp_right)
cv2.imwrite("debug_player_left_raw.png", player_left)
cv2.imwrite("debug_player_right_raw.png", player_right)

# Process opponent left (red) bar
hsv = cv2.cvtColor(opp_left, cv2.COLOR_BGR2HSV)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)
cv2.imwrite("debug_opp_left_red_mask.png", red_mask)

# Process player left (blue) bar
hsv = cv2.cvtColor(player_left, cv2.COLOR_BGR2HSV)
lower_blue = np.array([100, 100, 100])
upper_blue = np.array([130, 255, 255])
blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
cv2.imwrite("debug_player_left_blue_mask.png", blue_mask)

# Show HSV values of center pixel for each bar
def show_center_pixel_info(img, name):
    h, w = img.shape[:2]
    center_pixel = img[h//2, w//2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    center_hsv = hsv[h//2, w//2]
    print(f"\n{name}:")
    print(f"  BGR: {center_pixel}")
    print(f"  HSV: {center_hsv}")
    print(f"  Size: {w}x{h}")

show_center_pixel_info(opp_left, "Opponent Left")
show_center_pixel_info(opp_right, "Opponent Right")
show_center_pixel_info(player_left, "Player Left")
show_center_pixel_info(player_right, "Player Right")

print("\n[INFO] Debug images saved. Check the PNG files to see what's being captured.")
print("[INFO] The *_mask.png files show what colors were detected (white = detected).")
