"""
Debug EasyOCR to see what it's detecting in the timer ROI.
Saves the ROI image and all detected text with bounding boxes to disk.
"""

import cv2
import numpy as np
import mss
import sys
import os
from datetime import datetime

try:
    import easyocr
    print("[INFO] EasyOCR loaded")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    print("[INFO] Reader initialized")
except Exception as e:
    print(f"[ERROR] {e}")
    sys.exit(1)

# Timer ROI from troop_iden.py
timer_roi = {'x': 0.815, 'y': 0.030, 'w': 0.190, 'h': 0.070}

def main():
    print("=== EasyOCR Timer Debug ===")
    print("Press CTRL+C to quit, or it will auto-save images every 2 seconds")
    print("=" * 50)
    
    # Create output directory
    output_dir = "debug_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[INFO] Created output directory: {output_dir}")
    
    with mss.mss() as sct:
        # Capture right 30% of screen
        monitors = sct.monitors
        full_screen = monitors[1]
        
        capture_width = int(full_screen["width"] * 0.3)
        capture_x = full_screen["left"] + (full_screen["width"] - capture_width)
        
        monitor = {
            "top": full_screen["top"],
            "left": capture_x,
            "width": capture_width,
            "height": full_screen["height"]
        }
        
        screen_w = monitor["width"]
        screen_h = monitor["height"]
        
        x = int(timer_roi['x'] * screen_w)
        y = int(timer_roi['y'] * screen_h)
        w = int(timer_roi['w'] * screen_w)
        h = int(timer_roi['h'] * screen_h)
        
        print(f"[INFO] Timer ROI: x={x}, y={y}, w={w}, h={h}")
        print(f"[INFO] Screen: {screen_w}x{screen_h}")
        print(f"[INFO] Images will be saved to: {output_dir}/")
        print()
        
        frame_count = 0
        
        try:
            while True:
                # Capture
                img = np.array(sct.grab(monitor))
                frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Extract timer ROI
                roi = frame[y:y+h, x:x+w]
                
                # Upscale
                scaled = cv2.resize(roi, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
                
                # Run EasyOCR with detail
                results = reader.readtext(scaled, detail=1)
                
                # Draw results on scaled image
                display = scaled.copy()
                
                print(f"\n[Frame {frame_count}] Found {len(results)} text regions:")
                for i, (bbox, text, conf) in enumerate(results):
                    print(f"  {i+1}. '{text}' (conf: {conf:.2f})")
                    
                    # Draw bounding box
                    pts = np.array(bbox, dtype=np.int32)
                    cv2.polylines(display, [pts], True, (0, 255, 0), 2)
                    
                    # Draw text
                    cv2.putText(display, text, tuple(pts[0]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{output_dir}/timer_debug_{timestamp}_{frame_count:04d}.png"
                cv2.imwrite(filename, display)
                print(f"[SAVED] {filename}")
                
                frame_count += 1
                
                # Wait 2 seconds between captures
                import time
                time.sleep(2)
                
        except KeyboardInterrupt:
            print("\n\n[INFO] Stopped by user")
            print(f"[INFO] Total frames captured: {frame_count}")
            print(f"[INFO] Images saved in: {output_dir}/")

if __name__ == "__main__":
    main()
