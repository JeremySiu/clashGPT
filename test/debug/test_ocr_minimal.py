"""
Minimal OCR test - uses simple preprocessing to isolate issues.
Shows what Tesseract reads with different preprocessing approaches.
"""

import cv2
import numpy as np
import mss
import sys
import os

try:
    import pytesseract
    tesseract_path = r"C:\Users\jerem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[INFO] Using Tesseract at: {tesseract_path}")
except Exception as e:
    print(f"[ERROR] Tesseract setup failed: {e}")
    sys.exit(1)

# ROI positions - SYNCED with tune_ocr_positions.py
roi_configs = {
    'timer': {'x': 0.815, 'y': 0.030, 'w': 0.190, 'h': 0.070},
    'left_hp': {'x': 0.230, 'y': 0.145, 'w': 0.120, 'h': 0.055},  # Adjusted
    'right_hp': {'x': 0.730, 'y': 0.145, 'w': 0.120, 'h': 0.055},  # Adjusted
}

def simple_preprocess(roi, upscale=5.0):
    """Minimal preprocessing - aggressive upscale for better OCR."""
    if len(roi.shape) == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi
    
    # Aggressive upscale (5x instead of 3x)
    scaled = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    
    # Try both normal and inverted
    _, binary = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = 255 - binary
    
    return scaled, binary, inverted

def main():
    print("=== Minimal OCR Test ===")
    print("Testing different preprocessing approaches")
    print("Press 'q' to quit")
    print("=" * 50)
    
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
        
        while True:
            # Capture
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            print("\n" + "=" * 70)
            
            for name, cfg in roi_configs.items():
                x = int(cfg['x'] * screen_w)
                y = int(cfg['y'] * screen_h)
                w = int(cfg['w'] * screen_w)
                h = int(cfg['h'] * screen_h)
                
                # Extract ROI
                roi = frame_gray[y:y+h, x:x+w]
                
                if roi.size == 0:
                    print(f"[{name}] ROI empty!")
                    continue
                
                # Try different preprocessing
                scaled, binary, inverted = simple_preprocess(roi, upscale=5.0)
                
                # Try multiple PSM modes
                configs = [
                    ('PSM 7 (single line)', '--psm 7'),
                    ('PSM 8 (single word)', '--psm 8'),
                    ('PSM 13 (raw line)', '--psm 13'),
                ]
                
                if name == 'timer':
                    whitelist = ' -c tessedit_char_whitelist=0123456789:'
                else:
                    whitelist = ' -c tessedit_char_whitelist=0123456789'
                
                print(f"\n[{name.upper()}]")
                print(f"  ROI shape: {roi.shape} → Scaled: {scaled.shape}")
                
                # Try OCR on inverted (white text) with different PSM modes
                for psm_name, psm_config in configs:
                    try:
                        text = pytesseract.image_to_string(inverted, config=psm_config + whitelist).strip()
                        print(f"  {psm_name}: '{text}'")
                    except Exception as e:
                        print(f"  {psm_name}: ERROR - {e}")
                
                # Show windows
                cv2.imshow(f"{name} - Raw", cv2.resize(roi, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
                cv2.imshow(f"{name} - Binary", cv2.resize(binary, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
                cv2.imshow(f"{name} - Inverted", cv2.resize(inverted, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
