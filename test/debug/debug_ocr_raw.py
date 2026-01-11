"""
Debug tool to see what OCR is actually reading from each ROI.
Shows the raw ROI image, preprocessed image, and OCR output.
"""

import cv2
import numpy as np
import mss
import sys
import os

# Add tesseract path
try:
    import pytesseract
    tesseract_path = r"C:\Users\jerem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"[INFO] Using Tesseract at: {tesseract_path}")
except Exception as e:
    print(f"[ERROR] Tesseract setup failed: {e}")
    sys.exit(1)

# ROI positions (from troop_iden.py)
roi_configs = {
    'timer': {'x': 0.815, 'y': 0.030, 'w': 0.190, 'h': 0.070},
    'left_hp': {'x': 0.230, 'y': 0.150, 'w': 0.120, 'h': 0.040},
    'right_hp': {'x': 0.730, 'y': 0.150, 'w': 0.120, 'h': 0.040},
}

def preprocess_timer(roi_image):
    """Preprocess timer ROI (from troop_iden.py)."""
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image
    
    # Upscale 2.5x
    scaled = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    
    # Threshold
    _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Morphological close
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return processed

def preprocess_hp(roi_image):
    """Preprocess HP ROI (from troop_iden.py)."""
    if len(roi_image.shape) == 3:
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_image
    
    # Upscale 3x
    scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    
    return processed

def main():
    print("=== OCR Debug Tool ===")
    print("Press 'q' to quit")
    print("Shows: Raw ROI | Preprocessed | OCR Output")
    print("=" * 50)
    
    with mss.mss() as sct:
        # Capture right 30% of screen (MUST MATCH troop_iden.py)
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
        
        print(f"[INFO] Capture region: {capture_width}x{monitor['height']}")
        print(f"[INFO] Screen dimensions for ROI: {screen_w}x{screen_h}")
        
        while True:
            # Capture screen
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            print("\n" + "=" * 70)
            print(f"[DEBUG] Frame shape: {frame_gray.shape}")
            
            # Process each ROI
            for name, cfg in roi_configs.items():
                x = int(cfg['x'] * screen_w)
                y = int(cfg['y'] * screen_h)
                w = int(cfg['w'] * screen_w)
                h = int(cfg['h'] * screen_h)
                
                # Check bounds
                x_end = x + w
                y_end = y + h
                
                print(f"\n[{name.upper()}] Requested: x={x}, y={y}, w={w}, h={h} (x_end={x_end}, y_end={y_end})")
                print(f"  Normalized: x={cfg['x']:.3f}, y={cfg['y']:.3f}, w={cfg['w']:.3f}, h={cfg['h']:.3f}")
                print(f"  Sum: x+w={cfg['x']+cfg['w']:.3f}, y+h={cfg['y']+cfg['h']:.3f}")
                
                # Clip to frame bounds
                if x_end > frame_gray.shape[1]:
                    print(f"  WARNING: ROI extends beyond frame width! {x_end} > {frame_gray.shape[1]}")
                    w = frame_gray.shape[1] - x
                if y_end > frame_gray.shape[0]:
                    print(f"  WARNING: ROI extends beyond frame height! {y_end} > {frame_gray.shape[0]}")
                    h = frame_gray.shape[0] - y
                
                # Extract ROI
                roi = frame_gray[y:y+h, x:x+w]
                
                if roi.size == 0:
                    print(f"[{name.upper()}] ROI is empty!")
                    continue
                
                # Preprocess based on type
                if name == 'timer':
                    processed = preprocess_timer(roi)
                    config = '--psm 7 -c tessedit_char_whitelist=0123456789:'
                else:  # HP
                    processed = preprocess_hp(roi)
                    config = '--psm 7 -c tessedit_char_whitelist=0123456789'
                
                # OCR
                try:
                    text = pytesseract.image_to_string(processed, config=config).strip()
                except Exception as e:
                    text = f"ERROR: {e}"
                
                # Display
                print(f"\n[{name.upper()}]")
                print(f"  Position: x={x}, y={y}, w={w}, h={h}")
                print(f"  Raw shape: {roi.shape}")
                print(f"  Processed shape: {processed.shape}")
                print(f"  OCR Result: '{text}'")
                
                # Show windows
                cv2.imshow(f"{name} - Raw", cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
                cv2.imshow(f"{name} - Preprocessed", cv2.resize(processed, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
            
            # Check for quit
            if cv2.waitKey(2000) & 0xFF == ord('q'):
                break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
