"""
Save the ROI images to disk so we can examine them manually
and try different OCR approaches offline.
"""

import cv2
import numpy as np
import mss
import os

# ROI positions
roi_configs = {
    'timer': {'x': 0.815, 'y': 0.030, 'w': 0.190, 'h': 0.070},
    'left_hp': {'x': 0.230, 'y': 0.145, 'w': 0.120, 'h': 0.055},
    'right_hp': {'x': 0.730, 'y': 0.145, 'w': 0.120, 'h': 0.055},
}

def main():
    print("=== Save ROI Images for Manual Testing ===")
    print("This will save the ROI images to disk")
    print("Press SPACE to capture, 'q' to quit")
    print("=" * 50)
    
    # Create output directory
    output_dir = "ocr_samples"
    os.makedirs(output_dir, exist_ok=True)
    
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
        
        capture_count = 0
        
        while True:
            # Capture
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Draw ROIs on preview
            preview = frame.copy()
            for name, cfg in roi_configs.items():
                x = int(cfg['x'] * screen_w)
                y = int(cfg['y'] * screen_h)
                w = int(cfg['w'] * screen_w)
                h = int(cfg['h'] * screen_h)
                
                color = (0, 255, 0) if name == 'timer' else (255, 0, 0)
                cv2.rectangle(preview, (x, y), (x+w, y+h), color, 2)
                cv2.putText(preview, name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, color, 2)
            
            cv2.putText(preview, "Press SPACE to capture", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("ROI Preview", preview)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):  # Space bar
                capture_count += 1
                print(f"\n[Capture {capture_count}]")
                
                for name, cfg in roi_configs.items():
                    x = int(cfg['x'] * screen_w)
                    y = int(cfg['y'] * screen_h)
                    w = int(cfg['w'] * screen_w)
                    h = int(cfg['h'] * screen_h)
                    
                    # Extract ROI
                    roi_color = frame[y:y+h, x:x+w]
                    roi_gray = frame_gray[y:y+h, x:x+w]
                    
                    # Save multiple versions
                    base_name = f"{output_dir}/{name}_capture{capture_count}"
                    
                    # Original
                    cv2.imwrite(f"{base_name}_original.png", roi_color)
                    
                    # Grayscale
                    cv2.imwrite(f"{base_name}_gray.png", roi_gray)
                    
                    # Upscaled 5x
                    scaled = cv2.resize(roi_gray, None, fx=5.0, fy=5.0, 
                                       interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(f"{base_name}_scaled5x.png", scaled)
                    
                    # Binary threshold
                    _, binary = cv2.threshold(scaled, 0, 255, 
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cv2.imwrite(f"{base_name}_binary.png", binary)
                    
                    # Inverted
                    inverted = 255 - binary
                    cv2.imwrite(f"{base_name}_inverted.png", inverted)
                    
                    print(f"  {name}: Saved 5 versions")
                
                print(f"  All saved to '{output_dir}/' directory")
    
    cv2.destroyAllWindows()
    print(f"\n[Done] Captured {capture_count} samples")
    print(f"Check '{output_dir}/' directory for images")
    print("\nYou can:")
    print("1. Examine images manually to see if text is readable")
    print("2. Try online OCR tools to see if they work better")
    print("3. Check if the font is too stylized for Tesseract")

if __name__ == "__main__":
    main()
