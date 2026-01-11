"""
Test Tesseract on the saved sample images to see which preprocessing works best.
"""

import cv2
import os
import sys

try:
    import pytesseract
    tesseract_path = r"C:\Users\jerem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except Exception as e:
    print(f"[ERROR] Tesseract setup failed: {e}")
    sys.exit(1)

def test_image(image_path, roi_type):
    """Test OCR on a single image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return None
    
    # Config based on ROI type
    if 'timer' in roi_type:
        configs = [
            ('digits+colon, PSM 7', '--psm 7 -c tessedit_char_whitelist=0123456789:'),
            ('digits+colon, PSM 8', '--psm 8 -c tessedit_char_whitelist=0123456789:'),
            ('digits+colon, PSM 13', '--psm 13 -c tessedit_char_whitelist=0123456789:'),
            ('no whitelist, PSM 7', '--psm 7'),
        ]
    else:  # HP
        configs = [
            ('digits only, PSM 7', '--psm 7 -c tessedit_char_whitelist=0123456789'),
            ('digits only, PSM 8', '--psm 8 -c tessedit_char_whitelist=0123456789'),
            ('digits only, PSM 13', '--psm 13 -c tessedit_char_whitelist=0123456789'),
            ('no whitelist, PSM 7', '--psm 7'),
        ]
    
    results = []
    for name, config in configs:
        try:
            text = pytesseract.image_to_string(img, config=config).strip()
            results.append((name, text))
        except Exception as e:
            results.append((name, f"ERROR: {e}"))
    
    return results

def main():
    sample_dir = "ocr_samples"
    
    if not os.path.exists(sample_dir):
        print(f"[ERROR] Directory '{sample_dir}' not found!")
        print("Run save_ocr_samples.py first")
        return
    
    # Find all sample files
    files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    
    if not files:
        print(f"[ERROR] No PNG files found in '{sample_dir}'")
        return
    
    # Group files by capture
    captures = {}
    for f in files:
        # Format: name_capture#_version.png
        parts = f.replace('.png', '').split('_')
        if len(parts) >= 3:
            name = parts[0]  # timer, left, right
            capture = parts[1]  # capture1, capture2, etc
            version = '_'.join(parts[2:])  # original, gray, scaled5x, etc
            
            key = f"{name}_{capture}"
            if key not in captures:
                captures[key] = {}
            captures[key][version] = f
    
    print("=" * 70)
    print("Testing Tesseract on Saved Samples")
    print("=" * 70)
    
    for capture_key in sorted(captures.keys()):
        versions = captures[capture_key]
        print(f"\n{'=' * 70}")
        print(f"[{capture_key.upper()}]")
        print(f"{'=' * 70}")
        
        # Test each version
        for version_name in ['inverted', 'binary', 'scaled5x', 'gray']:
            if version_name not in versions:
                continue
            
            file_path = os.path.join(sample_dir, versions[version_name])
            print(f"\n  Version: {version_name}")
            print(f"  File: {versions[version_name]}")
            
            results = test_image(file_path, capture_key)
            
            if results:
                for config_name, text in results:
                    print(f"    {config_name:25} → '{text}'")
            else:
                print(f"    ERROR: Could not read image")
    
    print(f"\n{'=' * 70}")
    print("[SUMMARY]")
    print("Look for any successful readings above.")
    print("If all are empty/wrong, Tesseract cannot read this font.")
    print("We may need to switch to EasyOCR or template matching.")
    print("=" * 70)

if __name__ == "__main__":
    main()
