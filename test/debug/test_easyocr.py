"""
Test EasyOCR on the saved samples - it's better with stylized fonts.
"""

import cv2
import os
import sys

try:
    import easyocr
    print("[INFO] EasyOCR loaded successfully")
except ImportError:
    print("[ERROR] EasyOCR not installed!")
    print("Install with: pip install easyocr")
    sys.exit(1)

def main():
    sample_dir = "ocr_samples"
    
    if not os.path.exists(sample_dir):
        print(f"[ERROR] Directory '{sample_dir}' not found!")
        return
    
    print("=" * 70)
    print("Initializing EasyOCR (this may take a moment on first run)...")
    print("=" * 70)
    
    # Initialize EasyOCR reader (English only, GPU if available)
    reader = easyocr.Reader(['en'], gpu=False)
    print("[INFO] EasyOCR reader initialized")
    
    # Find all sample files
    files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    
    # Group by capture
    captures = {}
    for f in files:
        parts = f.replace('.png', '').split('_')
        if len(parts) >= 3:
            name = parts[0]
            capture = parts[1]
            version = '_'.join(parts[2:])
            
            key = f"{name}_{capture}"
            if key not in captures:
                captures[key] = {}
            captures[key][version] = f
    
    print("\n" + "=" * 70)
    print("Testing EasyOCR on Saved Samples")
    print("=" * 70)
    
    for capture_key in sorted(captures.keys()):
        versions = captures[capture_key]
        print(f"\n{'=' * 70}")
        print(f"[{capture_key.upper()}]")
        print(f"{'=' * 70}")
        
        # Test each version
        for version_name in ['original', 'gray', 'scaled5x', 'inverted', 'binary']:
            if version_name not in versions:
                continue
            
            file_path = os.path.join(sample_dir, versions[version_name])
            print(f"\n  Testing: {version_name}")
            
            try:
                # Read image
                img = cv2.imread(file_path)
                
                # EasyOCR - try with detail and without
                result_detail = reader.readtext(img, detail=1)
                result_simple = reader.readtext(img, detail=0)
                
                if result_detail:
                    print(f"    Detailed results:")
                    for bbox, text, conf in result_detail:
                        print(f"      '{text}' (confidence: {conf:.2f})")
                else:
                    print(f"    No text detected")
                
                if result_simple:
                    print(f"    Simple: {result_simple}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
    
    print(f"\n{'=' * 70}")
    print("[SUMMARY]")
    print("If EasyOCR worked, we'll switch to it in troop_iden.py")
    print("If not, we'll need template matching or skip OCR entirely")
    print("=" * 70)

if __name__ == "__main__":
    main()
