"""
Interactive tool to position OCR ROIs (timer, HP, elixir).
Shows ROI boxes on screen capture, allows adjustment with keyboard.
Press 's' to save final positions to console.
"""

import cv2
import numpy as np
import mss
import time

# ROI state - synced with troop_iden.py current positions
roi_configs = {
    'timer': {'x': 0.815, 'y': 0.030, 'w': 0.190, 'h': 0.070, 'color': (0, 255, 0)},  # Green
    'left_hp': {'x': 0.230, 'y': 0.145, 'w': 0.120, 'h': 0.055, 'color': (255, 0, 0)},  # Blue - adjusted
    'right_hp': {'x': 0.730, 'y': 0.145, 'w': 0.120, 'h': 0.055, 'color': (255, 0, 0)},  # Blue - adjusted
    'elixir': {'x': 0.46, 'y': 0.80, 'w': 0.08, 'h': 0.04, 'color': (255, 255, 0)},  # Cyan
}

selected_roi = 'timer'
step = 0.01  # Movement step

def draw_rois(frame, screen_w, screen_h):
    """Draw all ROI boxes on frame."""
    overlay = frame.copy()
    
    for name, cfg in roi_configs.items():
        x = int(cfg['x'] * screen_w)
        y = int(cfg['y'] * screen_h)
        w = int(cfg['w'] * screen_w)
        h = int(cfg['h'] * screen_h)
        
        # Highlight selected ROI
        thickness = 3 if name == selected_roi else 2
        alpha = 0.3 if name == selected_roi else 0.1
        
        # Draw filled rectangle
        cv2.rectangle(overlay, (x, y), (x + w, y + h), cfg['color'], -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + w, y + h), cfg['color'], thickness)
        
        # Draw label
        label = f"{name} [{cfg['x']:.3f}, {cfg['y']:.3f}]"
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, cfg['color'], 2)
    
    return frame

def print_instructions():
    print("\n=== OCR ROI Position Tuner ===")
    print("Selected ROI: Use '1'=timer, '2'=left_hp, '3'=right_hp, '4'=elixir")
    print("Movement: Arrow keys OR i/j/k/l (up/left/down/right)")
    print("Size: 'w'=wider, 'n'=narrower, 't'=taller, 's'=shorter")
    print("Save: 'p' to print positions")
    print("Quit: 'q' or ESC")
    print("=" * 40)

def main():
    print_instructions()
    
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
        
        global selected_roi
        
        while True:
            # Capture screen
            img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Draw ROIs
            frame = draw_rois(frame, screen_w, screen_h)
            
            # Show selected ROI
            cv2.putText(frame, f"Selected: {selected_roi}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("OCR ROI Tuner", frame)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q') or key == 27:  # ESC
                break
            elif key == ord('1'):
                selected_roi = 'timer'
            elif key == ord('2'):
                selected_roi = 'left_hp'
            elif key == ord('3'):
                selected_roi = 'right_hp'
            elif key == ord('4'):
                selected_roi = 'elixir'
            elif key == ord('p'):
                print("\n=== Current ROI Positions ===")
                for name, cfg in roi_configs.items():
                    print(f"{name}: x={cfg['x']:.3f}, y={cfg['y']:.3f}, w={cfg['w']:.3f}, h={cfg['h']:.3f}")
            
            # Movement controls
            cfg = roi_configs[selected_roi]
            
            # Arrow keys (try multiple key codes for cross-platform compatibility)
            if key in [82, 0, ord('i')]:  # Up arrow or 'i'
                cfg['y'] = max(0, cfg['y'] - step)
                print(f"Move up: y={cfg['y']:.3f}")
            elif key in [84, 1, ord('k')]:  # Down arrow or 'k'
                cfg['y'] = min(1 - cfg['h'], cfg['y'] + step)
                print(f"Move down: y={cfg['y']:.3f}")
            elif key in [81, 2, ord('j')]:  # Left arrow or 'j'
                cfg['x'] = max(0, cfg['x'] - step)
                print(f"Move left: x={cfg['x']:.3f}")
            elif key in [83, 3, ord('l')]:  # Right arrow or 'l'
                cfg['x'] = min(1 - cfg['w'], cfg['x'] + step)
                print(f"Move right: x={cfg['x']:.3f}")
            elif key == ord('w'):  # Wider
                cfg['w'] = min(0.5, cfg['w'] + step)
                print(f"Wider: w={cfg['w']:.3f}")
            elif key == ord('n'):  # Narrower
                cfg['w'] = max(0.05, cfg['w'] - step)
                print(f"Narrower: w={cfg['w']:.3f}")
            elif key == ord('t'):  # Taller
                cfg['h'] = min(0.3, cfg['h'] + step)
                print(f"Taller: h={cfg['h']:.3f}")
            elif key == ord('s'):  # Shorter
                cfg['h'] = max(0.02, cfg['h'] - step)
                print(f"Shorter: h={cfg['h']:.3f}")
            elif key != 255 and key not in [ord('q'), 27, ord('1'), ord('2'), ord('3'), ord('4'), ord('p')]:
                # Debug: print unknown key codes
                print(f"Key pressed: {key}")
    
    cv2.destroyAllWindows()
    
    print("\n=== Final ROI Positions ===")
    for name, cfg in roi_configs.items():
        print(f"{name}: x={cfg['x']:.3f}, y={cfg['y']:.3f}, w={cfg['w']:.3f}, h={cfg['h']:.3f}")

if __name__ == "__main__":
    main()
