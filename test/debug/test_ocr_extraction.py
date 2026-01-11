"""
OCR Extraction Test Utility
Test and visualize timer, HP, and elixir OCR extraction.
"""

import cv2
import numpy as np
import sys
import time

# Add parent directory to path
sys.path.insert(0, '.')

from troop_iden import CardDetector


def main():
    """Test OCR extraction with live visualization."""
    print("=" * 60)
    print("OCR Extraction Test")
    print("=" * 60)
    print()
    print("This will test timer, tower HP, and elixir phase OCR.")
    print("Press 'q' to quit")
    print("Press 'd' to toggle DEBUG_OCR mode")
    print()
    
    # Create detector with OCR debug enabled
    detector = CardDetector()
    detector.DEBUG_OCR = True
    
    print(f"[TEST] Match duration: {detector.match_duration}s")
    print(f"[TEST] 2x elixir starts at: {detector.double_elixir_start}s remaining")
    print(f"[TEST] Mode profile: {detector.mode_profile}")
    print()
    
    frame_count = 0
    last_print = time.time()
    
    try:
        while True:
            # Capture frame
            frame_color, frame_gray = detector.capture_screen()
            
            # Update game state (includes OCR)
            detector.update_game_state(frame_gray)
            
            # Print state every 2 seconds
            if time.time() - last_print >= 2.0:
                state = detector.get_clock_hp_elixir()
                print(f"[STATE] Time: {state['time']}s | " +
                      f"Left HP: {state['left_tower_hp']} | " +
                      f"Right HP: {state['right_tower_hp']} | " +
                      f"Elixir: {state['elixir_phase']}")
                last_print = time.time()
            
            # Visualize ROIs on frame
            vis_frame = frame_color.copy()
            
            # Draw ROI boxes
            for roi, color, label in [
                (detector.timer_roi, (0, 255, 255), "Timer"),
                (detector.left_tower_hp_roi, (0, 255, 0), "Left HP"),
                (detector.right_tower_hp_roi, (0, 255, 0), "Right HP"),
                (detector.elixir_indicator_roi, (255, 0, 255), "Elixir")
            ]:
                cv2.rectangle(vis_frame, (roi.x, roi.y),
                             (roi.x + roi.width, roi.y + roi.height),
                             color, 2)
                cv2.putText(vis_frame, label, (roi.x, roi.y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Show visualization
            cv2.imshow("OCR ROIs", vis_frame)
            
            # Keyboard input
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                detector.DEBUG_OCR = not detector.DEBUG_OCR
                print(f"[TEST] DEBUG_OCR = {detector.DEBUG_OCR}")
            
            frame_count += 1
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n[TEST] Stopped by user")
    finally:
        cv2.destroyAllWindows()
        
        # Print final state
        state = detector.get_clock_hp_elixir()
        print("\n" + "=" * 60)
        print("FINAL STATE:")
        print(f"  Match time: {state['time']}s")
        print(f"  Left tower HP: {state['left_tower_hp']}")
        print(f"  Right tower HP: {state['right_tower_hp']}")
        print(f"  Elixir phase: {state['elixir_phase']}")
        print("=" * 60)


if __name__ == "__main__":
    main()
