"""
Hand Detection Test Utility
Visualizes hand slot changes and helps tune thresholds.
Shows difference between card selection (float animation) vs card play (replacement).
"""

import cv2
import numpy as np
import mss
import time
from collections import deque


class HandDetectionTester:
    """Test utility for hand detection tuning."""
    
    def __init__(self):
        """Initialize the tester."""
        self.sct = mss.mss()
        self.setup_capture_region()
        self.setup_hand_region()
        
        self.previous_slots = []
        self.change_history = deque(maxlen=30)  # Track last 30 frames
        
        print("[TEST] Hand Detection Tester")
        print("[TEST] This will help identify card selection vs card play")
        print("[TEST] Try: 1) Clicking a card (should show small change)")
        print("[TEST]      2) Playing a card (should show large change)")
        print("[TEST] Press 'q' to quit, 's' to save current frame\n")
    
    def setup_capture_region(self):
        """Setup screen capture region (right 35% of screen)."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screen_width = monitor["width"]
            screen_height = monitor["height"]
            
            capture_width = int(screen_width * 0.35)
            capture_x = screen_width - capture_width
            
            self.monitor = {
                "top": 0,
                "left": capture_x,
                "width": capture_width,
                "height": screen_height
            }
            
            self.screen_width = capture_width
            self.screen_height = screen_height
    
    def setup_hand_region(self):
        """Define hand region and slots."""
        w, h = self.screen_width, self.screen_height
        
        hand_height = int(h * 0.12)
        hand_y = int(h * 0.88)
        hand_width = int(w * 0.8)
        hand_x = int(w * 0.1)
        
        self.hand_y1 = hand_y
        self.hand_y2 = hand_y + hand_height
        self.hand_x1 = hand_x
        self.hand_x2 = hand_x + hand_width
        
        # Define 4 slots
        slot_width = hand_width // 4
        self.slots = []
        
        for i in range(4):
            slot_x1 = hand_x + (i * slot_width)
            slot_x2 = slot_x1 + slot_width
            self.slots.append({
                'id': i,
                'x1': slot_x1,
                'x2': slot_x2,
                'y1': hand_y,
                'y2': hand_y + hand_height
            })
        
        print(f"[TEST] Hand region: y={hand_y}, height={hand_height}")
        print(f"[TEST] Slots: {len(self.slots)} slots, {slot_width}px wide each\n")
    
    def capture_screen(self):
        """Capture screen and extract hand region."""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        return frame, gray
    
    def analyze_slot_changes(self, current_gray):
        """Analyze changes in each slot with multiple thresholds."""
        current_slots = []
        
        # Extract slot images
        for slot in self.slots:
            slot_img = current_gray[slot['y1']:slot['y2'], slot['x1']:slot['x2']]
            slot_img = cv2.GaussianBlur(slot_img, (5, 5), 0)
            current_slots.append(slot_img)
        
        # Initialize on first frame
        if len(self.previous_slots) == 0:
            self.previous_slots = current_slots
            return None
        
        # Analyze each slot
        results = []
        for i, (current, previous) in enumerate(zip(current_slots, self.previous_slots)):
            if current.shape != previous.shape:
                results.append({
                    'slot': i,
                    'error': 'shape_mismatch',
                    'pixels_30': 0,
                    'pixels_50': 0,
                    'pixels_70': 0,
                    'percentage': 0.0,
                    'event_type': 'ERROR'
                })
                continue
            
            total_pixels = current.shape[0] * current.shape[1]
            
            # Test multiple thresholds
            diff = cv2.absdiff(previous, current)
            
            # Threshold 30 (low - catches everything)
            _, thresh_30 = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            pixels_30 = np.count_nonzero(thresh_30)
            
            # Threshold 50 (medium - filters selection)
            _, thresh_50 = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            pixels_50 = np.count_nonzero(thresh_50)
            
            # Threshold 70 (high - only major changes)
            _, thresh_70 = cv2.threshold(diff, 70, 255, cv2.THRESH_BINARY)
            pixels_70 = np.count_nonzero(thresh_70)
            
            percentage_50 = pixels_50 / total_pixels if total_pixels > 0 else 0.0
            
            # Classify event type
            if pixels_50 > 8000 and percentage_50 > 0.4:
                event_type = "CARD_PLAY"
            elif pixels_50 > 3000 and percentage_50 > 0.15:
                event_type = "CARD_SELECT"
            elif pixels_30 > 1000:
                event_type = "minor_change"
            else:
                event_type = "idle"
            
            results.append({
                'slot': i,
                'pixels_30': pixels_30,
                'pixels_50': pixels_50,
                'pixels_70': pixels_70,
                'percentage': percentage_50,
                'total_pixels': total_pixels,
                'event_type': event_type,
                'diff_img': thresh_50
            })
        
        self.previous_slots = current_slots
        return results
    
    def draw_visualization(self, frame, results):
        """Draw visualization overlay on frame."""
        if results is None:
            cv2.putText(frame, "INITIALIZING...", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        # Draw hand region box
        cv2.rectangle(frame, (self.hand_x1, self.hand_y1), 
                     (self.hand_x2, self.hand_y2), (255, 255, 0), 2)
        
        # Draw slot dividers and info
        for i, (slot, result) in enumerate(zip(self.slots, results)):
            # Slot divider
            if i > 0:
                cv2.line(frame, (slot['x1'], slot['y1']), 
                        (slot['x1'], slot['y2']), (255, 255, 0), 1)
            
            # Color code by event type
            if result['event_type'] == "CARD_PLAY":
                color = (0, 255, 0)  # Green
                thickness = 3
            elif result['event_type'] == "CARD_SELECT":
                color = (0, 165, 255)  # Orange
                thickness = 2
            elif result['event_type'] == "minor_change":
                color = (200, 200, 200)  # Gray
                thickness = 1
            else:
                color = (100, 100, 100)  # Dark gray
                thickness = 1
            
            # Draw slot box
            cv2.rectangle(frame, (slot['x1'], slot['y1']), 
                         (slot['x2'], slot['y2']), color, thickness)
            
            # Slot label
            label_y = slot['y1'] - 10
            cv2.putText(frame, f"Slot {i}", (slot['x1'] + 5, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw info panel
        info_y = 30
        info_x = 20
        
        cv2.putText(frame, "Hand Detection Test", (info_x, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        info_y += 30
        
        # Show results for each slot
        for result in results:
            if result['event_type'] in ['CARD_PLAY', 'CARD_SELECT', 'minor_change']:
                slot_id = result['slot']
                event = result['event_type']
                pixels = result['pixels_50']
                pct = result['percentage'] * 100
                
                if event == 'CARD_PLAY':
                    text_color = (0, 255, 0)
                elif event == 'CARD_SELECT':
                    text_color = (0, 165, 255)
                else:
                    text_color = (200, 200, 200)
                
                text = f"Slot{slot_id}: {event} - {pixels}px ({pct:.1f}%)"
                cv2.putText(frame, text, (info_x, info_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                info_y += 25
        
        # Draw legend
        legend_y = frame.shape[0] - 100
        cv2.putText(frame, "Legend:", (info_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(frame, "GREEN = Card Play (>8000px, >40%)", (info_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        legend_y += 20
        cv2.putText(frame, "ORANGE = Card Select (>3000px, >15%)", (info_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
        legend_y += 20
        cv2.putText(frame, "GRAY = Minor change", (info_x, legend_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def create_diff_view(self, results):
        """Create a view showing difference images for each slot."""
        if results is None or len(results) == 0:
            return None
        
        # Create horizontal montage of difference images
        diff_images = []
        for result in results:
            if 'diff_img' in result:
                diff = result['diff_img']
                # Resize to consistent height
                h, w = diff.shape
                target_h = 150
                scale = target_h / h
                target_w = int(w * scale)
                resized = cv2.resize(diff, (target_w, target_h))
                
                # Convert to BGR for color annotations
                colored = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
                
                # Add label
                label = f"Slot {result['slot']}"
                cv2.putText(colored, label, (5, 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add pixel count
                pixels = f"{result['pixels_50']}px"
                cv2.putText(colored, pixels, (5, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Add percentage
                pct = f"{result['percentage']*100:.1f}%"
                cv2.putText(colored, pct, (5, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                diff_images.append(colored)
        
        if diff_images:
            return np.hstack(diff_images)
        return None
    
    def run(self):
        """Run the test visualization."""
        frame_count = 0
        
        try:
            while True:
                # Capture frame
                frame, gray = self.capture_screen()
                
                # Analyze changes
                results = self.analyze_slot_changes(gray)
                
                # Draw visualization
                vis_frame = self.draw_visualization(frame.copy(), results)
                
                # Create difference view
                diff_view = self.create_diff_view(results)
                
                # Display main view
                cv2.imshow("Hand Detection Test - Main View", vis_frame)
                
                # Display difference view
                if diff_view is not None:
                    cv2.imshow("Hand Detection Test - Slot Differences", diff_view)
                
                # Log significant events
                if results:
                    for result in results:
                        if result['event_type'] == 'CARD_PLAY':
                            print(f"[CARD_PLAY] Slot {result['slot']}: {result['pixels_50']} pixels ({result['percentage']*100:.1f}%)")
                        elif result['event_type'] == 'CARD_SELECT':
                            print(f"[CARD_SELECT] Slot {result['slot']}: {result['pixels_50']} pixels ({result['percentage']*100:.1f}%)")
                
                # Handle keyboard input
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"hand_test_frame_{frame_count}.png"
                    cv2.imwrite(filename, vis_frame)
                    print(f"[TEST] Saved frame to {filename}")
                
                frame_count += 1
                time.sleep(0.1)  # ~10 FPS
                
        except KeyboardInterrupt:
            print("\n[TEST] Stopped by user")
        finally:
            cv2.destroyAllWindows()


def main():
    """Entry point."""
    print("=" * 60)
    print("Hand Detection Test Utility")
    print("=" * 60)
    print()
    
    tester = HandDetectionTester()
    tester.run()


if __name__ == "__main__":
    main()
