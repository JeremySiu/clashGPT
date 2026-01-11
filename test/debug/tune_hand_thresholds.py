"""
Hand Detection Threshold Tuner
Interactive tool to find optimal thresholds for distinguishing
card selection vs card play events.
"""

import cv2
import numpy as np
import mss
import time


class ThresholdTuner:
    """Interactive threshold tuning tool."""
    
    def __init__(self):
        self.sct = mss.mss()
        self.setup_capture_region()
        self.setup_hand_region()
        
        self.previous_slots = []
        
        # Tunable parameters
        self.diff_threshold = 50
        self.pixel_threshold = 8000
        self.percentage_threshold = 40  # Store as 0-100 for easier slider
        
        print("[TUNER] Hand Detection Threshold Tuner")
        print("[TUNER] Use trackbars to adjust thresholds in real-time")
        print("[TUNER] Try clicking and playing cards to see the difference")
        print("[TUNER] Press 'q' to quit\n")
    
    def setup_capture_region(self):
        """Setup screen capture."""
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
        """Define hand slots."""
        w, h = self.screen_width, self.screen_height
        
        hand_height = int(h * 0.12)
        hand_y = int(h * 0.88)
        hand_width = int(w * 0.8)
        hand_x = int(w * 0.1)
        
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
    
    def on_diff_threshold_change(self, value):
        """Callback for diff threshold slider."""
        self.diff_threshold = value
    
    def on_pixel_threshold_change(self, value):
        """Callback for pixel threshold slider."""
        self.pixel_threshold = value * 100  # Scale to 0-15000
    
    def on_percentage_threshold_change(self, value):
        """Callback for percentage threshold slider."""
        self.percentage_threshold = value
    
    def capture_screen(self):
        """Capture screen."""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, gray
    
    def analyze_with_current_thresholds(self, current_gray):
        """Analyze slots with current threshold settings."""
        current_slots = []
        
        for slot in self.slots:
            slot_img = current_gray[slot['y1']:slot['y2'], slot['x1']:slot['x2']]
            slot_img = cv2.GaussianBlur(slot_img, (5, 5), 0)
            current_slots.append(slot_img)
        
        if len(self.previous_slots) == 0:
            self.previous_slots = current_slots
            return None
        
        results = []
        for i, (current, previous) in enumerate(zip(current_slots, self.previous_slots)):
            if current.shape != previous.shape:
                results.append({'slot': i, 'detected': False, 'pixels': 0, 'pct': 0.0})
                continue
            
            total_pixels = current.shape[0] * current.shape[1]
            diff = cv2.absdiff(previous, current)
            
            _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
            pixels = np.count_nonzero(thresh)
            pct = (pixels / total_pixels * 100) if total_pixels > 0 else 0.0
            
            # Check if meets thresholds
            detected = (pixels >= self.pixel_threshold and 
                       pct >= self.percentage_threshold)
            
            results.append({
                'slot': i,
                'detected': detected,
                'pixels': pixels,
                'pct': pct,
                'meets_pixel': pixels >= self.pixel_threshold,
                'meets_pct': pct >= self.percentage_threshold
            })
        
        self.previous_slots = current_slots
        return results
    
    def draw_ui(self, frame, results):
        """Draw UI with current results."""
        if results is None:
            cv2.putText(frame, "INITIALIZING...", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return frame
        
        # Draw threshold info
        y = 30
        cv2.putText(frame, f"Diff Threshold: {self.diff_threshold}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Pixel Threshold: {self.pixel_threshold}", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 30
        cv2.putText(frame, f"Percentage Threshold: {self.percentage_threshold}%", (20, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 40
        
        # Draw slot results
        for result in results:
            slot = result['slot']
            detected = result['detected']
            pixels = result['pixels']
            pct = result['pct']
            meets_pixel = result['meets_pixel']
            meets_pct = result['meets_pct']
            
            if detected:
                color = (0, 255, 0)  # Green - DETECTED
                status = "DETECTED"
            elif meets_pixel or meets_pct:
                color = (0, 165, 255)  # Orange - PARTIAL
                status = f"PARTIAL (pix:{meets_pixel} pct:{meets_pct})"
            else:
                color = (150, 150, 150)  # Gray - idle
                status = "idle"
            
            text = f"Slot{slot}: {status} | {pixels}px ({pct:.1f}%)"
            cv2.putText(frame, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y += 25
        
        # Draw boxes on slots
        for slot_info, result in zip(self.slots, results):
            if result['detected']:
                color = (0, 255, 0)
                thickness = 3
            else:
                color = (100, 100, 100)
                thickness = 1
            
            cv2.rectangle(frame, (slot_info['x1'], slot_info['y1']),
                         (slot_info['x2'], slot_info['y2']), color, thickness)
        
        return frame
    
    def run(self):
        """Run the tuner."""
        # Create window
        window_name = "Threshold Tuner"
        cv2.namedWindow(window_name)
        
        # Create trackbars
        cv2.createTrackbar("Diff Threshold", window_name, self.diff_threshold, 100,
                          self.on_diff_threshold_change)
        cv2.createTrackbar("Pixel Threshold (x100)", window_name, 
                          self.pixel_threshold // 100, 150,
                          self.on_pixel_threshold_change)
        cv2.createTrackbar("Percentage Threshold", window_name, 
                          self.percentage_threshold, 100,
                          self.on_percentage_threshold_change)
        
        try:
            while True:
                frame, gray = self.capture_screen()
                results = self.analyze_with_current_thresholds(gray)
                vis_frame = self.draw_ui(frame, results)
                
                cv2.imshow(window_name, vis_frame)
                
                # Log detections
                if results:
                    for result in results:
                        if result['detected']:
                            print(f"[DETECT] Slot {result['slot']}: {result['pixels']}px ({result['pct']:.1f}%)")
                
                key = cv2.waitKey(100) & 0xFF
                if key == ord('q'):
                    break
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n[TUNER] Stopped")
        finally:
            # Print final values
            print("\n" + "=" * 60)
            print("FINAL THRESHOLD VALUES:")
            print(f"HAND_CHANGE_THRESHOLD = {self.pixel_threshold}")
            print(f"HAND_CHANGE_PERCENTAGE = {self.percentage_threshold / 100.0}")
            print(f"Diff threshold in detection: {self.diff_threshold}")
            print("=" * 60)
            cv2.destroyAllWindows()


def main():
    print("=" * 60)
    print("Hand Detection Threshold Tuner")
    print("=" * 60)
    print()
    
    tuner = ThresholdTuner()
    tuner.run()


if __name__ == "__main__":
    main()
