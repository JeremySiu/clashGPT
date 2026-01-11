"""
Clash Royale OCR Reader
Reads all text from screen capture and identifies timer/HP by position.
"""

import cv2
import numpy as np
import mss
import time
from typing import Optional, Dict, List, Tuple

import easyocr


class ClashOCRReader:
    """Reads all text from Clash Royale screen and identifies game elements by position."""
    
    def __init__(self):
        """Initialize OCR reader."""
        self.screen_width = 1920
        self.screen_height = 1080
        
        # Capture right 30% of screen
        self.capture_width = int(self.screen_width * 0.30)
        self.capture_x = self.screen_width - self.capture_width
        
        # Initialize EasyOCR
        print("[INFO] Initializing EasyOCR reader...")
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        
        # HP bar calibration data (set after initial detection)
        self.calibrated = False
        self.opponent_left_bar = None  # (x, y, w, h, baseline_colored_pixels)
        self.opponent_right_bar = None
        self.player_left_bar = None
        self.player_right_bar = None
        
        # Define expected regions (normalized coordinates relative to capture area)
        # Timer: top right area (includes "TIME LEFT" text above timer)
        self.timer_region = {
            'x_min': 0.85, 'x_max': 0.98,  # Far right side
            'y_min': 0.00, 'y_max': 0.10   # Top area
        }
        
        # Opponent HP bars (RED) - Top of screen, smaller and more centered
        self.opponent_left_hp_region = {
            'x_min': 0.15, 'x_max': 0.40,
            'y_min': 0.05, 'y_max': 0.12
        }
        
        self.opponent_right_hp_region = {
            'x_min': 0.60, 'x_max': 0.85,
            'y_min': 0.05, 'y_max': 0.12
        }
        
        # Player HP bars (BLUE) - Bottom of screen, smaller and more centered
        self.player_left_hp_region = {
            'x_min': 0.15, 'x_max': 0.40,
            'y_min': 0.88, 'y_max': 0.95
        }
        
        self.player_right_hp_region = {
            'x_min': 0.60, 'x_max': 0.85,
            'y_min': 0.88, 'y_max': 0.95
        }
        
        print(f"[INFO] Capture region: {self.capture_width}x{self.screen_height} "
              f"at ({self.capture_x}, 0)")
    
    def capture_screen(self) -> np.ndarray:
        """Capture the game screen."""
        with mss.mss() as sct:
            monitor = {
                "top": 0,
                "left": self.capture_x,
                "width": self.capture_width,
                "height": self.screen_height
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def capture_timer_region(self) -> np.ndarray:
        """Capture only the timer region for faster OCR processing."""
        # Calculate pixel coordinates for timer region
        timer_x = int(self.capture_width * self.timer_region['x_min'])
        timer_y = int(self.screen_height * self.timer_region['y_min'])
        timer_w = int(self.capture_width * (self.timer_region['x_max'] - self.timer_region['x_min']))
        timer_h = int(self.screen_height * (self.timer_region['y_max'] - self.timer_region['y_min']))
        
        with mss.mss() as sct:
            monitor = {
                "top": timer_y,
                "left": self.capture_x + timer_x,
                "width": timer_w,
                "height": timer_h
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def capture_hp_bar_region(self, region: Dict[str, float]) -> np.ndarray:
        """Capture a specific HP bar region."""
        # Calculate pixel coordinates for HP region
        hp_x = int(self.capture_width * region['x_min'])
        hp_y = int(self.screen_height * region['y_min'])
        hp_w = int(self.capture_width * (region['x_max'] - region['x_min']))
        hp_h = int(self.screen_height * (region['y_max'] - region['y_min']))
        
        with mss.mss() as sct:
            monitor = {
                "top": hp_y,
                "left": self.capture_x + hp_x,
                "width": hp_w,
                "height": hp_h
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def capture_hp_bar_region(self, region: Dict[str, float]) -> np.ndarray:
        """Capture a specific HP bar region."""
        # Calculate pixel coordinates for HP region
        hp_x = int(self.capture_width * region['x_min'])
        hp_y = int(self.screen_height * region['y_min'])
        hp_w = int(self.capture_width * (region['x_max'] - region['x_min']))
        hp_h = int(self.screen_height * (region['y_max'] - region['y_min']))
        
        with mss.mss() as sct:
            monitor = {
                "top": hp_y,
                "left": self.capture_x + hp_x,
                "width": hp_w,
                "height": hp_h
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def capture_hp_bar_region(self, region: Dict[str, float]) -> np.ndarray:
        """Capture a specific HP bar region."""
        # Calculate pixel coordinates for HP region
        hp_x = int(self.capture_width * region['x_min'])
        hp_y = int(self.screen_height * region['y_min'])
        hp_w = int(self.capture_width * (region['x_max'] - region['x_min']))
        hp_h = int(self.screen_height * (region['y_max'] - region['y_min']))
        
        with mss.mss() as sct:
            monitor = {
                "top": hp_y,
                "left": self.capture_x + hp_x,
                "width": hp_w,
                "height": hp_h
            }
            screenshot = sct.grab(monitor)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        return frame
    
    def preprocess_image(self, image: np.ndarray, scale: int = 3) -> np.ndarray:
        """Preprocess image for better OCR results."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Upscale
        height, width = gray.shape
        scaled = cv2.resize(gray, (width * scale, height * scale), 
                           interpolation=cv2.INTER_CUBIC)
        
        # Apply threshold
        _, binary = cv2.threshold(scaled, 0, 255, 
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def is_in_region(self, x: float, y: float, region: Dict[str, float]) -> bool:
        """Check if normalized coordinates are within a region."""
        return (region['x_min'] <= x <= region['x_max'] and 
                region['y_min'] <= y <= region['y_max'])
    
    def parse_timer(self, text: str) -> Optional[int]:
        """Parse timer text to seconds. Expects M:SS, MM:SS or M.SS format."""
        text = text.strip()
        
        # Remove any non-digit/colon/period characters
        cleaned = ''.join(c for c in text if c.isdigit() or c in ':.').replace('.', ':')
        
        if ':' not in cleaned:
            return None
        
        parts = cleaned.split(':')
        if len(parts) != 2:
            return None
        
        try:
            minutes = int(parts[0])
            seconds = int(parts[1])
            
            # Validate ranges (Clash Royale timer is 0:00 to 6:00)
            if 0 <= minutes <= 6 and 0 <= seconds <= 59:
                return minutes * 60 + seconds
        except ValueError:
            pass
        
        return None
    
    def find_hp_bars_by_color(self, frame: np.ndarray, color_type: str = 'red', search_top: bool = True) -> List[Tuple[int, int, int, int, float]]:
        """Automatically find HP bars by color in the frame.
        
        Args:
            frame: The captured frame
            color_type: 'red' or 'blue'
            search_top: True to search top half, False for bottom half
        
        Returns:
            List of (x, y, width, height, fill_percentage) for each detected bar
        """
        height, width = frame.shape[:2]
        
        # Search only top or bottom half
        if search_top:
            search_frame = frame[0:height//2, :]
        else:
            search_frame = frame[height//2:, :]
        
        hsv = cv2.cvtColor(search_frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for the target color
        if color_type == 'red':
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # blue
            lower_blue = np.array([100, 120, 70])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find contours of colored regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bars = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter for bar-like shapes (reasonable size)
            if area < 200:  # Too small
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by minimum width
            if w < 50:  # Too narrow to be an HP bar
                continue
            
            # Adjust y if searching bottom half
            if not search_top:
                y += height // 2
            
            # Filter for horizontal bar shape (width > height)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 1.5:  # Not bar-shaped enough
                continue
            
            # Calculate fill percentage for this bar region
            bar_region = frame[y:y+h, x:x+w]
            fill_percent = self.calculate_hp_bar_percentage(bar_region, color_type)
            
            if fill_percent is not None:
                bars.append((x, y, w, h, fill_percent))
        
        # Sort bars by x position (left to right)
        bars.sort(key=lambda b: b[0])
        
        # Filter to keep only bars that are similar size (the two tower HP bars)
        if len(bars) > 2:
            # Group bars by similar width (within 20% tolerance)
            size_groups = []
            for bar in bars:
                x, y, w, h, fill = bar
                # Find a group with similar width
                found_group = False
                for group in size_groups:
                    avg_width = sum(b[2] for b in group) / len(group)
                    if abs(w - avg_width) / avg_width < 0.3:  # Within 30% of average
                        group.append(bar)
                        found_group = True
                        break
                if not found_group:
                    size_groups.append([bar])
            
            # Take the group with exactly 2 bars, or the largest group, or the 2 largest bars
            if any(len(g) == 2 for g in size_groups):
                bars = [g for g in size_groups if len(g) == 2][0]
            elif size_groups:
                # Take the largest group and limit to 2 bars
                largest_group = max(size_groups, key=len)
                # Sort by area and take 2 largest
                largest_group.sort(key=lambda b: b[2] * b[3], reverse=True)
                bars = largest_group[:2]
                # Re-sort by x position
                bars.sort(key=lambda b: b[0])
        
        # Return only first 2 bars (left and right towers)
        return bars[:2]
    
    def calculate_hp_bar_percentage(self, hp_bar_image: np.ndarray, color_type: str = 'red') -> Optional[float]:
        """Calculate the fill percentage of an HP bar by detecting red or blue color.
        
        Args:
            hp_bar_image: The image of the HP bar region
            color_type: 'red' for opponent bars (top), 'blue' for player bars (bottom)
        """
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(hp_bar_image, cv2.COLOR_BGR2HSV)
        
        if color_type == 'red':
            # Red color range in HSV (red wraps around at 0/180)
            # Lower red range (0-10)
            lower_red1 = np.array([0, 100, 100])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            
            # Upper red range (170-180)
            lower_red2 = np.array([170, 100, 100])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            
            # Combine both red ranges
            mask = cv2.bitwise_or(mask1, mask2)
        
        elif color_type == 'blue':
            # Blue color range in HSV
            lower_blue = np.array([100, 100, 100])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        else:
            return None
        
        # Calculate the percentage of colored pixels (HP remaining)
        total_pixels = mask.size
        filled_pixels = np.count_nonzero(mask)
        
        if total_pixels == 0:
            return None
        
        percentage = (filled_pixels / total_pixels) * 100.0
        return percentage
    
    def calibrate_hp_bars(self) -> bool:
        """Detect and calibrate HP bar positions at 100% HP."""
        print("[INFO] Calibrating HP bar positions...")
        frame = self.capture_screen()
        
        # Find bars by color
        opponent_bars = self.find_hp_bars_by_color(frame, 'red', search_top=True)
        player_bars = self.find_hp_bars_by_color(frame, 'blue', search_top=False)
        
        if len(opponent_bars) < 2:
            print(f"[WARNING] Only found {len(opponent_bars)} opponent bars, need 2")
            return False
        
        if len(player_bars) < 2:
            print(f"[WARNING] Only found {len(player_bars)} player bars, need 2")
            return False
        
        # Store bar positions, sizes, and baseline colored pixel counts
        self.opponent_left_bar = self.create_bar_baseline(frame, opponent_bars[0], 'red')
        self.opponent_right_bar = self.create_bar_baseline(frame, opponent_bars[1], 'red')
        self.player_left_bar = self.create_bar_baseline(frame, player_bars[0], 'blue')
        self.player_right_bar = self.create_bar_baseline(frame, player_bars[1], 'blue')
        
        self.calibrated = True
        
        print("[INFO] HP bars calibrated successfully:")
        print(f"  Opponent Left:  pos=({self.opponent_left_bar[0]}, {self.opponent_left_bar[1]}) size=({self.opponent_left_bar[2]}x{self.opponent_left_bar[3]}) baseline={self.opponent_left_bar[4]}px")
        print(f"  Opponent Right: pos=({self.opponent_right_bar[0]}, {self.opponent_right_bar[1]}) size=({self.opponent_right_bar[2]}x{self.opponent_right_bar[3]}) baseline={self.opponent_right_bar[4]}px")
        print(f"  Player Left:    pos=({self.player_left_bar[0]}, {self.player_left_bar[1]}) size=({self.player_left_bar[2]}x{self.player_left_bar[3]}) baseline={self.player_left_bar[4]}px")
        print(f"  Player Right:   pos=({self.player_right_bar[0]}, {self.player_right_bar[1]}) size=({self.player_right_bar[2]}x{self.player_right_bar[3]}) baseline={self.player_right_bar[4]}px")
        
        return True
    
    def create_bar_baseline(self, frame: np.ndarray, bar_info: Tuple[int, int, int, int, float], color_type: str) -> Tuple[int, int, int, int, int]:
        """Create baseline data for a bar by counting colored pixels at calibration.
        
        Returns:
            (x, y, w, h, baseline_colored_pixels)
        """
        x, y, w, h, _ = bar_info
        
        # Extract the bar region
        bar_region = frame[y:y+h, x:x+w]
        
        # Convert to HSV and create mask
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        
        if color_type == 'red':
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # blue
            lower_blue = np.array([100, 120, 70])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Count total colored pixels as baseline (100% HP)
        baseline_colored_pixels = np.count_nonzero(mask)
        
        return (x, y, w, h, baseline_colored_pixels)
    
    def _print_pixel_debug(self, frame: np.ndarray, label: str, bar_info: Tuple[int, int, int, int, int], color_type: str):
        """Print debug info comparing baseline to current pixels."""
        x, y, w, h, baseline_pixels = bar_info
        
        # Extract the bar region
        bar_region = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for the target color
        if color_type == 'red':
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # blue
            lower_blue = np.array([100, 120, 70])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        current_pixels = np.count_nonzero(mask)
        percentage = (current_pixels / baseline_pixels * 100.0) if baseline_pixels > 0 else 0
        
        print(f"    [{label}] baseline={baseline_pixels}px | current={current_pixels}px | {percentage:.1f}%")
    
    def read_tower_hp_bars(self) -> Dict[str, Optional[float]]:
        """Read all four tower HP bars using calibrated positions."""
        if not self.calibrated:
            return {
                'opponent_left_hp': None,
                'opponent_right_hp': None,
                'player_left_hp': None,
                'player_right_hp': None,
                'calibrated': False
            }
        
        frame = self.capture_screen()
        
        # Read each bar region and calculate HP percentage
        opp_left_hp = self.read_calibrated_bar(frame, self.opponent_left_bar, 'red')
        opp_right_hp = self.read_calibrated_bar(frame, self.opponent_right_bar, 'red')
        player_left_hp = self.read_calibrated_bar(frame, self.player_left_bar, 'blue')
        player_right_hp = self.read_calibrated_bar(frame, self.player_right_bar, 'blue')
        
        return {
            'opponent_left_hp': opp_left_hp,
            'opponent_right_hp': opp_right_hp,
            'player_left_hp': player_left_hp,
            'player_right_hp': player_right_hp,
            'calibrated': True
        }
    
    def read_calibrated_bar(self, frame: np.ndarray, bar_info: Tuple[int, int, int, int, int], color_type: str) -> Optional[float]:
        """Read HP from a calibrated bar region by counting colored pixels.
        
        Args:
            frame: The current frame
            bar_info: (x, y, w, h, baseline_colored_pixels) from calibration
            color_type: 'red' or 'blue'
        
        Returns:
            HP percentage (0-100) relative to the calibrated baseline
        """
        x, y, w, h, baseline_pixels = bar_info
        
        # Extract the SAME bar region using calibrated position
        bar_region = frame[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv = cv2.cvtColor(bar_region, cv2.COLOR_BGR2HSV)
        
        # Create mask for the target color
        if color_type == 'red':
            lower_red1 = np.array([0, 120, 70])
            upper_red1 = np.array([10, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            lower_red2 = np.array([170, 120, 70])
            upper_red2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # blue
            lower_blue = np.array([100, 120, 70])
            upper_blue = np.array([130, 255, 255])
            mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Count current colored pixels in this SAME region
        current_pixels = np.count_nonzero(mask)
        
        if baseline_pixels == 0:
            return None
        
        # Calculate percentage relative to baseline
        percentage = (current_pixels / baseline_pixels) * 100.0
        
        # Cap at 100%
        percentage = min(percentage, 100.0)
        percentage = max(percentage, 0.0)
        
        return percentage
    
    def parse_hp(self, text: str) -> Optional[int]:
        """Parse HP text to integer."""
        # Remove any non-digit characters
        cleaned = ''.join(c for c in text if c.isdigit())
        
        if not cleaned:
            return None
        
        try:
            hp = int(cleaned)
            # Validate range (typical tower HP is 1000-5000)
            if 500 <= hp <= 10000:
                return hp
        except ValueError:
            pass
        
        return None
    
    def read_text_easyocr(self, image: np.ndarray, timer_only: bool = False) -> List[Tuple[str, float, float, float, float]]:
        """
        Read all text using EasyOCR.
        Returns list of (text, x_center_norm, y_center_norm, confidence, width_norm, height_norm).
        If timer_only=True, assumes image is already cropped to timer region.
        """
        results = []
        
        # Preprocess
        processed = self.preprocess_image(image, scale=2)  # Reduced from 3x to 2x for speed
        
        # Run EasyOCR (detail=1 gives bounding boxes)
        # allowlist for faster processing - only look for numbers, colon, and common letters
        detections = self.reader.readtext(processed, detail=1, paragraph=False, allowlist='0123456789:. ')
        
        img_height, img_width = processed.shape[:2]
        
        for detection in detections:
            bbox, text, confidence = detection
            
            # Calculate center point and dimensions (normalized)
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            x_center = sum(x_coords) / 4 / img_width
            y_center = sum(y_coords) / 4 / img_height
            width = (max(x_coords) - min(x_coords)) / img_width
            height = (max(y_coords) - min(y_coords)) / img_height
            
            results.append((text, x_center, y_center, confidence, width, height))
        
        return results
    
    def read_timer_only(self) -> Optional[int]:
        """Fast timer-only reading by capturing just the timer region."""
        # Capture only timer region
        timer_frame = self.capture_timer_region()
        
        # Read text from timer region
        detections = self.read_text_easyocr(timer_frame, timer_only=True)
        
        # Try to parse any detected text as timer
        for text, x, y, conf, w, h in detections:
            parsed_time = self.parse_timer(text)
            if parsed_time is not None:
                return parsed_time
        
        return None
    
    def classify_text(self, detections: List[Tuple[str, float, float, float, float]]) -> Dict[str, any]:
        """
        Classify detected text into timer, left HP, right HP based on position.
        """
        game_state = {
            'time': None,
            'opponent_left_hp': None,
            'opponent_right_hp': None,
            'player_left_hp': None,
            'player_right_hp': None,
            'raw_detections': []
        }
        
        for text, x, y, conf, w, h in detections:
            detection_info = {
                'text': text,
                'x': x,
                'y': y,
                'confidence': conf,
                'width': w,
                'height': h
            }
            game_state['raw_detections'].append(detection_info)
            
            # Check if in timer region
            if self.is_in_region(x, y, self.timer_region):
                parsed_time = self.parse_timer(text)
                if parsed_time is not None:
                    game_state['time'] = parsed_time
                    detection_info['classified_as'] = 'timer'
                    continue
            
            # Check opponent HP regions (top)
            if self.is_in_region(x, y, self.opponent_left_hp_region):
                parsed_hp = self.parse_hp(text)
                if parsed_hp is not None:
                    game_state['opponent_left_hp'] = parsed_hp
                    detection_info['classified_as'] = 'opp_left_hp'
                    continue
            
            if self.is_in_region(x, y, self.opponent_right_hp_region):
                parsed_hp = self.parse_hp(text)
                if parsed_hp is not None:
                    game_state['opponent_right_hp'] = parsed_hp
                    detection_info['classified_as'] = 'opp_right_hp'
                    continue
            
            # Check player HP regions (bottom)
            if self.is_in_region(x, y, self.player_left_hp_region):
                parsed_hp = self.parse_hp(text)
                if parsed_hp is not None:
                    game_state['player_left_hp'] = parsed_hp
                    detection_info['classified_as'] = 'player_left_hp'
                    continue
            
            if self.is_in_region(x, y, self.player_right_hp_region):
                parsed_hp = self.parse_hp(text)
                if parsed_hp is not None:
                    game_state['player_right_hp'] = parsed_hp
                    detection_info['classified_as'] = 'player_right_hp'
                    continue
            
            detection_info['classified_as'] = 'unknown'
        
        return game_state
    
    def read_game_state(self) -> Dict[str, any]:
        """Capture screen and extract game state."""
        frame = self.capture_screen()
        
        # Read all text using EasyOCR
        detections = self.read_text_easyocr(frame)
        
        # Classify detections
        game_state = self.classify_text(detections)
        
        return game_state
    
    def visualize_detections(self, frame: np.ndarray, game_state: Dict[str, any]) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        vis = frame.copy()
        height, width = vis.shape[:2]
        
        # Draw region boundaries
        regions = [
            (self.timer_region, (255, 255, 0), "TIMER"),
            (self.opponent_left_hp_region, (0, 0, 255), "OPP_LEFT"),
            (self.opponent_right_hp_region, (0, 0, 255), "OPP_RIGHT"),
            (self.player_left_hp_region, (255, 0, 0), "PLAYER_LEFT"),
            (self.player_right_hp_region, (255, 0, 0), "PLAYER_RIGHT")
        ]
        
        for region, color, label in regions:
            x1 = int(region['x_min'] * width)
            y1 = int(region['y_min'] * height)
            x2 = int(region['x_max'] * width)
            y2 = int(region['y_max'] * height)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, color, 1)
        
        # Draw detections
        for det in game_state['raw_detections']:
            x = int(det['x'] * width)
            y = int(det['y'] * height)
            w = int(det['width'] * width)
            h = int(det['height'] * height)
            
            x1 = x - w // 2
            y1 = y - h // 2
            x2 = x + w // 2
            y2 = y + h // 2
            
            color = (0, 255, 0) if det.get('classified_as') != 'unknown' else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            label = f"{det['text']} ({det.get('classified_as', 'unk')})"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1)
        
        return vis
    
    def run(self, show_visualization: bool = False):
        """Main loop."""
        print("[INFO] Starting OCR reader. Press Ctrl+C to quit.")
        print("[INFO] Waiting 5 seconds before calibration...")
        print("[INFO] Make sure the battle starts and towers are at 100% HP!")
        
        try:
            # Wait 5 seconds before starting calibration
            for i in range(5, 0, -1):
                print(f"  Starting calibration in {i}...")
                time.sleep(1)
            
            print("\n[INFO] Beginning HP bar detection...")
            
            # Keep calibrating until we get bars
            calibration_attempts = 0
            while not self.calibrated:
                calibration_attempts += 1
                print(f"\n[ATTEMPT {calibration_attempts}] Searching for HP bars...")
                
                if self.calibrate_hp_bars():
                    print("[SUCCESS] HP bars detected and calibrated at 100%!")
                    print("[INFO] Starting continuous monitoring...")
                    break
                else:
                    print("[WAITING] Bars not detected. Retrying in 1 second...")
                    time.sleep(1)
            
            print("\n[INFO] Reading game state every 10 seconds")
            print("=" * 60)
            
            while True:
                loop_start = time.time()
                
                # Read timer region
                timer_seconds = self.read_timer_only()
                
                # Read HP bars
                hp_data = self.read_tower_hp_bars()
                
                # Format output
                if timer_seconds is not None:
                    time_str = f"{timer_seconds//60:.0f}:{timer_seconds%60:02.0f}"
                else:
                    time_str = "--:--"
                
                # Format HP percentages
                opp_left = f"{hp_data['opponent_left_hp']:.1f}%" if hp_data['opponent_left_hp'] is not None else "--.--%"
                opp_right = f"{hp_data['opponent_right_hp']:.1f}%" if hp_data['opponent_right_hp'] is not None else "--.--%"
                player_left = f"{hp_data['player_left_hp']:.1f}%" if hp_data['player_left_hp'] is not None else "--.--%"
                player_right = f"{hp_data['player_right_hp']:.1f}%" if hp_data['player_right_hp'] is not None else "--.--%"
                
                # Print output
                print(f"\n[TIME] {time_str}")
                print(f"  [OPPONENT] Left: {opp_left} | Right: {opp_right}")
                print(f"  [PLAYER]   Left: {player_left} | Right: {player_right}")
                
                # Debug: show baseline vs current pixel counts
                if hp_data['calibrated']:
                    frame = self.capture_screen()
                    self._print_pixel_debug(frame, "OPP_L", self.opponent_left_bar, 'red')
                    self._print_pixel_debug(frame, "OPP_R", self.opponent_right_bar, 'red')
                    self._print_pixel_debug(frame, "PLY_L", self.player_left_bar, 'blue')
                    self._print_pixel_debug(frame, "PLY_R", self.player_right_bar, 'blue')
                
                # Print processing time for debugging
                elapsed = time.time() - loop_start
                print(f"  [DEBUG] Processing took {elapsed:.2f}s")
                
                # Wait 10 seconds between reads (accounting for processing time)
                sleep_time = max(0, 10.0 - elapsed)
                time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping OCR reader.")
        finally:
            if show_visualization:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    reader = ClashOCRReader()
    reader.run(show_visualization=False)
