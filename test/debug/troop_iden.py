"""
Clash Royale Card Detection System
Detects card plays in a Hog 2.6 mirror match using screen capture and computer vision.
Focuses on the far-right region where the emulator is running.
"""

import cv2
import numpy as np
import mss
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import sys
import re

# Try EasyOCR (better for stylized fonts)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("[INFO] EasyOCR available - using for timer/HP")
except ImportError:
    EASYOCR_AVAILABLE = False
    print("[WARNING] EasyOCR not available. Install with: pip install easyocr")

# Fallback to Tesseract
try:
    import pytesseract
    import os
    PYTESSERACT_AVAILABLE = True
    
    # Try common Windows installation paths
    common_paths = [
        r"C:\Users\jerem\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    
    tesseract_found = False
    for path in common_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            tesseract_found = True
            print(f"[INFO] Found Tesseract at: {path}")
            break
    
    # Test if tesseract executable is available
    try:
        pytesseract.get_tesseract_version()
        tesseract_found = True
    except Exception as e:
        if not tesseract_found:
            PYTESSERACT_AVAILABLE = False
            print("[WARNING] Tesseract OCR engine not found. OCR features disabled.")
            print("[WARNING] Download from: https://github.com/UB-Mannheim/tesseract/wiki")
            print(f"[WARNING] Error: {e}")
except ImportError:
    PYTESSERACT_AVAILABLE = False
    print("[WARNING] pytesseract not installed. OCR features disabled.")
    print("[WARNING] Install with: pip install pytesseract")


@dataclass
class TroopFeatures:
    """Explicit OpenCV-extracted features for a detected object."""
    # Geometric features
    area: float
    width: float
    height: float
    aspect_ratio: float  # height / width
    solidity: float  # area / bounding_box_area
    compactness: float  # area / perimeter^2
    
    # Motion features (temporal)
    velocity: float  # pixels per frame
    velocity_x: float
    velocity_y: float
    direction: str  # "vertical", "horizontal", "stationary"
    lifetime: int  # frames visible
    
    # Position
    centroid: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    
    # Color features (secondary)
    mean_rgb: Tuple[float, float, float]
    dominant_color: str  # "red", "blue", "neutral"
    
    # Context
    spawn_roi: str
    object_id: Optional[int] = None


@dataclass
class Detection:
    """Represents a detected card play event."""
    timestamp: float
    player: str  # "OPPONENT" or "PLAYER"
    card: str
    lane: str  # "LEFT", "RIGHT", or "CENTER"
    confidence: float
    position: Tuple[int, int]  # centroid
    features: Optional[TroopFeatures] = None  # Debug/validation


@dataclass
class ROI:
    """Region of Interest with coordinates."""
    name: str
    x: int
    y: int
    width: int
    height: int
    
    def get_slice(self):
        """Return slice coordinates for cropping."""
        return (self.y, self.y + self.height, self.x, self.x + self.width)


class CardDetector:
    """Main card detection system."""
    
    # Card names
    CARDS = ["Hog Rider", "Ice Golem", "Musketeer", "Cannon", 
             "Ice Spirit", "Skeletons", "Fireball", "The Log"]
    
    # Detection thresholds
    MOTION_THRESHOLD = 25  # Pixel intensity difference to detect motion
    MIN_CONTOUR_AREA = 100  # Minimum pixels for a valid detection
    CONFIDENCE_THRESHOLD = 0.6  # Minimum confidence to report
    DUPLICATE_SUPPRESS_TIME = 2.0  # Seconds to suppress duplicate detections
    TEMPORAL_FRAMES = 3  # Number of frames to confirm detection
    
    # Spawn event thresholds
    SPAWN_PIXEL_THRESHOLD = 500  # Minimum changed pixels for spawn event
    SPAWN_FRAMES_REQUIRED = 2  # Frames required to confirm spawn
    ROI_COOLDOWN = 1.2  # Seconds cooldown per ROI after detection
    
    # Troop-specific feature thresholds (OpenCV-based)
    TROOP_SPECS = {
        "Hog Rider": {
            "area_range": (2500, 10000),  # Relaxed for varied captures
            "aspect_ratio_range": (0.8, 3.0),  # Very relaxed - use size + location instead
            "min_velocity": 10,  # Reduced for 5 FPS large displacement
            "spawn_rois": ["BRIDGE"],
            "lifetime_range": (2, 30),  # Quick confirmation
            "min_vy": 5,  # Must move vertically toward opponent
            "quick_confirm": True  # Can confirm in 2 frames
        },
        "Cannon": {
            "area_range": (800, 6000),  # Wider for fragmented contours
            "aspect_ratio_range": (0.6, 2.0),  # Very relaxed
            "max_velocity": 8,  # Allow initial placement motion
            "solidity_min": 0.4,  # Lowered for fragmented contours
            "spawn_rois": ["CENTER"],
            "lifetime_range": (2, 100),  # Quick confirmation for static object
            "require_static": True,  # Must become stationary
            "quick_confirm": True  # Can confirm in 2 frames
        },
        "Musketeer": {
            "area_range": (1500, 4000),
            "aspect_ratio_range": (2.0, 4.0),
            "max_velocity": 10,
            "spawn_rois": ["BACK", "BRIDGE"],
            "lifetime_range": (5, 50)
        },
        "Ice Golem": {
            "area_range": (1000, 2500),
            "aspect_ratio_range": (0.8, 1.4),
            "velocity_range": (3, 15),
            "spawn_rois": ["BACK", "BRIDGE"],
            "lifetime_range": (4, 40)
        },
        "Ice Spirit": {
            "area_range": (300, 800),
            "aspect_ratio_range": (0.5, 1.5),
            "min_velocity": 15,
            "spawn_rois": ["BRIDGE", "BACK"],
            "lifetime_range": (1, 5)  # Very short
        },
        "Skeletons": {
            "area_range": (150, 600),
            "aspect_ratio_range": (0.5, 2.0),
            "velocity_range": (2, 10),
            "spawn_rois": ["BRIDGE", "BACK"],
            "lifetime_range": (2, 15),
            "requires_group": True  # Multiple small contours
        },
        "Fireball": {
            "area_range": (800, 3000),
            "compactness_min": 0.1,  # Circular
            "lifetime_range": (1, 4),  # Very short
            "color_required": "red",
            "spawn_rois": ["BRIDGE", "BACK", "CENTER"]
        },
        "The Log": {
            "area_range": (1000, 5000),
            "aspect_ratio_max": 0.8,  # Relaxed for axis-aligned bbox
            "min_velocity_x": 10,  # Reduced for 5 FPS
            "max_velocity_y": 8,  # Max vertical drift
            "spawn_rois": ["BRIDGE", "BACK"],
            "lifetime_range": (2, 8),
            "use_oriented_bbox": True,  # Use minAreaRect
            "quick_confirm": True  # Can confirm in 2 frames
        }
    }
    
    # Hand detection thresholds
    HAND_CHANGE_THRESHOLD = 8000  # Minimum changed pixels in a slot (full card replacement)
    HAND_CHANGE_PERCENTAGE = 0.3  # Minimum 40% of slot must change
    HAND_DEBOUNCE_TIME = 0.5  # Seconds to suppress duplicate hand changes
    HAND_VALID_WINDOW = 0.6  # Seconds a hand change is considered valid
    
    # Debug mode
    DEBUG = False  # Set to True for verbose rejection logging
    DEBUG_HAND = False  # Set to True for hand detection debug output
    DEBUG_FEATURES = False  # Set to True to log extracted features
    DEBUG_OCR = False  # Set to True for OCR debug output
    
    # OCR disabled - Clash Royale fonts are too stylized for OCR engines
    ENABLE_OCR = False  # Set to True to attempt OCR (requires EasyOCR or Tesseract)
    
    # OCR timing (seconds between reads)
    OCR_TIMER_INTERVAL = 1.0
    OCR_HP_INTERVAL = 1.5
    OCR_ELIXIR_INTERVAL = 1.0
    
    def __init__(self):
        """Initialize the card detector."""
        self.sct = mss.mss()
        self.start_time = time.time()
        self.previous_frame = None
        self.frame_buffer = deque(maxlen=self.TEMPORAL_FRAMES)
        self.recent_detections = deque(maxlen=20)  # Track recent detections
        
        # Initialize EasyOCR reader if available
        if EASYOCR_AVAILABLE:
            print("[INFO] Initializing EasyOCR reader...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("[INFO] EasyOCR reader ready")
        else:
            self.easyocr_reader = None
        
        # Spawn event tracking
        self.roi_backgrounds = {}  # Background reference per ROI
        self.roi_spawn_buffer = {}  # Spawn events per ROI over frames
        self.roi_last_detection = {}  # Last detection time per ROI for cooldown
        
        # Temporal object tracking for feature extraction
        self.tracked_objects = {}  # {object_id: {history, features}}
        self.next_object_id = 0
        self.skeleton_groups = {}  # Track skeleton group candidates
        
        # Hand detection tracking
        self.previous_hand_slots = []  # Previous frame's slot images
        self.last_hand_change_time = 0.0  # Last detected hand change
        self.hand_roi = None  # Hand region ROI
        self.hand_slots = []  # Individual card slot ROIs
        
        # Game state tracking (OCR)
        self.match_time_seconds = None  # Remaining time in seconds
        self.left_tower_hp = None
        self.right_tower_hp = None
        self.elixir_phase = "1x"  # "1x", "2x", "3x"
        
        # OCR temporal smoothing
        self.timer_history = deque(maxlen=3)
        self.left_hp_history = deque(maxlen=3)
        self.right_hp_history = deque(maxlen=3)
        self.elixir_history = deque(maxlen=3)
        
        # OCR timing control
        self.last_ocr_timer = 0.0
        self.last_ocr_hp = 0.0
        self.last_ocr_elixir = 0.0
        
        # OCR ROIs
        self.timer_roi = None
        self.left_tower_hp_roi = None
        self.right_tower_hp_roi = None
        self.elixir_indicator_roi = None
        
        # Mode configuration
        self.mode_profile = "ladder"  # "ladder", "triple_elixir", etc.
        self.match_duration = 180  # Standard 3 minutes
        self.double_elixir_start = 120  # 2x starts at 1:00 remaining
        
        # Get screen dimensions and setup capture area
        self.setup_capture_region()
        self.setup_rois()
        self.setup_hand_region()
        self.setup_ocr_rois()
        
        # Initialize tracking structures
        for roi_name in self.rois.keys():
            self.roi_spawn_buffer[roi_name] = deque(maxlen=self.SPAWN_FRAMES_REQUIRED)
            self.roi_last_detection[roi_name] = 0.0
        
        print("[INFO] Card Detector initialized")
        print(f"[INFO] Monitoring region: {self.monitor}")
        print(f"[INFO] Debug mode: {self.DEBUG}")
        print(f"[INFO] Hand debug mode: {self.DEBUG_HAND}")
        print(f"[INFO] Press Ctrl+C to stop\n")
    
    def setup_capture_region(self):
        """Setup the screen capture region (right 35% of screen)."""
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Primary monitor
            screen_width = monitor["width"]
            screen_height = monitor["height"]
            
            # Capture right 30% of screen
            capture_width = int(screen_width * 0.3)
            capture_x = screen_width - capture_width
            
            self.monitor = {
                "top": 0,
                "left": capture_x,
                "width": capture_width,
                "height": screen_height
            }
            
            self.screen_width = capture_width
            self.screen_height = screen_height
    
    def setup_rois(self):
        """Define regions of interest for card detection."""
        w, h = self.screen_width, self.screen_height
        
        # Adjust these based on emulator layout
        # Assuming standard Clash Royale layout
        
        self.rois = {
            # Bridge regions (where troops spawn at bridge)
            "BRIDGE_LEFT": ROI("BRIDGE_LEFT", int(w*0.2), int(h*0.45), int(w*0.25), int(h*0.15)),
            "BRIDGE_RIGHT": ROI("BRIDGE_RIGHT", int(w*0.55), int(h*0.45), int(w*0.25), int(h*0.15)),
            
            # Back regions (defensive spawns)
            "BACK_LEFT": ROI("BACK_LEFT", int(w*0.15), int(h*0.65), int(w*0.25), int(h*0.2)),
            "BACK_RIGHT": ROI("BACK_RIGHT", int(w*0.6), int(h*0.65), int(w*0.25), int(h*0.2)),
            
            # Center defensive building zone
            "CENTER_DEFENSE": ROI("CENTER_DEFENSE", int(w*0.35), int(h*0.55), int(w*0.3), int(h*0.15)),
            
            # Opponent bridge (top half for opponent plays)
            "OPP_BRIDGE_LEFT": ROI("OPP_BRIDGE_LEFT", int(w*0.2), int(h*0.25), int(w*0.25), int(h*0.15)),
            "OPP_BRIDGE_RIGHT": ROI("OPP_BRIDGE_RIGHT", int(w*0.55), int(h*0.25), int(w*0.25), int(h*0.15)),
            
            # Opponent back
            "OPP_BACK_LEFT": ROI("OPP_BACK_LEFT", int(w*0.15), int(h*0.1), int(w*0.25), int(h*0.15)),
            "OPP_BACK_RIGHT": ROI("OPP_BACK_RIGHT", int(w*0.6), int(h*0.1), int(w*0.25), int(h*0.15)),
        }
    
    def setup_hand_region(self):
        """Define hand region and individual card slots."""
        w, h = self.screen_width, self.screen_height
        
        # Hand region at bottom of screen
        # Typical Clash Royale hand is bottom 12-15% of screen, centered
        hand_height = int(h * 0.12)
        hand_y = int(h * 0.88)
        hand_width = int(w * 0.8)  # Hand doesn't span full width
        hand_x = int(w * 0.1)  # Centered with 10% margin each side
        
        self.hand_roi = ROI("HAND", hand_x, hand_y, hand_width, hand_height)
        
        # Divide into 4 equal card slots
        slot_width = hand_width // 4
        self.hand_slots = []
        
        for i in range(4):
            slot_x = hand_x + (i * slot_width)
            slot_roi = ROI(f"SLOT_{i}", slot_x, hand_y, slot_width, hand_height)
            self.hand_slots.append(slot_roi)
        
        if self.DEBUG_HAND:
            print(f"[HAND DEBUG] Hand region: x={hand_x}, y={hand_y}, w={hand_width}, h={hand_height}")
            for i, slot in enumerate(self.hand_slots):
                print(f"[HAND DEBUG] Slot {i}: x={slot.x}, y={slot.y}, w={slot.width}, h={slot.height}")
    
    def setup_ocr_rois(self):
        """Define ROIs for OCR extraction (timer, HP, elixir)."""
        w, h = self.screen_width, self.screen_height
        
        # Timer at top center (tuned position)
        timer_width = int(w * 0.190)
        timer_height = int(h * 0.070)
        timer_x = int(w * 0.815)
        timer_y = int(h * 0.030)
        self.timer_roi = ROI("TIMER", timer_x, timer_y, timer_width, timer_height)
        
        # Left tower HP (tuned position - adjusted for full digit capture)
        hp_width = int(w * 0.120)
        hp_height = int(h * 0.055)  # Increased height
        left_hp_x = int(w * 0.230)
        left_hp_y = int(h * 0.145)  # Moved up slightly
        self.left_tower_hp_roi = ROI("LEFT_HP", left_hp_x, left_hp_y, hp_width, hp_height)
        
        # Right tower HP (tuned position - adjusted for full digit capture)
        right_hp_x = int(w * 0.730)
        right_hp_y = int(h * 0.145)  # Moved up slightly
        self.right_tower_hp_roi = ROI("RIGHT_HP", right_hp_x, right_hp_y, hp_width, hp_height)
        
        # Elixir phase indicator (near elixir bar)
        elixir_width = int(w * 0.08)
        elixir_height = int(h * 0.04)
        elixir_x = int(w * 0.46)
        elixir_y = int(h * 0.80)
        self.elixir_indicator_roi = ROI("ELIXIR", elixir_x, elixir_y, elixir_width, elixir_height)
        
        if self.DEBUG_OCR:
            print(f"[OCR] Timer ROI: {self.timer_roi}")
            print(f"[OCR] Left HP ROI: {self.left_tower_hp_roi}")
            print(f"[OCR] Right HP ROI: {self.right_tower_hp_roi}")
            print(f"[OCR] Elixir ROI: {self.elixir_indicator_roi}")
    
    def capture_screen(self) -> np.ndarray:
        """Capture the emulator region and convert to grayscale."""
        screenshot = self.sct.grab(self.monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame, gray
    
    def detect_motion(self, current_gray: np.ndarray) -> np.ndarray:
        """Detect motion using frame differencing."""
        if self.previous_frame is None:
            self.previous_frame = current_gray
            return np.zeros_like(current_gray)
        
        # Frame difference
        diff = cv2.absdiff(self.previous_frame, current_gray)
        
        # Threshold to binary
        _, motion_mask = cv2.threshold(diff, self.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        self.previous_frame = current_gray
        return motion_mask
    
    def detect_spawn_event(self, current_gray: np.ndarray, roi: ROI, roi_name: str) -> bool:
        """Detect if a spawn event occurred in the ROI."""
        y1, y2, x1, x2 = roi.get_slice()
        roi_region = current_gray[y1:y2, x1:x2]
        
        # Initialize background if not exists
        if roi_name not in self.roi_backgrounds:
            self.roi_backgrounds[roi_name] = roi_region.copy()
            return False
        
        # Compare with background
        background = self.roi_backgrounds[roi_name]
        diff = cv2.absdiff(background, roi_region)
        _, thresh = cv2.threshold(diff, self.MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)
        
        # Count changed pixels
        changed_pixels = np.count_nonzero(thresh)
        
        # Spawn event if enough pixels changed
        spawn_detected = changed_pixels > self.SPAWN_PIXEL_THRESHOLD
        
        # Update spawn buffer
        self.roi_spawn_buffer[roi_name].append(spawn_detected)
        
        # Update background slowly (running average)
        alpha = 0.05
        self.roi_backgrounds[roi_name] = cv2.addWeighted(
            roi_region, alpha, background, 1 - alpha, 0
        ).astype(np.uint8)
        
        # Confirm spawn if detected in required number of frames
        if len(self.roi_spawn_buffer[roi_name]) >= self.SPAWN_FRAMES_REQUIRED:
            confirmed = sum(self.roi_spawn_buffer[roi_name]) >= self.SPAWN_FRAMES_REQUIRED
            if confirmed and self.DEBUG:
                print(f"[DEBUG] Spawn event confirmed in {roi_name}")
            return confirmed
        
        return False
    
    def is_roi_in_cooldown(self, roi_name: str, current_time: float) -> bool:
        """Check if ROI is in cooldown period."""
        last_time = self.roi_last_detection[roi_name]
        in_cooldown = (current_time - last_time) < self.ROI_COOLDOWN
        if in_cooldown and self.DEBUG:
            print(f"[DEBUG] {roi_name} in cooldown")
        return in_cooldown
    
    def extract_contours(self, motion_mask: np.ndarray, roi: ROI) -> List[Tuple]:
        """Extract contours from motion mask within an ROI."""
        y1, y2, x1, x2 = roi.get_slice()
        roi_mask = motion_mask[y1:y2, x1:x2]
        
        contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid in global coordinates
            centroid_x = x1 + x + w // 2
            centroid_y = y1 + y + h // 2
            
            results.append({
                'contour': contour,
                'area': area,
                'bbox': (x + x1, y + y1, w, h),
                'centroid': (centroid_x, centroid_y),
                'width': w,
                'height': h
            })
        
        return results
    
    def calculate_velocity(self, x: int, y: int) -> float:
        """Calculate velocity of object by comparing with previous positions."""
        # Find closest tracked object
        min_dist = float('inf')
        closest_id = None
        
        for obj_id, data in self.tracked_objects.items():
            prev_x, prev_y, _ = data
            dist = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
            if dist < min_dist and dist < 150:  # Must be within 150 pixels
                min_dist = dist
                closest_id = obj_id
        
        # Create or update tracked object
        obj_id = closest_id if closest_id is not None else len(self.tracked_objects)
        
        if closest_id is not None:
            velocity = min_dist  # Distance moved = velocity
        else:
            velocity = 0.0
        
        self.tracked_objects[obj_id] = (x, y, time.time())
        
        # Clean old tracked objects (>1 second old)
        current_time = time.time()
        to_remove = [k for k, v in self.tracked_objects.items() if current_time - v[2] > 1.0]
        for k in to_remove:
            del self.tracked_objects[k]
        
        return velocity
    
    def extract_features(self, contour_data: dict, frame_color: np.ndarray, roi_name: str) -> TroopFeatures:
        """Extract comprehensive OpenCV features from a contour."""
        # Geometric features
        contour = contour_data['contour']
        area = contour_data['area']
        x, y, w, h = contour_data['bbox']
        centroid = contour_data['centroid']
        
        # Aspect ratio (axis-aligned)
        aspect_ratio = h / w if w > 0 else 0.0
        
        # Also compute oriented bounding box for Log detection
        if len(contour) >= 5:  # minAreaRect requires 5+ points
            oriented_rect = cv2.minAreaRect(contour)
            oriented_w, oriented_h = oriented_rect[1]
            # Ensure width > height by convention
            if oriented_h > oriented_w:
                oriented_w, oriented_h = oriented_h, oriented_w
            oriented_aspect = oriented_h / oriented_w if oriented_w > 0 else 0.0
        else:
            oriented_aspect = aspect_ratio
        
        # Store oriented aspect ratio in contour_data for later use
        contour_data['oriented_aspect'] = oriented_aspect
        
        # Solidity (contour area / bounding box area)
        bbox_area = w * h
        solidity = area / bbox_area if bbox_area > 0 else 0.0
        
        # Compactness (area / perimeter^2) - measures circularity
        perimeter = cv2.arcLength(contour, True)
        compactness = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0.0
        
        # Motion features - track this object with vx/vy
        velocity, velocity_x, velocity_y, lifetime, object_id = self.track_object(centroid)
        
        # Determine direction
        if abs(velocity_x) > abs(velocity_y) * 1.5:
            direction = "horizontal"
        elif abs(velocity_y) > abs(velocity_x) * 1.5:
            direction = "vertical"
        elif velocity < 3:
            direction = "stationary"
        else:
            direction = "diagonal"
        
        # Color features
        roi_region = frame_color[y:y+h, x:x+w]
        if roi_region.size > 0:
            mean_rgb = tuple(np.mean(roi_region, axis=(0, 1)))
            b, g, r = mean_rgb
            
            # Determine dominant color
            if r > 150 and r > g * 1.3 and r > b * 1.5:
                dominant_color = "red"
            elif b > 100 and b > r * 1.2 and b > g * 1.1:
                dominant_color = "blue"
            else:
                dominant_color = "neutral"
        else:
            mean_rgb = (0, 0, 0)
            dominant_color = "neutral"
        
        features = TroopFeatures(
            area=area,
            width=w,
            height=h,
            aspect_ratio=aspect_ratio,
            solidity=solidity,
            compactness=compactness,
            velocity=velocity,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            direction=direction,
            lifetime=lifetime,
            centroid=centroid,
            bbox=(x, y, w, h),
            mean_rgb=mean_rgb,
            dominant_color=dominant_color,
            spawn_roi=roi_name,
            object_id=object_id
        )
        
        if self.DEBUG_FEATURES:
            print(f"[FEATURES] area={area:.0f} aspect={aspect_ratio:.2f} solid={solidity:.2f} " +
                  f"comp={compactness:.3f} vel={velocity:.1f} dir={direction} life={lifetime} roi={roi_name}")
        
        return features
    
    def track_object(self, centroid: Tuple[int, int]) -> Tuple[float, float, float, int, int]:
        """Track object across frames and return motion features."""
        x, y = centroid
        current_time = time.time()
        
        # Find closest existing tracked object
        min_dist = float('inf')
        closest_id = None
        
        for obj_id, obj_data in self.tracked_objects.items():
            if 'centroid_history' not in obj_data:
                continue
            
            last_centroid = obj_data['centroid_history'][-1]
            dist = np.sqrt((x - last_centroid[0])**2 + (y - last_centroid[1])**2)
            
            if dist < min_dist and dist < 150:  # Match within 150 pixels
                min_dist = dist
                closest_id = obj_id
        
        # Create new object or update existing
        if closest_id is None:
            obj_id = self.next_object_id
            self.next_object_id += 1
            self.tracked_objects[obj_id] = {
                'centroid_history': deque([(x, y)], maxlen=10),
                'first_seen': current_time,
                'last_seen': current_time
            }
            velocity = 0.0
            velocity_x = 0.0
            velocity_y = 0.0
            lifetime = 1
        else:
            obj_id = closest_id
            obj_data = self.tracked_objects[obj_id]
            obj_data['centroid_history'].append((x, y))
            obj_data['last_seen'] = current_time
            
            # Calculate velocity from recent history
            history = list(obj_data['centroid_history'])
            if len(history) >= 2:
                prev_x, prev_y = history[-2]
                velocity_x = x - prev_x
                velocity_y = y - prev_y
                velocity = np.sqrt(velocity_x**2 + velocity_y**2)
            else:
                velocity = 0.0
                velocity_x = 0.0
                velocity_y = 0.0
            
            # Calculate lifetime in frames
            lifetime = len(history)
        
        # Clean old objects (>1.5 seconds since last seen)
        to_remove = [k for k, v in self.tracked_objects.items() 
                     if current_time - v['last_seen'] > 1.5]
        for k in to_remove:
            del self.tracked_objects[k]
        
        return velocity, velocity_x, velocity_y, lifetime, obj_id
    
    def preprocess_timer(self, roi_image: np.ndarray) -> np.ndarray:
        """Preprocess timer ROI for OCR."""
        # Grayscale
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image
        
        # Upscale 2.5x
        scaled = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        
        # Otsu threshold
        _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological close to connect broken digits
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def preprocess_hp(self, roi_image: np.ndarray) -> np.ndarray:
        """Preprocess HP ROI for OCR."""
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image
        
        # Upscale 3x
        scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        # Strong threshold for white text
        _, thresh = cv2.threshold(scaled, 200, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to reduce noise
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_open)
        
        return processed
    
    def preprocess_elixir(self, roi_image: np.ndarray) -> np.ndarray:
        """Preprocess elixir indicator ROI for OCR."""
        if len(roi_image.shape) == 3:
            gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi_image
        
        # Upscale 3x
        scaled = cv2.resize(gray, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
        
        # Threshold for UI text
        _, thresh = cv2.threshold(scaled, 150, 255, cv2.THRESH_BINARY)
        
        return thresh
    
    def ocr_timer(self, frame_color: np.ndarray, frame_gray: np.ndarray) -> Optional[int]:
        """Extract match timer in seconds using EasyOCR."""
        if not EASYOCR_AVAILABLE or self.easyocr_reader is None:
            return None
        
        y1, y2, x1, x2 = self.timer_roi.get_slice()
        roi_color = frame_color[y1:y2, x1:x2]
        
        # Upscale for better OCR
        scaled = cv2.resize(roi_color, None, fx=5.0, fy=5.0, interpolation=cv2.INTER_CUBIC)
        
        if self.DEBUG_OCR:
            cv2.imshow("Timer ROI", cv2.resize(scaled, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))
        
        try:
            # EasyOCR - read text
            result = self.easyocr_reader.readtext(scaled, detail=0, allowlist='0123456789:')
            
            if not result:
                print("[OCR-RAW] Timer: no text detected")
                return None
            
            # Get first result
            text = result[0].strip()
            print(f"[OCR-RAW] Timer: '{text}'")
            
            # Parse M:SS or MM:SS
            match = re.match(r'(\d{1,2}):(\d{2})', text)
            if match:
                minutes = int(match.group(1))
                seconds = int(match.group(2))
                
                if seconds >= 60:
                    print(f"[OCR-REJECT] Timer invalid: seconds >= 60")
                    return None
                
                total_seconds = minutes * 60 + seconds
                
                # Validate monotonic decrease
                if self.match_time_seconds is not None:
                    if total_seconds > self.match_time_seconds + 5:
                        print(f"[OCR-REJECT] Timer non-monotonic: {self.match_time_seconds} -> {total_seconds}")
                        return None
                
                print(f"[OCR-ACCEPT] Timer: {total_seconds}s ({minutes}:{seconds:02d})")
                return total_seconds
            
            print(f"[OCR-REJECT] Timer parse failed: '{text}'")
            return None
            
        except Exception as e:
            print(f"[OCR-ERROR] Timer: {e}")
            return None
        
        # OCR with digit + colon whitelist
        config = '--psm 7 -c tessedit_char_whitelist=0123456789:'
        text = pytesseract.image_to_string(processed, config=config).strip()
        
        # Always log raw OCR
        print(f"[OCR-RAW] Timer: '{text}'")
        
        # Parse M:SS or MM:SS
        match = re.match(r'(\d{1,2}):(\d{2})', text)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            
            # Validate
            if seconds >= 60:
                if self.DEBUG_OCR:
                    print(f"[OCR] Timer invalid: seconds >= 60")
                return None
            
            total_seconds = minutes * 60 + seconds
            
            # Validate monotonic decrease
            if self.match_time_seconds is not None:
                if total_seconds > self.match_time_seconds + 5:  # Allow small jumps for overtime
                    if self.DEBUG_OCR:
                        print(f"[OCR] Timer rejected: non-monotonic increase")
                    return None
            
            return total_seconds
        
        return None
    
    def ocr_tower_hp(self, frame_gray: np.ndarray, tower: str) -> Optional[int]:
        """Extract tower HP (tower = 'left' or 'right')."""
        if not PYTESSERACT_AVAILABLE:
            return None
        
        roi_obj = self.left_tower_hp_roi if tower == 'left' else self.right_tower_hp_roi
        y1, y2, x1, x2 = roi_obj.get_slice()
        roi = frame_gray[y1:y2, x1:x2]
        
        processed = self.preprocess_hp(roi)
        
        if self.DEBUG_OCR:
            cv2.imshow(f"{tower.capitalize()} HP OCR", processed)
        
        # OCR with digits only
        config = '--psm 7 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(processed, config=config).strip()
        
        # Always log raw OCR for debugging
        print(f"[OCR-RAW] {tower.capitalize()} HP: '{text}'")
        
        # Parse integer
        try:
            hp = int(text)
            
            # Validate range (Princess Towers have ~2400 max HP)
            if hp < 0 or hp > 5000:
                print(f"[OCR-REJECT] {tower} HP out of range: {hp}")
                return None
            
            # Validate monotonic decrease
            current_hp = self.left_tower_hp if tower == 'left' else self.right_tower_hp
            if current_hp is not None:
                if hp > current_hp:
                    print(f"[OCR-REJECT] {tower} HP increased: {current_hp} -> {hp}")
                    return None
                if current_hp - hp > 2000:
                    print(f"[OCR-REJECT] {tower} HP drop too large: {current_hp} -> {hp}")
                    return None
            
            print(f"[OCR-ACCEPT] {tower} HP: {hp}")
            return hp
        except ValueError:
                print(f"[OCR-REJECT] {tower} HP parse failed: '{text}'")
                return None
    
    def ocr_elixir_indicator(self, frame_gray: np.ndarray) -> Optional[str]:
        """Extract elixir phase indicator (1x, 2x, 3x)."""
        if not PYTESSERACT_AVAILABLE:
            return None
        
        y1, y2, x1, x2 = self.elixir_indicator_roi.get_slice()
        roi = frame_gray[y1:y2, x1:x2]
        
        processed = self.preprocess_elixir(roi)
        
        if self.DEBUG_OCR:
            cv2.imshow("Elixir OCR", processed)
        
        # OCR with digits + x
        config = '--psm 7 -c tessedit_char_whitelist=0123456789xX'
        text = pytesseract.image_to_string(processed, config=config).strip().lower()
        
        if self.DEBUG_OCR:
            print(f"[OCR] Elixir raw: '{text}'")
        
        # Parse 1x, 2x, 3x
        if '1' in text and 'x' in text:
            return "1x"
        elif '2' in text and 'x' in text:
            return "2x"
        elif '3' in text and 'x' in text:
            return "3x"
        
        return None
    
    def infer_elixir_phase_from_time(self, match_seconds_remaining: int, overtime: bool) -> str:
        """Infer elixir phase from match timer as fallback."""
        if overtime:
            return "2x"  # Most modes use 2x in overtime
        
        if self.mode_profile == "triple_elixir":
            if match_seconds_remaining <= 60:
                return "3x"
            elif match_seconds_remaining <= 120:
                return "2x"
            else:
                return "1x"
        else:
            # Standard ladder: 2x at 1:00 remaining
            if match_seconds_remaining <= self.double_elixir_start:
                return "2x"
            else:
                return "1x"
    
    def update_game_state(self, frame_color: np.ndarray, frame_gray: np.ndarray):
        """Update game state via OCR with temporal gating."""
        if not self.ENABLE_OCR:
            return  # OCR disabled
        
        current_time = time.time() - self.start_time
        
        # OCR timer
        if current_time - self.last_ocr_timer >= self.OCR_TIMER_INTERVAL:
            timer_result = self.ocr_timer(frame_color, frame_gray)
            if timer_result is not None:
                self.timer_history.append(timer_result)
                # Smooth with median
                if len(self.timer_history) >= 2:
                    self.match_time_seconds = int(np.median(list(self.timer_history)))
                else:
                    self.match_time_seconds = timer_result
            self.last_ocr_timer = current_time
        
        # OCR tower HP
        if current_time - self.last_ocr_hp >= self.OCR_HP_INTERVAL:
            left_hp = self.ocr_tower_hp(frame_gray, 'left')
            if left_hp is not None:
                self.left_hp_history.append(left_hp)
                if len(self.left_hp_history) >= 2:
                    self.left_tower_hp = int(np.median(list(self.left_hp_history)))
                else:
                    self.left_tower_hp = left_hp
            
            right_hp = self.ocr_tower_hp(frame_gray, 'right')
            if right_hp is not None:
                self.right_hp_history.append(right_hp)
                if len(self.right_hp_history) >= 2:
                    self.right_tower_hp = int(np.median(list(self.right_hp_history)))
                else:
                    self.right_tower_hp = right_hp
            
            self.last_ocr_hp = current_time
        
        # OCR elixir phase
        if current_time - self.last_ocr_elixir >= self.OCR_ELIXIR_INTERVAL:
            elixir_ocr = self.ocr_elixir_indicator(frame_gray)
            if elixir_ocr is not None:
                self.elixir_history.append(elixir_ocr)
                
                # Require 2/3 consistency before changing
                if len(self.elixir_history) >= 2:
                    recent = list(self.elixir_history)[-3:]
                    counts = {phase: recent.count(phase) for phase in set(recent)}
                    most_common = max(counts, key=counts.get)
                    
                    if counts[most_common] >= 2:
                        # Validate transition (only forward)
                        phases = ["1x", "2x", "3x"]
                        current_idx = phases.index(self.elixir_phase)
                        new_idx = phases.index(most_common)
                        
                        if new_idx >= current_idx:
                            if new_idx != current_idx and self.DEBUG_OCR:
                                print(f"[OCR] Elixir phase changed: {self.elixir_phase} -> {most_common} (OCR)")
                            self.elixir_phase = most_common
            
            # Fallback: infer from timer
            if self.match_time_seconds is not None and elixir_ocr is None:
                inferred = self.infer_elixir_phase_from_time(self.match_time_seconds, self.match_time_seconds == 0)
                if inferred != self.elixir_phase:
                    if self.DEBUG_OCR:
                        print(f"[OCR] Elixir phase changed: {self.elixir_phase} -> {inferred} (timer fallback)")
            
            self.last_ocr_elixir = current_time
    
    def get_clock_hp_elixir(self) -> Dict:
        """Get current game state as dictionary."""
        return {
            "time": self.match_time_seconds,
            "left_tower_hp": self.left_tower_hp,
            "right_tower_hp": self.right_tower_hp,
            "elixir_phase": self.elixir_phase
        }
    
    def classify_card(self, features: TroopFeatures, contour_data: dict = None) -> Optional[Tuple[str, float]]:
        """Classify card using explicit OpenCV features and multi-factor scoring."""
        
        if contour_data is None:
            contour_data = {}
        
        # Check each troop against feature specs
        candidates = []
        
        for troop_name, specs in self.TROOP_SPECS.items():
            score = 0.0
            reasons = []
            penalties = []
            
            # 1. Area check (mandatory)
            if 'area_range' in specs:
                min_area, max_area = specs['area_range']
                if min_area <= features.area <= max_area:
                    score += 0.25
                    reasons.append(f"area_match({features.area:.0f})")
                else:
                    penalties.append(f"area_mismatch({features.area:.0f} not in {specs['area_range']})")
                    continue  # Hard reject
            
            # 2. Aspect ratio check
            if 'aspect_ratio_range' in specs:
                min_ar, max_ar = specs['aspect_ratio_range']
                if min_ar <= features.aspect_ratio <= max_ar:
                    score += 0.20
                    reasons.append(f"aspect_ok({features.aspect_ratio:.2f})")
                else:
                    penalties.append(f"aspect_bad({features.aspect_ratio:.2f})")
                    continue
            elif 'aspect_ratio_max' in specs:
                if features.aspect_ratio <= specs['aspect_ratio_max']:
                    score += 0.20
                    reasons.append(f"aspect_low({features.aspect_ratio:.2f})")
                else:
                    penalties.append(f"aspect_too_high({features.aspect_ratio:.2f})")
                    continue
            
            # 3. Velocity checks
            if 'min_velocity' in specs:
                if features.velocity >= specs['min_velocity']:
                    score += 0.20
                    reasons.append(f"velocity_high({features.velocity:.1f})")
                else:
                    penalties.append(f"velocity_too_low({features.velocity:.1f}<{specs['min_velocity']})")
                    continue
            
            if 'max_velocity' in specs:
                if features.velocity <= specs['max_velocity']:
                    score += 0.20
                    reasons.append(f"velocity_low({features.velocity:.1f})")
                else:
                    penalties.append(f"velocity_too_high({features.velocity:.1f}>{specs['max_velocity']})")
                    continue
            
            if 'velocity_range' in specs:
                min_vel, max_vel = specs['velocity_range']
                if min_vel <= features.velocity <= max_vel:
                    score += 0.20
                    reasons.append(f"velocity_range({features.velocity:.1f})")
                else:
                    penalties.append(f"velocity_out_of_range({features.velocity:.1f})")
                    continue
            
            # 4. Spawn location check (mandatory)
            if 'spawn_rois' in specs:
                roi_matched = any(roi_keyword in features.spawn_roi for roi_keyword in specs['spawn_rois'])
                if roi_matched:
                    score += 0.15
                    reasons.append(f"spawn_ok({features.spawn_roi})")
                else:
                    penalties.append(f"wrong_spawn({features.spawn_roi})")
                    continue
            
            # 5. Lifetime check
            if 'lifetime_range' in specs:
                min_life, max_life = specs['lifetime_range']
                if min_life <= features.lifetime <= max_life:
                    score += 0.10
                    reasons.append(f"lifetime_ok({features.lifetime})")
                elif features.lifetime < min_life:
                    penalties.append(f"lifetime_too_short({features.lifetime}<{min_life})")
                    # Partial penalty for new objects
                    if features.lifetime >= min_life - 1:
                        score += 0.05
            
            # 6. Solidity check
            if 'solidity_min' in specs:
                if features.solidity >= specs['solidity_min']:
                    score += 0.10
                    reasons.append(f"solidity_ok({features.solidity:.2f})")
                else:
                    penalties.append(f"solidity_low({features.solidity:.2f})")
            
            # 7. Compactness check (for circular objects)
            if 'compactness_min' in specs:
                if features.compactness >= specs['compactness_min']:
                    score += 0.15
                    reasons.append(f"compact({features.compactness:.2f})")
                else:
                    penalties.append(f"not_compact({features.compactness:.2f})")
            
            # 8. Color requirement
            if 'color_required' in specs:
                if features.dominant_color == specs['color_required']:
                    score += 0.15
                    reasons.append(f"color_{specs['color_required']}")
                else:
                    penalties.append(f"color_mismatch(got_{features.dominant_color})")
                    score -= 0.10  # Penalty but not hard reject
            
            # 9. Directional velocity checks (vx/vy based)
            if 'min_velocity_x' in specs:
                if abs(features.velocity_x) >= specs['min_velocity_x']:
                    score += 0.20
                    reasons.append(f"vx_sufficient({features.velocity_x:.1f})")
                else:
                    penalties.append(f"vx_too_low({abs(features.velocity_x):.1f}<{specs['min_velocity_x']})")
                    continue
            
            if 'max_velocity_y' in specs:
                if abs(features.velocity_y) <= specs['max_velocity_y']:
                    score += 0.15
                    reasons.append(f"vy_ok({features.velocity_y:.1f})")
                else:
                    penalties.append(f"vy_too_high({abs(features.velocity_y):.1f}>{specs['max_velocity_y']})")
                    continue
            
            if 'min_vy' in specs:
                # For Hog - must move vertically (positive = down toward opponent)
                if abs(features.velocity_y) >= specs['min_vy']:
                    score += 0.15
                    reasons.append(f"vy_forward({features.velocity_y:.1f})")
                else:
                    if features.lifetime > 2:  # Give a frame or two to start moving
                        penalties.append(f"vy_insufficient({features.velocity_y:.1f})")
                        continue
            
            # 10. Static requirement for Cannon
            if 'require_static' in specs and specs['require_static']:
                if features.lifetime > 2 and features.velocity > specs.get('max_velocity', 8):
                    penalties.append(f"cannon_still_moving({features.velocity:.1f})")
                    continue
                elif features.lifetime > 2:
                    score += 0.15
                    reasons.append(f"static_confirmed")
            
            # 11. Oriented bbox check for Log
            if 'use_oriented_bbox' in specs and specs['use_oriented_bbox']:
                # Use oriented aspect ratio if available
                oriented_aspect = contour_data.get('oriented_aspect', features.aspect_ratio)
                if oriented_aspect <= specs.get('aspect_ratio_max', 0.8):
                    score += 0.10
                    reasons.append(f"oriented_flat({oriented_aspect:.2f})")
                else:
                    penalties.append(f"oriented_not_flat({oriented_aspect:.2f})")
            
            # Add candidate with confidence score
            if score > 0.5:  # Minimum viable score
                candidates.append((troop_name, score, reasons, penalties))
            elif self.DEBUG:
                print(f"[DEBUG] {troop_name} rejected: score={score:.2f} penalties={penalties}")
        
        # Return best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_troop, best_score, reasons, penalties = candidates[0]
            
            if self.DEBUG:
                print(f"[DEBUG] CLASSIFIED as {best_troop} (score={best_score:.2f}): {', '.join(reasons)}")
            
            return (best_troop, best_score)
        
        if self.DEBUG:
            print(f"[DEBUG] No classification: all troops rejected")
        return None
    
    def check_skeleton_group(self, contours_in_roi: List[dict]) -> Optional[Tuple[str, float]]:
        """Check if multiple small contours form a skeleton group."""
        small_contours = [c for c in contours_in_roi if 150 <= c['area'] <= 600]
        
        if len(small_contours) >= 3:  # At least 3 skeletons
            # Check they're close together
            centroids = [c['centroid'] for c in small_contours]
            avg_x = sum(c[0] for c in centroids) / len(centroids)
            avg_y = sum(c[1] for c in centroids) / len(centroids)
            
            # All should be within ~200 pixels of group center
            max_dist = max(np.sqrt((c[0]-avg_x)**2 + (c[1]-avg_y)**2) for c in centroids)
            
            if max_dist < 200:
                if self.DEBUG:
                    print(f"[DEBUG] Skeleton group detected: {len(small_contours)} units, max_dist={max_dist:.0f}")
                return ("Skeletons", 0.80)
        
        return None
    
    def determine_lane(self, centroid: Tuple[int, int], roi_name: str) -> str:
        """Determine which lane the card was played in."""
        x, y = centroid
        
        if "CENTER" in roi_name:
            return "CENTER"
        
        # Use horizontal position to determine lane
        third = self.screen_width // 3
        
        if x < third:
            return "LEFT"
        elif x > 2 * third:
            return "RIGHT"
        else:
            return "CENTER"
    
    def determine_player(self, roi_name: str, centroid: Tuple[int, int]) -> str:
        """Determine if card was played by opponent or player."""
        _, y = centroid
        
        # Opponent plays are in top half, player plays in bottom half
        if "OPP" in roi_name or y < self.screen_height * 0.4:
            return "OPPONENT"
        else:
            return "PLAYER"
    
    def is_duplicate(self, detection: Detection) -> bool:
        """Check if this detection is a duplicate of a recent one."""
        current_time = detection.timestamp
        
        for recent in self.recent_detections:
            time_diff = current_time - recent.timestamp
            
            # Check if within suppression window
            if time_diff < self.DUPLICATE_SUPPRESS_TIME:
                # Same card, same lane, same player
                if (recent.card == detection.card and 
                    recent.lane == detection.lane and 
                    recent.player == detection.player):
                    
                    # Check spatial proximity
                    dx = detection.position[0] - recent.position[0]
                    dy = detection.position[1] - recent.position[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 100:  # Within 100 pixels
                        return True
        
        return False
    
    def detect_hand_change(self, current_gray: np.ndarray) -> bool:
        """Detect if a card was played by checking hand slot changes."""
        current_time = time.time() - self.start_time
        
        # Extract current slot images
        current_slots = []
        for slot in self.hand_slots:
            y1, y2, x1, x2 = slot.get_slice()
            slot_img = current_gray[y1:y2, x1:x2]
            
            # Apply Gaussian blur to reduce noise
            slot_img = cv2.GaussianBlur(slot_img, (5, 5), 0)
            current_slots.append(slot_img)
        
        # Initialize on first frame
        if len(self.previous_hand_slots) == 0:
            self.previous_hand_slots = current_slots
            return False
        
        # Check debounce
        if (current_time - self.last_hand_change_time) < self.HAND_DEBOUNCE_TIME:
            if self.DEBUG_HAND:
                print(f"[HAND DEBUG] Debounce active, suppressing check")
            self.previous_hand_slots = current_slots
            return False
        
        # Compute changes for each slot
        slot_changes = []
        slot_percentages = []
        
        for i, (current, previous) in enumerate(zip(current_slots, self.previous_hand_slots)):
            # Ensure same size
            if current.shape != previous.shape:
                if self.DEBUG_HAND:
                    print(f"[HAND DEBUG] Slot {i} shape mismatch, skipping")
                slot_changes.append(0)
                slot_percentages.append(0.0)
                continue
            
            total_pixels = current.shape[0] * current.shape[1]
            
            # Method 1: Pixel difference (for complete card replacement)
            diff = cv2.absdiff(previous, current)
            
            # Higher threshold to ignore selection animations
            _, thresh = cv2.threshold(diff, 50, 255, cv2.THRESH_BINARY)
            
            # Count changed pixels
            changed_pixels = np.count_nonzero(thresh)
            change_percentage = changed_pixels / total_pixels if total_pixels > 0 else 0.0
            
            slot_changes.append(changed_pixels)
            slot_percentages.append(change_percentage)
            
            if self.DEBUG_HAND:
                print(f"[HAND DEBUG] Slot {i}: {changed_pixels} pixels ({change_percentage*100:.1f}%)")
        
        # Update previous slots
        self.previous_hand_slots = current_slots
        
        # Check if exactly one slot changed significantly
        # Require BOTH high pixel count AND high percentage
        slots_above_threshold = sum(
            1 for pixels, pct in zip(slot_changes, slot_percentages)
            if pixels > self.HAND_CHANGE_THRESHOLD and pct > self.HAND_CHANGE_PERCENTAGE
        )
        
        if slots_above_threshold == 1:
            # Find which slot changed
            changed_slot = next(
                i for i, (pixels, pct) in enumerate(zip(slot_changes, slot_percentages))
                if pixels > self.HAND_CHANGE_THRESHOLD and pct > self.HAND_CHANGE_PERCENTAGE
            )
            
            if self.DEBUG_HAND:
                print(f"[HAND DEBUG] *** Slot {changed_slot} CARD REPLACED: {slot_changes[changed_slot]} pixels ({slot_percentages[changed_slot]*100:.1f}%) ***")
            
            self.last_hand_change_time = current_time
            return True
        
        elif slots_above_threshold > 1 and self.DEBUG_HAND:
            print(f"[HAND DEBUG] Multiple slots changed ({slots_above_threshold}), likely not a card play")
        elif slots_above_threshold == 0 and max(slot_changes) > self.HAND_CHANGE_THRESHOLD * 0.3 and self.DEBUG_HAND:
            print(f"[HAND DEBUG] Change detected but below threshold - likely card selection animation")
        
        return False
    
    def was_hand_changed_recently(self, current_time: float) -> bool:
        """Check if a hand change occurred within the valid time window."""
        time_since_change = current_time - self.last_hand_change_time
        return time_since_change <= self.HAND_VALID_WINDOW
    
    def process_frame(self, frame_color: np.ndarray, frame_gray: np.ndarray):
        """Process a single frame and detect card plays."""
        current_time = time.time() - self.start_time
        
        # Update game state via OCR (timer, HP, elixir)
        self.update_game_state(frame_color, frame_gray)
        
        # Detect hand changes
        hand_changed = self.detect_hand_change(frame_gray)
        if hand_changed:
            print(f"[HAND] Card played detected at t={current_time:.2f}")
        
        # Detect motion
        motion_mask = self.detect_motion(frame_gray)
        
        # Process each ROI
        detections = []
        
        for roi_name, roi in self.rois.items():
            # Check cooldown first
            if self.is_roi_in_cooldown(roi_name, current_time):
                continue
            
            # Detect spawn event
            spawn_event = self.detect_spawn_event(frame_gray, roi, roi_name)
            
            if not spawn_event:
                if self.DEBUG and np.count_nonzero(motion_mask) > 0:
                    print(f"[DEBUG] No spawn event in {roi_name}, skipping classification")
                continue
            
            # Extract contours only after spawn event confirmed
            contours = self.extract_contours(motion_mask, roi)
            
            # Check for skeleton group first (multiple small contours)
            skeleton_result = self.check_skeleton_group(contours)
            if skeleton_result:
                card_name, confidence = skeleton_result
                
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    # Use first contour centroid as representative position
                    centroid = contours[0]['centroid']
                    lane = self.determine_lane(centroid, roi_name)
                    player = self.determine_player(roi_name, centroid)
                    
                    detection = Detection(
                        timestamp=current_time,
                        player=player,
                        card=card_name,
                        lane=lane,
                        confidence=confidence,
                        position=centroid,
                        features=None
                    )
                    
                    if not self.is_duplicate(detection):
                        detections.append(detection)
                continue  # Skip individual skeleton contours
            
            # Process individual contours
            for contour_data in contours:
                # Extract comprehensive features
                features = self.extract_features(contour_data, frame_color, roi_name)
                
                # Classify based on features (pass contour_data for oriented bbox)
                result = self.classify_card(features, contour_data)
                
                if result is None:
                    if self.DEBUG:
                        print(f"[DEBUG] Classification returned None in {roi_name}")
                    continue
                
                card_name, confidence = result
                
                if confidence < self.CONFIDENCE_THRESHOLD:
                    if self.DEBUG:
                        print(f"[DEBUG] {card_name} rejected: low confidence {confidence:.2f}")
                    continue
                
                # Determine lane and player
                centroid = features.centroid
                lane = self.determine_lane(centroid, roi_name)
                player = self.determine_player(roi_name, centroid)
                
                # Create detection
                detection = Detection(
                    timestamp=current_time,
                    player=player,
                    card=card_name,
                    lane=lane,
                    confidence=confidence,
                    position=centroid,
                    features=features
                )
                
                # Check for duplicates
                if not self.is_duplicate(detection):
                    detections.append(detection)
        
        # Add to frame buffer for temporal smoothing
        self.frame_buffer.append(detections)
        
        # Confirm detections that appear in multiple frames
        self.confirm_detections()
    
    def confirm_detections(self):
        """Confirm detections that appear consistently across multiple frames."""
        if len(self.frame_buffer) < 2:  # Minimum 2 frames
            return
        
        # Get all detections from buffer
        all_detections = []
        for frame_detections in self.frame_buffer:
            all_detections.extend(frame_detections)
        
        # Group similar detections
        confirmed = []
        seen = set()
        
        for detection in all_detections:
            detection_key = (detection.card, detection.lane, detection.player)
            
            if detection_key in seen:
                continue
            
            # Count how many times this card appears in recent frames
            count = sum(1 for d in all_detections 
                       if d.card == detection.card 
                       and d.lane == detection.lane 
                       and d.player == detection.player)
            
            # Check if troop allows quick confirmation
            troop_spec = self.TROOP_SPECS.get(detection.card, {})
            quick_confirm = troop_spec.get('quick_confirm', False)
            
            # Quick confirm troops (Hog, Cannon, Log): 2 frames sufficient
            # Others: require majority (>= 2 out of 3 frames)
            min_required = 2 if quick_confirm else 2
            frames_available = len(self.frame_buffer)
            
            if quick_confirm and count >= 2:
                if not self.is_duplicate(detection):
                    confirmed.append(detection)
                    seen.add(detection_key)
            elif frames_available >= self.TEMPORAL_FRAMES and count >= min_required:
                if not self.is_duplicate(detection):
                    confirmed.append(detection)
                    seen.add(detection_key)
        
        # Report confirmed detections and activate cooldown
        for detection in confirmed:
            self.report_detection(detection)
            self.recent_detections.append(detection)
            
            # Activate cooldown for the ROI
            for roi_name in self.rois.keys():
                roi = self.rois[roi_name]
                x1, x2 = roi.x, roi.x + roi.width
                y1, y2 = roi.y, roi.y + roi.height
                dx, dy = detection.position
                
                if x1 <= dx <= x2 and y1 <= dy <= y2:
                    self.roi_last_detection[roi_name] = detection.timestamp + (time.time() - self.start_time) - detection.timestamp
                    if self.DEBUG:
                        print(f"[DEBUG] Cooldown activated for {roi_name}")
                    break
    
    def report_detection(self, detection: Detection):
        """Print a detection event to console."""
        print(f"[{detection.timestamp:.2f}] "
              f"PLAYER={detection.player} "
              f"CARD={detection.card} "
              f"LANE={detection.lane} "
              f"CONFIDENCE={detection.confidence:.2f}")
    
    def run(self):
        """Main detection loop."""
        fps = 5  # Target frames per second
        frame_delay = 1.0 / fps
        
        try:
            while True:
                loop_start = time.time()
                
                # Capture and process frame
                frame_color, frame_gray = self.capture_screen()
                self.process_frame(frame_color, frame_gray)
                
                # Maintain target FPS
                elapsed = time.time() - loop_start
                sleep_time = max(0, frame_delay - elapsed)
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print("\n[INFO] Detection stopped by user")
            if self.DEBUG_OCR:
                cv2.destroyAllWindows()
            sys.exit(0)
        except Exception as e:
            print(f"\n[ERROR] {type(e).__name__}: {e}")
            if self.DEBUG_OCR:
                cv2.destroyAllWindows()
            sys.exit(1)


def main():
    """Entry point."""
    print('=' * 60)
    print("Clash Royale Card Detection System")
    print("Hog 2.6 Mirror Match")
    print("=" * 60)
    
    detector = CardDetector()
    detector.run()


if __name__ == "__main__":
    main()
