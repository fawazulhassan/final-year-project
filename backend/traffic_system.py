import cv2
import numpy as np
import time
from ultralytics import YOLO

# ===========================
#  CONFIGURATION CONSTANTS
# ===========================
MODEL_PATH = "best.pt"  # Place your model in the same directory
VIDEO_PATHS = [
    "test_videos/lane1.mp4",
    "test_videos/lane2.mp4",
    "test_videos/lane3.mp4",
    "test_videos/lane4.mp4"
]

# Timing constants
DEFAULT_GREEN_TIME = 10
EXTRA_TIME_PER_VEHICLE = 0.5
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 60
YELLOW_TIME = 3
ALL_RED_TIME = 2

# Display settings
FRAME_WIDTH = 480
FRAME_HEIGHT = 270

# ===========================
#  GPIO Traffic Light Setup (MOCK)
# ===========================
LIGHTS_CONFIG = {
    'Lane 1': {'red': 26, 'yellow': 27, 'green': 22},
    'Lane 2': {'red': 23, 'yellow': 4, 'green': 25},
    'Lane 3': {'red': 5,  'yellow': 6,  'green': 12},
    'Lane 4': {'red': 13, 'yellow': 16, 'green': 19}
}

# ===========================
#  Vehicle Class Definitions
# ===========================
CLASS_NAMES = {
    0: "car", 1: "bus", 2: "motorbike", 3: "three wheelers -CNG-", 4: "rickshaw",
    5: "truck", 6: "pickup", 7: "minivan", 8: "suv", 9: "van", 10: "bicycle",
    11: "auto rickshaw", 12: "human hauler", 13: "wheelbarrow", 14: "ambulance",
    15: "minibus", 16: "taxi", 17: "army vehicle", 18: "scooter", 19: "policecar",
    20: "garbagevan"
}

# ===========================
#  Mock GPIO Class
# ===========================
class MockGPIO:
    """Mock GPIO for computer testing"""
    BCM = "BCM"
    OUT = "OUT"
    LOW = 0
    HIGH = 1
    
    def __init__(self):
        self.pins = {}
    
    def setmode(self, mode):
        pass
    
    def setup(self, pin, mode):
        self.pins[pin] = 0
    
    def output(self, pin, state):
        self.pins[pin] = state
    
    def cleanup(self):
        self.pins = {}

GPIO = MockGPIO()

# ===========================
#  Traffic Light Controller Class
# ===========================
class TrafficLightController:
    def __init__(self, lights_config):
        self.lights = lights_config
        self.current_states = {lane: 'RED' for lane in lights_config.keys()}
        GPIO.setmode(GPIO.BCM)
        
        # Setup all GPIO pins
        for road in self.lights.values():
            for pin in road.values():
                GPIO.setup(pin, GPIO.OUT)
                GPIO.output(pin, GPIO.LOW)
        
        # Initialize all lights to RED
        self.set_all_red()
    
    def set_light(self, lane_name, color):
        """Turn on a specific color for a given lane."""
        pins = self.lights[lane_name]
        
        # Turn off all lights first
        GPIO.output(pins['red'], GPIO.LOW)
        GPIO.output(pins['yellow'], GPIO.LOW)
        GPIO.output(pins['green'], GPIO.LOW)
        
        # Turn on requested color
        if color == 'RED':
            GPIO.output(pins['red'], GPIO.HIGH)
        elif color == 'YELLOW':
            GPIO.output(pins['yellow'], GPIO.HIGH)
        elif color == 'GREEN':
            GPIO.output(pins['green'], GPIO.HIGH)
        
        # Store current state
        self.current_states[lane_name] = color
        print(f"  💡 GPIO: {lane_name} set to {color}")
    
    def set_all_red(self):
        """Set all lanes to RED for safety."""
        for lane_name in self.lights.keys():
            self.set_light(lane_name, 'RED')
    
    def get_state(self, lane_name):
        """Get current light state for a lane."""
        return self.current_states.get(lane_name, 'RED')
    
    def cleanup(self):
        """Clean up GPIO resources."""
        GPIO.cleanup()

# ===========================
#  Vehicle Detection Class
# ===========================
class VehicleDetector:
    def __init__(self, model_path):
        print("Loading YOLOv8 model...")
        self.model = YOLO(model_path)
        print("✅ Model loaded successfully!\n")
    
    def detect_vehicles(self, frame, lane_name):
        """Detect vehicles in frame and return annotated frame with count."""
        try:
            results = self.model(frame, verbose=False)
            count = 0
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    
                    if cls_id in CLASS_NAMES:
                        count += 1
                        label = CLASS_NAMES[cls_id]
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Draw label with background
                        label_text = f"{label} {confidence:.2f}"
                        (text_width, text_height), _ = cv2.getTextSize(
                            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                    (x1 + text_width, y1), (0, 255, 0), -1)
                        cv2.putText(frame, label_text, (x1, y1 - 5),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # Display lane info and count
            cv2.putText(frame, f"{lane_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Vehicles: {count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            return frame, count
            
        except Exception as e:
            print(f"⚠️ Detection error in {lane_name}: {e}")
            return frame, 0

# ===========================
#  Video Manager Class
# ===========================
class VideoManager:
    def __init__(self, video_paths):
        self.video_paths = video_paths
        self.caps = []
        self.initialize_videos()
    
    def initialize_videos(self):
        """Initialize all video captures."""
        for path in self.video_paths:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"⚠️ Warning: Could not open video {path}")
            self.caps.append(cap)
    
    def read_frame(self, cap_index):
        """Read frame from video capture, loop if ended."""
        cap = self.caps[cap_index]
        ret, frame = cap.read()
        
        # If video ended, restart it
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        # If still can't read, return black frame
        if not ret:
            frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
            cv2.putText(frame, f"Lane {cap_index + 1}: Video Error", 
                       (50, FRAME_HEIGHT // 2), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 0, 255), 2)
        else:
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        return frame
    
    def release_all(self):
        """Release all video captures."""
        for cap in self.caps:
            cap.release()

# ===========================
#  Smart Traffic System Class
# ===========================
class SmartTrafficSystem:
    def __init__(self):
        self.detector = VehicleDetector(MODEL_PATH)
        self.video_manager = VideoManager(VIDEO_PATHS)
        self.light_controller = TrafficLightController(LIGHTS_CONFIG)
        
        self.lane_status = {
            "Lane 1": {"count": 0, "time": DEFAULT_GREEN_TIME},
            "Lane 2": {"count": 0, "time": DEFAULT_GREEN_TIME},
            "Lane 3": {"count": 0, "time": DEFAULT_GREEN_TIME},
            "Lane 4": {"count": 0, "time": DEFAULT_GREEN_TIME},
        }
    
    def scan_all_lanes(self):
        """Scan all lanes and detect vehicles."""
        frames = []
        
        for i in range(4):
            frame = self.video_manager.read_frame(i)
            lane_name = f"Lane {i + 1}"
            
            # Detect vehicles
            annotated_frame, count = self.detector.detect_vehicles(frame, lane_name)
            
            # Update lane status
            self.lane_status[lane_name]["count"] = count
            green_time = DEFAULT_GREEN_TIME + (count * EXTRA_TIME_PER_VEHICLE)
            green_time = max(MIN_GREEN_TIME, min(green_time, MAX_GREEN_TIME))
            self.lane_status[lane_name]["time"] = green_time
            
            frames.append(annotated_frame)
        
        return frames
    
    def add_traffic_light_indicator(self, frame, lane_name):
        """Add visual traffic light indicator to frame."""
        light_state = self.light_controller.get_state(lane_name)
        
        # Draw traffic light circle
        center_x = FRAME_WIDTH - 40
        center_y = 40
        radius = 20
        
        # Background circle
        cv2.circle(frame, (center_x, center_y), radius + 5, (50, 50, 50), -1)
        
        # Light color based on state
        if light_state == 'RED':
            color = (0, 0, 255)
        elif light_state == 'YELLOW':
            color = (0, 255, 255)
        elif light_state == 'GREEN':
            color = (0, 255, 0)
        else:
            color = (100, 100, 100)
        
        cv2.circle(frame, (center_x, center_y), radius, color, -1)
        
        return frame
    
    def display_grid(self, frames):
        """Display all 4 lanes in a 2x2 grid with traffic light indicators."""
        # Add traffic light indicators to each frame
        for i in range(4):
            lane_name = f"Lane {i + 1}"
            frames[i] = self.add_traffic_light_indicator(frames[i], lane_name)
        
        top_row = np.hstack((frames[0], frames[1]))
        bottom_row = np.hstack((frames[2], frames[3]))
        grid_frame = np.vstack((top_row, bottom_row))
        
        # Add system info
        info_text = "SMART TRAFFIC SYSTEM - Density Based Control"
        cv2.putText(grid_frame, info_text, (10, grid_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Traffic Detection Grid", grid_frame)
    
    def get_priority_order(self):
        """Sort lanes by vehicle count (highest to lowest)."""
        return sorted(
            self.lane_status.keys(),
            key=lambda x: self.lane_status[x]["count"],
            reverse=True
        )
    
    def execute_light_cycle(self):
        """Execute one complete traffic light cycle based on priority."""
        priority_lanes = self.get_priority_order()
        
        print("\n" + "="*60)
        print("🚦 NEW TRAFFIC CYCLE STARTING")
        print("="*60)
        for i, lane in enumerate(priority_lanes, 1):
            count = self.lane_status[lane]["count"]
            print(f"{i}. {lane}: {count} vehicles")
        print("="*60 + "\n")
        
        # Process each lane according to priority
        for lane in priority_lanes:
            green_time = self.lane_status[lane]["time"]
            vehicle_count = self.lane_status[lane]["count"]
            
            print(f"\n🟢 {lane} → GREEN for {green_time:.1f}s ({vehicle_count} vehicles)")
            
            # Set current lane to GREEN, others stay RED
            self.light_controller.set_light(lane, 'GREEN')
            
            # Wait for green time with continuous frame updates
            start_time = time.time()
            while time.time() - start_time < green_time:
                frames = self.scan_all_lanes()
                self.display_grid(frames)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
            
            # YELLOW phase
            print(f"🟡 {lane} → YELLOW for {YELLOW_TIME}s")
            self.light_controller.set_light(lane, 'YELLOW')
            
            yellow_start = time.time()
            while time.time() - yellow_start < YELLOW_TIME:
                frames = self.scan_all_lanes()
                self.display_grid(frames)
                
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    raise KeyboardInterrupt
            
            # RED phase
            print(f"🔴 {lane} → RED")
            self.light_controller.set_light(lane, 'RED')
            
            # All-red safety phase (except for last lane)
            if lane != priority_lanes[-1]:
                print(f"⏸️  All RED safety phase for {ALL_RED_TIME}s")
                time.sleep(ALL_RED_TIME)
    
    def run(self):
        """Main system loop."""
        print("\n" + "="*60)
        print("🚦 SMART TRAFFIC LIGHT SYSTEM - DENSITY BASED")
        print("="*60)
        print("System Logic:")
        print("1. Scan all lanes for vehicles")
        print("2. Prioritize lanes: Most traffic → Least traffic")
        print("3. Allocate green time based on vehicle count")
        print("4. Repeat cycle continuously")
        print("\nPress 'q' to quit")
        print("="*60 + "\n")
        
        try:
            while True:
                # Scan all lanes first
                print("📊 Scanning all lanes...")
                frames = self.scan_all_lanes()
                self.display_grid(frames)
                
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                
                # Execute traffic light cycle based on density
                self.execute_light_cycle()
                
        except KeyboardInterrupt:
            print("\n\n⚠️ System interrupted by user")
        except Exception as e:
            print(f"\n\n❌ System error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all resources."""
        print("\n🧹 Cleaning up resources...")
        self.video_manager.release_all()
        self.light_controller.cleanup()
        cv2.destroyAllWindows()
        print("✅ System stopped. All resources released.\n")

# ===========================
#  MAIN ENTRY POINT
# ===========================
if __name__ == "__main__":
    system = SmartTrafficSystem()
    system.run()