from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# ===========================
#  CONFIGURATION
# ===========================
MODEL_PATH = "best.pt"
VIDEO_PATHS = [
    "test_videos/lane1.mp4",
    "test_videos/lane2.mp4",
    "test_videos/lane3.mp4",
    "test_videos/lane4.mp4"
]

DEFAULT_GREEN_TIME = 10
EXTRA_TIME_PER_VEHICLE = 0.5
MIN_GREEN_TIME = 5
MAX_GREEN_TIME = 60
YELLOW_TIME = 3
ALL_RED_TIME = 2

CLASS_NAMES = {
    0: "car", 1: "bus", 2: "motorbike", 3: "three wheelers -CNG-", 4: "rickshaw",
    5: "truck", 6: "pickup", 7: "minivan", 8: "suv", 9: "van", 10: "bicycle",
    11: "auto rickshaw", 12: "human hauler", 13: "wheelbarrow", 14: "ambulance",
    15: "minibus", 16: "taxi", 17: "army vehicle", 18: "scooter", 19: "policecar",
    20: "garbagevan"
}

# ===========================
#  GLOBAL VARIABLES
# ===========================
model = None
video_captures = []
lane_data = {
    "lane1": {"vehicles": 0, "status": "RED", "priority": 1, "waitTime": 0, "lastRedTime": time.time()},
    "lane2": {"vehicles": 0, "status": "RED", "priority": 2, "waitTime": 0, "lastRedTime": time.time()},
    "lane3": {"vehicles": 0, "status": "RED", "priority": 3, "waitTime": 0, "lastRedTime": time.time()},
    "lane4": {"vehicles": 0, "status": "RED", "priority": 4, "waitTime": 0, "lastRedTime": time.time()},
}
system_running = False
current_logs = []

# ===========================
#  INITIALIZE MODEL & VIDEOS
# ===========================
def initialize_system():
    global model, video_captures
    
    print("Loading YOLOv8 model...")
    model = YOLO(MODEL_PATH)
    print("✅ Model loaded successfully!")
    
    for path in VIDEO_PATHS:
        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            video_captures.append(cap)
            print(f"✅ Loaded video: {path}")
        else:
            print(f"⚠️ Failed to load: {path}")
            video_captures.append(None)

# ===========================
#  VEHICLE DETECTION
# ===========================
def detect_vehicles(frame):
    """Detect vehicles in frame and return count."""
    if model is None:
        return 0
    
    try:
        results = model(frame, verbose=False)
        count = 0
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id in CLASS_NAMES:
                    count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    label = CLASS_NAMES[cls_id]
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    label_text = f"{label} {confidence:.2f}"
                    (text_width, text_height), _ = cv2.getTextSize(
                        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                                (x1 + text_width, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label_text, (x1, y1 - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return count
    except Exception as e:
        print(f"Detection error: {e}")
        return 0

# ===========================
#  WAIT TIME CALCULATION
# ===========================
def update_wait_times():
    """Update wait times for all lanes that are RED"""
    current_time = time.time()
    for lane_key, data in lane_data.items():
        if data["status"] == "RED":
            data["waitTime"] = int(current_time - data["lastRedTime"])
        else:
            data["waitTime"] = 0

# ===========================
#  TRAFFIC LIGHT LOGIC
# ===========================
def update_traffic_cycle():
    """Main traffic light cycle logic"""
    global lane_data, system_running
    
    while system_running:
        update_wait_times()
        
        # Scan all lanes and detect vehicles
        for i, cap in enumerate(video_captures):
            if cap is None:
                continue
            
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            if ret:
                vehicle_count = detect_vehicles(frame)
                lane_key = f"lane{i+1}"
                lane_data[lane_key]["vehicles"] = vehicle_count
                
                green_time = DEFAULT_GREEN_TIME + (vehicle_count * EXTRA_TIME_PER_VEHICLE)
                green_time = max(MIN_GREEN_TIME, min(green_time, MAX_GREEN_TIME))
                lane_data[lane_key]["greenTime"] = green_time
        
        # Sort lanes by priority (most vehicles first)
        sorted_lanes = sorted(
            lane_data.items(),
            key=lambda x: x[1]["vehicles"],
            reverse=True
        )
        
        # Update priority rankings
        for idx, (lane_key, data) in enumerate(sorted_lanes):
            lane_data[lane_key]["priority"] = idx + 1
        
        # Execute traffic cycle
        for lane_key, data in sorted_lanes:
            if not system_running:
                break
            
            # GREEN phase
            lane_data[lane_key]["status"] = "GREEN"
            lane_data[lane_key]["waitTime"] = 0
            add_log(f"{lane_key.upper()} → GREEN ({data['vehicles']} vehicles)")
            time.sleep(data.get("greenTime", DEFAULT_GREEN_TIME))
            
            # YELLOW phase
            lane_data[lane_key]["status"] = "YELLOW"
            add_log(f"{lane_key.upper()} → YELLOW")
            time.sleep(YELLOW_TIME)
            
            # RED phase
            lane_data[lane_key]["status"] = "RED"
            lane_data[lane_key]["lastRedTime"] = time.time()
            add_log(f"{lane_key.upper()} → RED")
            time.sleep(ALL_RED_TIME)

def add_log(message):
    """Add log message with timestamp"""
    timestamp = time.strftime("%H:%M:%S")
    current_logs.append(f"[{timestamp}] {message}")
    if len(current_logs) > 20:
        current_logs.pop(0)

# ===========================
#  VIDEO STREAMING
# ===========================
def generate_frames(lane_id):
    """Generator function to stream video frames"""
    cap = video_captures[lane_id - 1]
    
    while True:
        if cap is None:
            break
            
        ret, frame = cap.read()
        
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.resize(frame, (640, 360))
        vehicle_count = detect_vehicles(frame)
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.033)

# ===========================
#  API ENDPOINTS
# ===========================
@app.route('/api/status', methods=['GET'])
def get_status():
    """Get overall system status"""
    update_wait_times()
    return jsonify({
        'running': system_running,
        'totalVehicles': sum(data['vehicles'] for data in lane_data.values()),
        'avgWaitTime': sum(data['waitTime'] for data in lane_data.values()) // 4 if lane_data else 0
    })

@app.route('/api/lanes', methods=['GET'])
def get_lanes():
    """Get all lane data"""
    update_wait_times()
    lanes_list = []
    for i, (lane_key, data) in enumerate(lane_data.items()):
        lanes_list.append({
            'id': i + 1,
            'name': f'Lane {i + 1}',
            'vehicles': data['vehicles'],
            'status': data['status'],
            'priority': data['priority'],
            'waitTime': data['waitTime']
        })
    
    return jsonify({'lanes': lanes_list})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get system activity logs"""
    return jsonify({'logs': current_logs})

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start the traffic management system"""
    global system_running
    
    if not system_running:
        system_running = True
        current_time = time.time()
        for lane_key in lane_data:
            lane_data[lane_key]["status"] = "RED"
            lane_data[lane_key]["lastRedTime"] = current_time
            lane_data[lane_key]["waitTime"] = 0
        
        add_log("System STARTED")
        thread = threading.Thread(target=update_traffic_cycle, daemon=True)
        thread.start()
        return jsonify({'status': 'started', 'message': 'System is now running'})
    
    return jsonify({'status': 'already_running', 'message': 'System is already active'})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop the traffic management system"""
    global system_running
    
    if system_running:
        system_running = False
        add_log("System STOPPED")
        
        for lane_key in lane_data:
            lane_data[lane_key]["status"] = "RED"
            lane_data[lane_key]["waitTime"] = 0
        
        return jsonify({'status': 'stopped', 'message': 'System stopped successfully'})
    
    return jsonify({'status': 'not_running', 'message': 'System is not active'})

@app.route('/api/video/stream/<int:lane_id>')
def video_stream(lane_id):
    """Stream video for specific lane"""
    if lane_id < 1 or lane_id > 4:
        return jsonify({'error': 'Invalid lane ID'}), 400
    
    return Response(generate_frames(lane_id),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# ===========================
#  MAIN
# ===========================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚦 SMART TRAFFIC MANAGEMENT SYSTEM - API SERVER")
    print("="*60)
    
    initialize_system()
    add_log("System initialized successfully")
    
    print("\n✅ Server starting on http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)