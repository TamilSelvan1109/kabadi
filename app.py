from flask import Flask, render_template, request, jsonify, Response, session
import cv2
import json
import numpy as np
import os
import threading
import time
from werkzeug.utils import secure_filename
import base64
import mediapipe as mp

# Import our existing modules
from foot_detector import SkeletonTracker
from violation_detector import ViolationDetector
from player_tracker import PlayerTracker

app = Flask(__name__)
app.secret_key = 'kabaddi_tracking_secret_key_2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2GB limit

# Global variables
current_video_path = None
boundary_points = []
detection_method = "TWO_POINTS"
tracking_active = False
violated_players = set()  # Track players who violated
violation_log = []

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)

class LiveKabaddiTracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.skeleton_tracker = SkeletonTracker()
        self.player_tracker = PlayerTracker()
        self.violation_detector = ViolationDetector(self.player_tracker)
        self.pose = self._init_mediapipe()
        self.current_poses = []
        self.frame_count = 0
        self.violated_players = set()
        self.violation_log = []
        
    def _init_mediapipe(self):
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
                running_mode=RunningMode.VIDEO,
                num_poses=10,  # Increased from 8
                min_pose_detection_confidence=0.2,  # Lowered for better detection
                min_pose_presence_confidence=0.2,   # Lowered for better detection
                min_tracking_confidence=0.2         # Lowered for better tracking
            )
            
            return PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            return None
    
    def process_frame(self, frame):
        global violated_players, violation_log
        
        self.frame_count += 1
        original_h, original_w = frame.shape[:2]
        
        # Don't crop video - keep full frame for tracking
        # Only resize if too large for display
        if original_w > 1920:  # Only resize if very large
            scale = 1920 / original_w
            new_w = 1920
            new_h = int(original_h * scale)
            display_frame = cv2.resize(frame, (new_w, new_h))
        else:
            display_frame = frame.copy()
            
        h, w = display_frame.shape[:2]
        
        # Process with MediaPipe
        if self.pose:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            results = self.pose.detect_for_video(mp_image, int(time.time() * 1000))
            self.current_poses = results.pose_landmarks if results.pose_landmarks else []
        
        # Draw boundary line
        if boundary_points:
            for i in range(len(boundary_points) - 1):
                pt1 = tuple(map(int, boundary_points[i]))
                pt2 = tuple(map(int, boundary_points[i + 1]))
                cv2.line(display_frame, pt1, pt2, (0, 255, 255), 4)  # Thicker line
        
        # Process each detected person
        active_players = 0
        for i, pose_landmarks in enumerate(self.current_poses):
            if pose_landmarks:
                player_id = f"P{i+1}"
                
                # Skip if player already violated
                if player_id in violated_players:
                    continue
                
                active_players += 1
                
                # Get ground contact points
                ground_points = self.skeleton_tracker.get_ground_contact_points(pose_landmarks, w, h)
                
                # Check for violations
                violation_detected = False
                if boundary_points:
                    violation_detected = any(
                        self.violation_detector.check_boundary_crossing(p['position'][0], p['position'][1]) 
                        for p in ground_points[:2]
                    )
                
                # Handle violation
                if violation_detected:
                    violated_players.add(player_id)
                    violation_entry = {
                        'player_id': player_id,
                        'frame': self.frame_count,
                        'timestamp': time.strftime('%H:%M:%S')
                    }
                    violation_log.append(violation_entry)
                    print(f"🚨 VIOLATION: {player_id} at frame {self.frame_count}")
                
                # Draw skeleton (red if violated, green if active)
                is_violated = player_id in violated_players
                self.skeleton_tracker.draw_skeleton(
                    display_frame, pose_landmarks, player_id, is_violated, is_violated
                )
        
        # Enhanced status overlay with better visibility
        overlay_height = 150
        cv2.rectangle(display_frame, (10, 10), (500, overlay_height), (0, 0, 0), -1)
        cv2.rectangle(display_frame, (10, 10), (500, overlay_height), (255, 255, 255), 2)
        
        cv2.putText(display_frame, f"Frame: {self.frame_count}", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Active Players: {active_players}", (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Violated Players: {len(violated_players)}", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Total Violations: {len(violation_log)}", (20, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
        
        return display_frame

def detect_hough_lines(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Enhanced edge detection
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        # Improved Hough line detection with lower threshold
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        return lines[:10] if lines is not None else []
    except Exception as e:
        print(f"Hough line detection error: {e}")
        return []

@app.route('/')
def index():
    return render_template('live_dashboard.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global current_video_path
    
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video file'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file:
        filename = secure_filename(file.filename)
        current_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(current_video_path)
        
        print(f"Debug: Video uploaded to: {current_video_path}")
        print(f"Debug: File exists: {os.path.exists(current_video_path)}")
        
        return jsonify({'success': True, 'message': 'Video uploaded successfully'})

@app.route('/get_video_frame')
def get_video_frame():
    global current_video_path
    
    # Use current_video_path directly
    video_path = current_video_path
    
    print(f"Debug: Looking for video at: {video_path}")
    
    if not video_path or not os.path.exists(video_path):
        print(f"Debug: Video not found. Path: {video_path}, Exists: {os.path.exists(video_path) if video_path else False}")
        return jsonify({'success': False, 'error': 'No video available'})
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        height, width = frame.shape[:2]
        # Don't resize too small - keep larger for better visibility
        if width > 1200:
            scale = 1200 / width
            new_width = 1200
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"Debug: Frame loaded successfully. Size: {frame.shape}")
        
        return jsonify({
            'success': True, 
            'frame': frame_base64,
            'width': frame.shape[1],
            'height': frame.shape[0]
        })
    
    print("Debug: Could not read video frame")
    return jsonify({'success': False, 'error': 'Could not read video frame'})

@app.route('/detect_lines', methods=['POST'])
def detect_lines():
    global current_video_path, detection_method
    
    data = request.json
    detection_method = data.get('method', 'TWO_POINTS')
    
    if not current_video_path:
        return jsonify({'success': False, 'error': 'No video available'})
    
    cap = cv2.VideoCapture(current_video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return jsonify({'success': False, 'error': 'Could not read video'})
    
    if detection_method == "HOUGH":
        lines = detect_hough_lines(frame)
        line_data = []
        
        h, w = frame.shape[:2]
        for i, line in enumerate(lines):
            rho, theta = line[0]
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a * rho, b * rho
            
            if abs(b) > 0.001:
                x1, y1 = 0, int(y0 - (x0 * a / b))
                x2, y2 = w, int(y0 + ((w - x0) * a / b))
            else:
                x1, y1 = int(x0), 0
                x2, y2 = int(x0), h
            
            x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
            x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
            
            line_data.append({
                'id': i,
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'center_x': int((x1 + x2) / 2),
                'center_y': int((y1 + y2) / 2)
            })
        
        return jsonify({'success': True, 'lines': line_data})
    
    return jsonify({'success': True, 'method': detection_method})

@app.route('/save_boundary', methods=['POST'])
def save_boundary():
    global boundary_points, detection_method
    
    data = request.json
    points = data.get('points', [])
    method = data.get('method', detection_method)
    
    if not points:
        return jsonify({'success': False, 'error': 'No points provided'})
    
    # Scale points to original video size
    if current_video_path:
        cap = cv2.VideoCapture(current_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            orig_h, orig_w = frame.shape[:2]
            display_w = data.get('display_width', 1200)
            display_h = data.get('display_height', int(orig_h * (1200/orig_w)))
            
            if orig_w > 1200:
                scale_x = orig_w / display_w
                scale_y = orig_h / display_h
                points = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in points]
    
    boundary_points = points
    
    config_data = {"boundary_points": points, "method": method}
    with open("config.json", "w") as f:
        json.dump(config_data, f)
    
    return jsonify({'success': True, 'message': 'Boundary saved successfully'})

@app.route('/preview_tracking', methods=['POST'])
def preview_tracking():
    """Show a preview frame with player tracking before starting live tracking"""
    global current_video_path, boundary_points
    
    if not current_video_path or not boundary_points:
        return jsonify({'success': False, 'error': 'Video or boundary not available'})
    
    try:
        tracker = LiveKabaddiTracker(current_video_path)
        cap = cv2.VideoCapture(current_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Process one frame to show preview
            processed_frame = tracker.process_frame(frame)
            
            # Resize for web display
            height, width = processed_frame.shape[:2]
            if width > 1200:
                scale = 1200 / width
                new_width = 1200
                new_height = int(height * scale)
                processed_frame = cv2.resize(processed_frame, (new_width, new_height))
            
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            return jsonify({
                'success': True,
                'preview_frame': frame_base64,
                'players_detected': len(tracker.current_poses)
            })
    except Exception as e:
        return jsonify({'success': False, 'error': f'Preview failed: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Could not generate preview'})
    data = request.json
    points = data.get('points', [])
    method = data.get('method', detection_method)
    
    if not points:
        return jsonify({'success': False, 'error': 'No points provided'})
    
    # Scale points to original video size
    if current_video_path:
        cap = cv2.VideoCapture(current_video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            orig_h, orig_w = frame.shape[:2]
            # Get the actual display size from the web interface
            display_w = data.get('display_width', 1200)
            display_h = data.get('display_height', int(orig_h * (1200/orig_w)))
            
            # Calculate scaling factor
            if orig_w > 1200:
                scale_x = orig_w / display_w
                scale_y = orig_h / display_h
                points = [[int(p[0] * scale_x), int(p[1] * scale_y)] for p in points]
            # If original is smaller than display, no scaling needed
    
    boundary_points = points
    
    # Save to config.json
    config_data = {"boundary_points": points, "method": method}
    with open("config.json", "w") as f:
        json.dump(config_data, f)
    
    return jsonify({'success': True, 'message': 'Boundary saved successfully'})

@app.route('/start_live_tracking', methods=['POST'])
def start_live_tracking():
    global tracking_active, violated_players, violation_log
    
    if not current_video_path:
        return jsonify({'success': False, 'error': 'No video available'})
    
    if not boundary_points:
        return jsonify({'success': False, 'error': 'No boundary defined'})
    
    # Reset tracking state
    violated_players.clear()
    violation_log.clear()
    tracking_active = True
    
    return jsonify({'success': True, 'message': 'Live tracking started'})

@app.route('/stop_tracking', methods=['POST'])
def stop_tracking():
    global tracking_active
    tracking_active = False
    return jsonify({'success': True, 'message': 'Tracking stopped'})

@app.route('/get_violations')
def get_violations():
    return jsonify({
        'violated_players': list(violated_players),
        'violation_log': violation_log,
        'total_violations': len(violation_log)
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        global tracking_active, current_video_path
        
        if not current_video_path or not tracking_active:
            return
        
        tracker = LiveKabaddiTracker(current_video_path)
        cap = cv2.VideoCapture(current_video_path)
        
        while tracking_active and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            # Process frame with player tracking
            processed_frame = tracker.process_frame(frame)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.033)  # ~30 FPS
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🌐 Starting Live Kabaddi Tracking Dashboard...")
    print("📱 Open http://localhost:5000 in your browser")
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)