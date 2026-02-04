import cv2
import json
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import math
import os
from datetime import datetime
from video_config import get_player_tracking_video, get_frame_config
from modules.skeleton_tracker import SkeletonTracker
from modules.kalman_tracker import KalmanTracker

class PlayerTracker:
    def __init__(self):
        # Load YOLO model
        self.yolo_model = YOLO('yolov8n.pt')
        
        # Initialize MediaPipe skeleton tracker
        self.skeleton_tracker = SkeletonTracker()
        
        # Load boundary configuration
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
                self.original_boundary_points = config['boundary_points']
                self.boundary_points = []
                print(f"SUCCESS: Loaded {len(self.original_boundary_points)} boundary points")
        except Exception as e:
            print(f"ERROR: Loading boundary points: {e}")
            self.original_boundary_points = []
            self.boundary_points = []
        
        # Player tracking with Kalman filters
        self.stable_players = {}
        self.kalman_filters = {}  # Store Kalman filter for each player
        self.next_stable_id = 1
        self.max_distance = 150
        self.max_frames_missing = 60
        
        # Violation tracking
        self.violation_status = {}
        self.frame_count = 0
        self.violation_records = {}
        self.active_violations = set()
        self.violation_start_frames = {}
        
        # Create output directories
        os.makedirs('violations/screenshots', exist_ok=True)
        os.makedirs('violations/videos', exist_ok=True)
        
        # Circular buffer for last 3 seconds (90 frames at 30fps)
        self.circular_buffer = []
        self.buffer_size = 90
        
    def scale_boundary_points(self, scale_factor):
        """Scale boundary points from original resolution to current display resolution"""
        self.boundary_points = []
        for point in self.original_boundary_points:
            scaled_x = int(point[0] * scale_factor)
            scaled_y = int(point[1] * scale_factor)
            self.boundary_points.append([scaled_x, scaled_y])
        print(f"📏 Scaled boundary points by {scale_factor:.3f}: {self.boundary_points}")
    
    def is_point_below_boundary(self, point):
        """Improved boundary violation detection with better interpolation"""
        if len(self.boundary_points) < 2:
            return False
            
        x, y = point
        
        # Find the boundary y-coordinate at the given x using linear interpolation
        boundary_y = None
        
        # Check if point is within boundary x-range
        min_x = min(p[0] for p in self.boundary_points)
        max_x = max(p[0] for p in self.boundary_points)
        
        if x < min_x or x > max_x:
            # Point is outside boundary x-range - use nearest boundary point
            if x < min_x:
                boundary_y = next(p[1] for p in self.boundary_points if p[0] == min_x)
            else:
                boundary_y = next(p[1] for p in self.boundary_points if p[0] == max_x)
        else:
            # Find the two boundary points that bracket this x coordinate
            for i in range(len(self.boundary_points) - 1):
                x1, y1 = self.boundary_points[i]
                x2, y2 = self.boundary_points[i + 1]
                
                if x1 <= x <= x2:
                    # Linear interpolation between the two points
                    if x2 != x1:
                        boundary_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                    else:
                        boundary_y = y1
                    break
        
        if boundary_y is None:
            return False
        
        # Check if foot is below (greater y value) the boundary line
        violation = y > boundary_y
        
        # Debug boundary check occasionally
        if self.frame_count % 120 == 0:  # Every 4 seconds
            print(f"🔍 Boundary check: foot=({x},{y}), boundary_y={boundary_y:.1f}, violation={violation}")
        
        return violation
    
    def get_stable_id(self, center_pos, bbox, yolo_id):
        """Improved stable ID assignment with Kalman filter prediction"""
        x, y = center_pos
        x1, y1, x2, y2 = bbox
        
        # Predict positions for all existing players using Kalman filters
        predicted_positions = {}
        for stable_id in self.kalman_filters:
            predicted_positions[stable_id] = self.kalman_filters[stable_id].predict()
        
        # First, check if we already have this YOLO ID mapped to a stable ID
        for stable_id, player_data in self.stable_players.items():
            if player_data.get('yolo_id') == yolo_id:
                # Update Kalman filter with new measurement
                self.kalman_filters[stable_id].update(center_pos)
                
                # Update existing player data
                self.stable_players[stable_id].update({
                    'position': center_pos,
                    'last_seen': self.frame_count,
                    'bbox': bbox
                })
                return stable_id
        
        # If YOLO ID not found, try to match by predicted position (Kalman-based)
        closest_id = None
        min_distance = float('inf')
        
        for stable_id in predicted_positions:
            px, py = predicted_positions[stable_id]
            distance = math.sqrt((x - px)**2 + (y - py)**2)
            
            # Check if this is likely the same player using Kalman prediction
            if distance < self.max_distance and distance < min_distance:
                min_distance = distance
                closest_id = stable_id
        
        if closest_id is not None:
            # Update Kalman filter with new measurement
            self.kalman_filters[closest_id].update(center_pos)
            
            # Update existing player with new YOLO ID
            self.stable_players[closest_id].update({
                'position': center_pos,
                'last_seen': self.frame_count,
                'yolo_id': yolo_id,
                'bbox': bbox
            })
            return closest_id
        
        # Check for bounding box overlap with existing players (prevent duplicates)
        for stable_id, player_data in self.stable_players.items():
            px1, py1, px2, py2 = player_data['bbox']
            
            # Calculate overlap
            overlap_x = max(0, min(x2, px2) - max(x1, px1))
            overlap_y = max(0, min(y2, py2) - max(y1, py1))
            overlap_area = overlap_x * overlap_y
            
            # Calculate areas
            current_area = (x2 - x1) * (y2 - y1)
            existing_area = (px2 - px1) * (py2 - py1)
            
            # If significant overlap, consider it the same player
            if overlap_area > 0.3 * min(current_area, existing_area):
                # Update Kalman filter with new measurement
                self.kalman_filters[stable_id].update(center_pos)
                
                # Update with new YOLO ID
                self.stable_players[stable_id].update({
                    'position': center_pos,
                    'last_seen': self.frame_count,
                    'yolo_id': yolo_id,
                    'bbox': bbox
                })
                return stable_id
        
        # Create new player with Kalman filter
        new_stable_id = self.next_stable_id
        self.stable_players[new_stable_id] = {
            'position': center_pos,
            'last_seen': self.frame_count,
            'yolo_id': yolo_id,
            'bbox': bbox
        }
        
        # Initialize Kalman filter for new player
        self.kalman_filters[new_stable_id] = KalmanTracker(center_pos)
        
        self.next_stable_id += 1
        
        print(f"🆕 New player created: Stable ID {new_stable_id} (YOLO ID: {yolo_id}) with Kalman filter")
        return new_stable_id
    
    def cleanup_old_players(self):
        """Remove players not seen for too long and save any ongoing violation videos"""
        to_remove = []
        for stable_id, player_data in self.stable_players.items():
            if self.frame_count - player_data['last_seen'] > self.max_frames_missing:
                to_remove.append(stable_id)
        
        for stable_id in to_remove:
            # Save any ongoing violation video before removing player
            if stable_id in self.violation_records:
                print(f"⚠️ Player {stable_id} disappeared during violation - saving video")
                self.save_violation_video(stable_id)
            
            # Remove from active violations
            self.active_violations.discard(stable_id)
            
            # Remove player and Kalman filter
            del self.stable_players[stable_id]
            if stable_id in self.kalman_filters:
                del self.kalman_filters[stable_id]
        
    def get_foot_position_with_skeleton(self, frame, bbox, player_id):
        """Use the working skeleton tracker for foot position detection"""
        return self.skeleton_tracker.get_foot_position(frame, bbox, player_id)

    
    def process_frame(self, frame):
        """Process frame with improved YOLO detection and stable ID tracking"""
        self.frame_count += 1
        
        # Add frame to circular buffer
        self.circular_buffer.append(frame.copy())
        if len(self.circular_buffer) > self.buffer_size:
            self.circular_buffer.pop(0)
        
        # YOLO detection with tracking - detect persons only
        results = self.yolo_model.track(
            frame, 
            persist=True, 
            classes=[0],  # Only detect persons
            conf=0.5,     # Confidence threshold
            iou=0.7,      # IoU threshold for NMS
            tracker="bytetrack.yaml"  # Use ByteTrack for better tracking
        )
        
        current_violations = set()
        current_detections = []
        
        # Process YOLO detections
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            # Check if tracking IDs are available
            if boxes.id is not None:
                # Get detection data
                xyxy_boxes = boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
                track_ids = boxes.id.int().cpu().tolist()  # YOLO tracking IDs
                confidences = boxes.conf.float().cpu().tolist()  # Confidence scores
                
                # Debug YOLO detections
                if self.frame_count % 60 == 0:
                    print(f"🎯 Frame {self.frame_count}: YOLO detected {len(track_ids)} players")
                
                # Process each detection
                for i, (bbox, yolo_id, conf) in enumerate(zip(xyxy_boxes, track_ids, confidences)):
                    if conf < 0.5:  # Skip low confidence detections
                        continue
                    
                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Ensure bbox is within frame boundaries
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    # Skip invalid bounding boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    # Calculate center position
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    center_pos = (center_x, center_y)
                    bbox_tuple = (x1, y1, x2, y2)
                    
                    # Get stable ID for this detection
                    stable_id = self.get_stable_id(center_pos, bbox_tuple, yolo_id)
                    current_detections.append(stable_id)
                    
                    # INDIVIDUAL PLAYER SKELETON TRACKING
                    foot_pos, skeleton_drawn = self.get_foot_position_with_skeleton(frame, bbox_tuple, stable_id)
                    
                    # Debug skeleton detection
                    if self.frame_count % 30 == 0:
                        status = "✅ SKELETON DETECTED" if skeleton_drawn else "❌ NO SKELETON"
                        print(f"Player {stable_id} (YOLO ID: {yolo_id}) at frame {self.frame_count}: {status}")
                    
                    # Check boundary violation
                    is_violation = self.is_point_below_boundary(foot_pos)
                    self.violation_status[stable_id] = is_violation
                    
                    if is_violation:
                        current_violations.add(stable_id)
                    
                    # Draw bounding box with appropriate color
                    color = (0, 0, 255) if is_violation else (0, 255, 0)  # Red for violation, Green for normal
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                    
                    # Create comprehensive label
                    label_parts = [f"Player {stable_id}"]
                    if is_violation:
                        label_parts.append("VIOLATION!")
                    if skeleton_drawn:
                        label_parts.append("[SKELETON ON]")
                    else:
                        label_parts.append("[SKELETON OFF]")
                    label_parts.append(f"(Y:{yolo_id})")
                    
                    label = " ".join(label_parts)
                    
                    # Draw label background
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(frame, (x1, y1-30), (x1 + text_size[0] + 10, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 5, y1-8), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Draw foot position marker
                    foot_color = (0, 255, 255) if skeleton_drawn else (255, 0, 255)  # Yellow for MediaPipe, Magenta for fallback
                    cv2.circle(frame, foot_pos, 6, foot_color, -1)
                    
                    # Draw foot label
                    foot_label = "MP_FOOT" if skeleton_drawn else "BBOX_FOOT"
                    cv2.putText(frame, foot_label, (foot_pos[0]-25, foot_pos[1]-15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, foot_color, 1)
            
            else:
                # No tracking IDs available - YOLO tracking failed
                if self.frame_count % 60 == 0:
                    print(f"⚠️ Frame {self.frame_count}: YOLO tracking IDs not available")
        
        else:
            # No detections
            if self.frame_count % 120 == 0:
                print(f"👻 Frame {self.frame_count}: No players detected")
        
        # Handle violation recording
        self.handle_violations(frame, current_violations)
        
        # Cleanup old players
        self.cleanup_old_players()
        
        return frame, current_violations
    
    def handle_violations(self, frame, current_violations):
        """Improved violation handling - one screenshot per violation, continuous video recording"""
        
        # Check for new violations (players who just started violating)
        new_violations = current_violations - self.active_violations
        
        # Check for ended violations (players who stopped violating)
        ended_violations = self.active_violations - current_violations
        
        # Handle new violations
        for player_id in new_violations:
            print(f"🚨 NEW VIOLATION: Player {player_id} at frame {self.frame_count}")
            
            # Take screenshot immediately (only once per violation)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"violations/screenshots/player_{player_id}_violation_{self.frame_count}_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved: {screenshot_path}")
            
            # Start video recording with pre-violation footage
            self.violation_records[player_id] = {
                'screenshot_taken': True,
                'frames': self.circular_buffer.copy() + [frame.copy()],  # Include 3-sec history
                'start_frame': self.frame_count
            }
            self.violation_start_frames[player_id] = self.frame_count
        
        # Continue recording for ongoing violations
        for player_id in current_violations:
            if player_id in self.violation_records:
                self.violation_records[player_id]['frames'].append(frame.copy())
                # Limit frames to prevent memory issues (max 5 seconds at 30fps = 150 frames)
                if len(self.violation_records[player_id]['frames']) > 150:
                    self.violation_records[player_id]['frames'].pop(0)
        
        # Handle ended violations - save video
        for player_id in ended_violations:
            self.save_violation_video(player_id)
        
        # Update active violations
        self.active_violations = current_violations.copy()
    
    def save_violation_video(self, player_id):
        """Save violation video when violation ends"""
        if player_id not in self.violation_records:
            return
            
        frames = self.violation_records[player_id]['frames']
        start_frame = self.violation_records[player_id]['start_frame']
        
        if len(frames) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"violations/videos/player_{player_id}_violation_{start_frame}_{timestamp}.mp4"
            
            # Get frame dimensions
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            
            # Write all frames
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            duration = len(frames) / 30.0  # Assuming 30 fps
            print(f"🎥 Video saved: {video_path} ({duration:.1f}s, {len(frames)} frames)")
        
        # Cleanup
        del self.violation_records[player_id]
        if player_id in self.violation_start_frames:
            del self.violation_start_frames[player_id]
        
        print(f"✅ Violation recording completed for Player {player_id}")
    
    def draw_boundary(self, frame):
        """Draw the boundary line on frame"""
        if len(self.boundary_points) > 1:
            # Convert to numpy array and draw thin yellow line
            pts = np.array(self.boundary_points, np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)  # Thin yellow line
            
            # Draw boundary points as small red circles
            for i, point in enumerate(self.boundary_points):
                cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)  # Small red circles
            
            # Add boundary label
            cv2.putText(frame, "BOUNDARY", (self.boundary_points[0][0], self.boundary_points[0][1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return frame

def main():
    tracker = PlayerTracker()
    
    # Open video to get original dimensions
    cap = cv2.VideoCapture(get_player_tracking_video())  # Centralized video config
    
    # Apply frame processing configuration
    frame_config = get_frame_config()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_config['height'])
    cap.set(cv2.CAP_PROP_FPS, frame_config['fps'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, frame_config['buffer_size'])
    
    if not cap.isOpened():
        print("Error: Could not open video")
        return
    
    # Get original video dimensions
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    orig_h, orig_w = first_frame.shape[:2]
    target_width = 1280
    scale_factor = target_width / float(orig_w)
    
    # Scale boundary points to match display resolution
    tracker.scale_boundary_points(scale_factor)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Original video: {orig_w}x{orig_h}")
    print(f"Display size: {target_width}x{int(orig_h * scale_factor)}")
    print(f"Scale factor: {scale_factor:.3f}")
    print(f"Boundary points in tracker: {tracker.boundary_points}")
    
    print("Starting player tracking...")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame to fit screen (no cropping)
        target_width = 1280
        orig_h, orig_w = frame.shape[:2]
        scale_factor = target_width / float(orig_w)
        new_dim = (target_width, int(orig_h * scale_factor))
        frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
        
        # ALWAYS draw boundary first (before processing)
        frame = tracker.draw_boundary(frame)
        
        # Process frame
        frame, violations = tracker.process_frame(frame)
        
        # Display statistics
        stats_text = [
            f"Frame: {tracker.frame_count}",
            f"Boundary Points: {len(tracker.boundary_points)}",
            f"Active Players: {len(tracker.stable_players)}",
            f"Active Violations: {len(violations)}",
            f"Violating Players: {list(violations) if violations else 'None'}"
        ]
        
        # Draw stats background
        cv2.rectangle(frame, (10, 10), (450, 140), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (450, 140), (255, 255, 255), 2)
        
        for i, text in enumerate(stats_text):
            cv2.putText(frame, text, (20, 35 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Create resizable window that fits screen
        cv2.namedWindow('Player Tracking', cv2.WINDOW_NORMAL)
        cv2.imshow('Player Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"Tracking completed.")

if __name__ == "__main__":
    main()