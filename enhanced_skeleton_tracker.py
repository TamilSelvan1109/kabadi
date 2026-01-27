import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
from datetime import datetime
from collections import deque
import threading

class EnhancedSkeletonTracker:
    def __init__(self):
        # Configuration
        self.VIDEO_SOURCE = 'assets/back_angle_video1.mp4'
        self.BOUNDARY_POINTS = self.load_boundary_points()
        
        # Enhanced tracking parameters
        self.MAX_PLAYERS = 8
        self.TRACKING_HISTORY_SIZE = 10
        self.VIOLATION_CLIP_DURATION = 3.0  # seconds
        self.FPS = 30
        
        # Player tracking
        self.players = {}  # {player_id: PlayerData}
        self.next_player_id = 1
        self.frame_count = 0
        self.current_poses = []
        
        # State management
        self.paused = False
        self.selection_mode = False
        self.recording_violations = True
        
        # Violation recording
        self.violation_clips = {}  # {player_id: VideoWriter}
        self.violation_frames = {}  # {player_id: deque of frames}
        
        # MediaPipe setup with enhanced settings
        self.pose = self.init_mediapipe()
        self.cap = cv2.VideoCapture(self.VIDEO_SOURCE)
        
        # Video properties
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Create directories
        self.ensure_directories()
        
        print(f"Enhanced Skeleton Tracker initialized")
        print(f"Video: {self.original_width}x{self.original_height} @ {self.video_fps}fps")
    
    def load_boundary_points(self):
        """Load boundary points from config"""
        try:
            with open("config.json", "r") as f:
                data = json.load(f)
                return [tuple(p) for p in data["boundary_points"]]
        except:
            return [(100, 400), (500, 400)]  # Default boundary
    
    def ensure_directories(self):
        """Create necessary directories"""
        directories = ['violations', 'violation_clips', 'logs']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def init_mediapipe(self):
        """Initialize MediaPipe with optimal multi-person settings"""
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
                running_mode=RunningMode.VIDEO,
                num_poses=self.MAX_PLAYERS,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4
            )
            
            return PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"MediaPipe initialization error: {e}")
            exit()
    
    class PlayerData:
        def __init__(self, player_id, pose_index):
            self.player_id = player_id
            self.pose_index = pose_index
            self.violations = 0
            self.is_out = False
            self.last_violation_time = 0
            self.tracking_history = deque(maxlen=10)
            self.skeleton_points = {}
            self.ground_contact_points = []
            self.is_violating = False
            self.violation_start_time = None
    
    def get_skeleton_landmarks(self, pose_landmarks):
        """Extract comprehensive skeleton landmarks"""
        landmarks = {
            # Head and face
            'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
            'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
            'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
            
            # Upper body
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_pinky': 17, 'right_pinky': 18,
            'left_index': 19, 'right_index': 20,
            'left_thumb': 21, 'right_thumb': 22,
            
            # Lower body
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        skeleton = {}
        for name, idx in landmarks.items():
            if idx < len(pose_landmarks) and pose_landmarks[idx].visibility > 0.3:
                skeleton[name] = {
                    'x': pose_landmarks[idx].x,
                    'y': pose_landmarks[idx].y,
                    'z': pose_landmarks[idx].z if hasattr(pose_landmarks[idx], 'z') else 0,
                    'visibility': pose_landmarks[idx].visibility
                }
        return skeleton
    
    def get_ground_contact_points(self, skeleton, w, h):
        """Get all possible ground contact points"""
        ground_points = []
        
        # Priority order for ground contact detection
        foot_landmarks = [
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
            'left_ankle', 'right_ankle',
            'left_knee', 'right_knee'
        ]
        
        for landmark in foot_landmarks:
            if landmark in skeleton:
                x = int(skeleton[landmark]['x'] * w)
                y = int(skeleton[landmark]['y'] * h)
                confidence = skeleton[landmark]['visibility']
                ground_points.append((x, y, landmark, confidence))
        
        return ground_points
    
    def check_boundary_violation(self, ground_points):
        """Enhanced boundary violation detection"""
        if not self.BOUNDARY_POINTS or not ground_points:
            return False, None, None
        
        violations = []
        
        for x, y, point_name, confidence in ground_points:
            if self.point_crosses_boundary(x, y):
                violations.append((x, y, point_name, confidence))
        
        if violations:
            # Return the most confident violation
            best_violation = max(violations, key=lambda v: v[3])
            return True, best_violation[2], (best_violation[0], best_violation[1])
        
        return False, None, None
    
    def point_crosses_boundary(self, x, y):
        """Check if point crosses the boundary line"""
        if len(self.BOUNDARY_POINTS) < 2:
            return False
        
        # For each boundary segment
        for i in range(len(self.BOUNDARY_POINTS) - 1):
            p1 = self.BOUNDARY_POINTS[i]
            p2 = self.BOUNDARY_POINTS[i + 1]
            
            # Calculate line equation
            x1, y1 = p1
            x2, y2 = p2
            
            if x1 == x2:  # Vertical line
                if min(y1, y2) <= y <= max(y1, y2) and x > x1:
                    return True
            else:
                # Calculate y on the line for given x
                if min(x1, x2) <= x <= max(x1, x2):
                    m = (y2 - y1) / (x2 - x1)
                    line_y = y1 + m * (x - x1)
                    if y > line_y + 10:  # 10px threshold
                        return True
        
        return False
    
    def draw_enhanced_skeleton(self, frame, skeleton, player_id=None, is_violation=False, is_out=False):
        """Draw enhanced skeleton with better visualization"""
        w, h = frame.shape[1], frame.shape[0]
        
        # Color coding based on status
        if is_out:
            color = (0, 0, 139)  # Dark red for OUT players
            thickness = 3
        elif is_violation:
            color = (0, 0, 255)  # Bright red for violations
            thickness = 3
        elif player_id:
            color = (0, 255, 0)  # Green for assigned players
            thickness = 2
        else:
            color = (255, 255, 255)  # White for unassigned
            thickness = 1
        
        # Enhanced skeleton connections
        connections = [
            # Head connections
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            
            # Arms
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('left_wrist', 'left_thumb'),
            ('left_wrist', 'left_index'),
            ('left_wrist', 'left_pinky'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            ('right_wrist', 'right_thumb'),
            ('right_wrist', 'right_index'),
            ('right_wrist', 'right_pinky'),
            
            # Legs
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_foot_index'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot_index')
        ]
        
        # Draw skeleton connections
        for start, end in connections:
            if start in skeleton and end in skeleton:
                start_point = (int(skeleton[start]['x'] * w), int(skeleton[start]['y'] * h))
                end_point = (int(skeleton[end]['x'] * w), int(skeleton[end]['y'] * h))
                cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw key points with different sizes and colors
        key_points = {
            'head': ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear'],
            'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
            'arms': ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
            'legs': ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
            'feet': ['left_heel', 'right_heel', 'left_foot_index', 'right_foot_index']
        }
        
        for category, points in key_points.items():
            for point in points:
                if point in skeleton:
                    x = int(skeleton[point]['x'] * w)
                    y = int(skeleton[point]['y'] * h)
                    
                    if category == 'feet':
                        # Special highlighting for feet (ground contact points)
                        cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)  # Yellow
                        cv2.circle(frame, (x, y), 8, color, 2)  # Colored border
                    elif category == 'head':
                        cv2.circle(frame, (x, y), 6, color, -1)
                    else:
                        cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw player ID and status
        if player_id and 'nose' in skeleton:
            nose_x = int(skeleton['nose']['x'] * w)
            nose_y = int(skeleton['nose']['y'] * h)
            
            # Status text
            if is_out:
                text = f"P{player_id}-OUT"
                text_color = (0, 0, 255)
            elif is_violation:
                text = f"P{player_id}-VIOLATION"
                text_color = (0, 0, 255)
            else:
                text = f"P{player_id}"
                text_color = color
            
            # Draw text with background
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (nose_x - text_size[0]//2 - 5, nose_y - 40), 
                         (nose_x + text_size[0]//2 + 5, nose_y - 15), (0, 0, 0), -1)
            cv2.putText(frame, text, (nose_x - text_size[0]//2, nose_y - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    def start_violation_recording(self, player_id, frame):
        """Start recording violation clip for a player"""
        if not self.recording_violations:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violation_clips/P{player_id}_violation_{timestamp}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.violation_clips[player_id] = cv2.VideoWriter(
            filename, fourcc, self.video_fps, 
            (self.original_width, self.original_height)
        )
        
        # Initialize frame buffer for this player
        self.violation_frames[player_id] = deque(maxlen=int(self.video_fps * self.VIOLATION_CLIP_DURATION))
        
        print(f"Started recording violation clip for P{player_id}: {filename}")
    
    def record_violation_frame(self, player_id, frame):
        """Record frame for violation clip"""
        if player_id in self.violation_clips and self.violation_clips[player_id]:
            self.violation_clips[player_id].write(frame)
        
        if player_id in self.violation_frames:
            self.violation_frames[player_id].append(frame.copy())
    
    def stop_violation_recording(self, player_id):
        """Stop recording violation clip"""
        if player_id in self.violation_clips and self.violation_clips[player_id]:
            self.violation_clips[player_id].release()
            del self.violation_clips[player_id]
            print(f"Stopped recording violation clip for P{player_id}")
        
        if player_id in self.violation_frames:
            del self.violation_frames[player_id]
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.selection_mode:
            # Find closest pose to click
            min_distance = float('inf')
            closest_pose_idx = -1
            
            for i, pose_landmarks in enumerate(self.current_poses):
                if pose_landmarks and len(pose_landmarks) > 0:
                    # Use nose position for head detection
                    head_x = int(pose_landmarks[0].x * self.original_width)
                    head_y = int(pose_landmarks[0].y * self.original_height)
                    
                    distance = np.sqrt((x - head_x)**2 + (y - head_y)**2)
                    if distance < min_distance and distance < 100:
                        min_distance = distance
                        closest_pose_idx = i
            
            if closest_pose_idx != -1:
                # Check if already assigned
                already_assigned = any(player.pose_index == closest_pose_idx 
                                     for player in self.players.values())
                
                if not already_assigned:
                    # Create new player
                    player_data = self.PlayerData(self.next_player_id, closest_pose_idx)
                    self.players[self.next_player_id] = player_data
                    
                    print(f"Assigned P{self.next_player_id} to pose {closest_pose_idx}")
                    self.next_player_id += 1
    
    def process_frame(self, frame):
        """Process frame for pose detection and tracking"""
        if not self.paused:
            self.frame_count += 1
            
            # MediaPipe processing
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = self.pose.detect_for_video(mp_image, int(time.time() * 1000))
            self.current_poses = results.pose_landmarks if results.pose_landmarks else []
            
            # Update player tracking
            self.update_player_tracking(frame)
    
    def update_player_tracking(self, frame):
        """Update tracking for all players"""
        w, h = frame.shape[1], frame.shape[0]
        
        for player_id, player in self.players.items():
            if player.is_out:
                continue
            
            pose_idx = player.pose_index
            if pose_idx < len(self.current_poses) and self.current_poses[pose_idx]:
                # Get skeleton data
                skeleton = self.get_skeleton_landmarks(self.current_poses[pose_idx])
                player.skeleton_points = skeleton
                
                # Get ground contact points
                ground_points = self.get_ground_contact_points(skeleton, w, h)
                player.ground_contact_points = ground_points
                
                # Check for violations
                is_violation, violation_point, violation_pos = self.check_boundary_violation(ground_points)
                
                current_time = time.time()
                
                if is_violation:
                    if not player.is_violating:
                        # Start of new violation
                        player.is_violating = True
                        player.violation_start_time = current_time
                        self.start_violation_recording(player_id, frame)
                    
                    # Continue recording
                    self.record_violation_frame(player_id, frame)
                    
                    # Count violation after 1 second
                    if (current_time - player.last_violation_time) > 2.0:
                        player.violations += 1
                        player.last_violation_time = current_time
                        
                        # Save screenshot
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = f"violations/P{player_id}_violation_{timestamp}.jpg"
                        cv2.imwrite(screenshot_path, frame)
                        
                        print(f"VIOLATION: P{player_id} at {violation_point} - Total: {player.violations}")
                        
                        # Check if player should be out
                        if player.violations >= 3:
                            player.is_out = True
                            self.stop_violation_recording(player_id)
                            print(f"PLAYER OUT: P{player_id}")
                
                else:
                    if player.is_violating:
                        # End of violation
                        player.is_violating = False
                        self.stop_violation_recording(player_id)
                
                # Update tracking history
                if 'nose' in skeleton:
                    nose_pos = (skeleton['nose']['x'], skeleton['nose']['y'])
                    player.tracking_history.append(nose_pos)
    
    def draw_ui(self, frame):
        """Draw comprehensive user interface"""
        h, w = frame.shape[:2]
        
        # Draw boundary line
        if self.BOUNDARY_POINTS:
            for i in range(len(self.BOUNDARY_POINTS) - 1):
                cv2.line(frame, self.BOUNDARY_POINTS[i], self.BOUNDARY_POINTS[i+1], 
                        (0, 255, 255), 3)
        
        # Status panel background
        cv2.rectangle(frame, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 200), (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "Enhanced Skeleton Tracker", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Statistics
        total_players = len(self.players)
        active_players = sum(1 for p in self.players.values() if not p.is_out)
        out_players = total_players - active_players
        
        cv2.putText(frame, f"Total Players: {total_players}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Active: {active_players} | Out: {out_players}", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mode indicators
        selection_color = (0, 255, 0) if self.selection_mode else (0, 0, 255)
        cv2.putText(frame, f"Selection: {'ON' if self.selection_mode else 'OFF'}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, selection_color, 1)
        
        recording_color = (0, 255, 0) if self.recording_violations else (0, 0, 255)
        cv2.putText(frame, f"Recording: {'ON' if self.recording_violations else 'OFF'}", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, recording_color, 1)
        
        # Player status
        y_offset = 140
        for player_id, player in self.players.items():
            if player.is_out:
                status = f"P{player_id}: OUT ({player.violations})"
                color = (0, 0, 255)
            elif player.is_violating:
                status = f"P{player_id}: VIOLATING ({player.violations})"
                color = (0, 165, 255)  # Orange
            else:
                status = f"P{player_id}: OK ({player.violations})"
                color = (0, 255, 0)
            
            cv2.putText(frame, status, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 15
        
        # Controls
        controls = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "S - Selection Mode", 
            "R - Toggle Recording",
            "C - Clear All Players",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, control, (w - 250, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def run(self):
        """Main application loop"""
        window_name = 'Enhanced Skeleton Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        print("\nENHANCED SKELETON TRACKER")
        print("=" * 50)
        print("Features:")
        print("- Multi-player skeleton tracking")
        print("- Red skeleton coloring for violations")
        print("- Automatic violation clip recording")
        print("- Enhanced boundary detection")
        print("- Comprehensive player management")
        print("=" * 50)
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            self.process_frame(frame)
            
            # Draw all skeletons
            assigned_indices = set()
            
            # Draw assigned players with enhanced visualization
            for player_id, player in self.players.items():
                pose_idx = player.pose_index
                assigned_indices.add(pose_idx)
                
                if (pose_idx < len(self.current_poses) and 
                    self.current_poses[pose_idx] and 
                    player.skeleton_points):
                    
                    self.draw_enhanced_skeleton(
                        frame, player.skeleton_points, player_id, 
                        player.is_violating, player.is_out
                    )
            
            # Draw unassigned skeletons
            for i, pose_landmarks in enumerate(self.current_poses):
                if i not in assigned_indices and pose_landmarks:
                    skeleton = self.get_skeleton_landmarks(pose_landmarks)
                    self.draw_enhanced_skeleton(frame, skeleton)
            
            # Draw UI
            self.draw_ui(frame)
            
            # Show frame
            cv2.imshow(window_name, frame)
            cv2.setMouseCallback(window_name, self.mouse_callback)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.paused = not self.paused
                print(f"Video {'paused' if self.paused else 'resumed'}")
            elif key == ord('s'):
                self.selection_mode = not self.selection_mode
                print(f"Selection mode {'ON' if self.selection_mode else 'OFF'}")
            elif key == ord('r'):
                self.recording_violations = not self.recording_violations
                print(f"Violation recording {'ON' if self.recording_violations else 'OFF'}")
            elif key == ord('c'):
                # Clear all players
                for player_id in list(self.violation_clips.keys()):
                    self.stop_violation_recording(player_id)
                self.players.clear()
                self.next_player_id = 1
                print("All players cleared")
        
        # Cleanup
        for player_id in list(self.violation_clips.keys()):
            self.stop_violation_recording(player_id)
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = EnhancedSkeletonTracker()
    tracker.run()