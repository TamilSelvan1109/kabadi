import cv2
import mediapipe as mp
import time
import os
import json
import numpy as np
from datetime import datetime

class MultiPersonSkeletonTracker:
    def __init__(self):
        # Configuration
        self.VIDEO_SOURCE = 'assets/back_angle_video1.mp4'
        self.BOUNDARY_POINTS = self.load_boundary_points()
        
        # Tracking variables
        self.selected_players = {}  # {player_id: pose_index}
        self.player_skeletons = {}  # {player_id: skeleton_data}
        self.next_player_id = 1
        self.frame_count = 0
        self.current_poses = []
        
        # State variables
        self.paused = False
        self.selection_mode = False
        self.show_all_skeletons = True
        self.show_unassigned = True
        
        # Violation tracking
        self.violations = {}  # {player_id: count}
        self.player_out = {}  # {player_id: bool}
        self.last_violation_time = {}
        
        # MediaPipe setup
        self.pose = self.init_mediapipe()
        self.cap = cv2.VideoCapture(self.VIDEO_SOURCE)
        
        # Ensure full video display
        self.setup_video_display()
        
        # Create output directory
        if not os.path.exists('violations'):
            os.makedirs('violations')
    
    def load_boundary_points(self):
        """Load boundary points from config"""
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    data = json.load(f)
                    return [tuple(p) for p in data["boundary_points"]]
            except Exception as e:
                print(f"Error loading boundary: {e}")
                return []
        return []
    
    def init_mediapipe(self):
        """Initialize MediaPipe with optimal settings for multi-person"""
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
                running_mode=RunningMode.VIDEO,
                num_poses=10,  # Detect up to 10 people
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3
            )
            
            return PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"MediaPipe init error: {e}")
            exit()
    
    def setup_video_display(self):
        """Setup video display to show full frame without cropping"""
        # Get video properties
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"Video: {self.original_width}x{self.original_height} @ {self.fps}fps")
        
        # Calculate display size (maintain aspect ratio)
        screen_width, screen_height = 1920, 1080  # Adjust to your screen
        
        # Scale to fit screen while maintaining aspect ratio
        scale_w = screen_width / self.original_width
        scale_h = screen_height / self.original_height
        self.display_scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        self.display_width = int(self.original_width * self.display_scale)
        self.display_height = int(self.original_height * self.display_scale)
        
        print(f"Display: {self.display_width}x{self.display_height} (scale: {self.display_scale:.2f})")
    
    def get_skeleton_landmarks(self, pose_landmarks):
        """Extract key skeleton points"""
        landmarks = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot': 31, 'right_foot': 32
        }
        
        skeleton = {}
        for name, idx in landmarks.items():
            if idx < len(pose_landmarks) and pose_landmarks[idx].visibility > 0.3:
                skeleton[name] = {
                    'x': pose_landmarks[idx].x,
                    'y': pose_landmarks[idx].y,
                    'visibility': pose_landmarks[idx].visibility
                }
        return skeleton
    
    def get_ground_contact_points(self, skeleton, w, h):
        """Get ground contact points for violation detection"""
        ground_points = []
        
        # Priority order for ground contact
        candidates = ['left_heel', 'right_heel', 'left_foot', 'right_foot', 
                     'left_ankle', 'right_ankle', 'left_knee', 'right_knee']
        
        for candidate in candidates:
            if candidate in skeleton:
                x = int(skeleton[candidate]['x'] * w)
                y = int(skeleton[candidate]['y'] * h)
                ground_points.append((x, y, candidate))
        
        return ground_points
    
    def check_boundary_violation(self, ground_points):
        """Check if any ground point violates boundary"""
        if not self.BOUNDARY_POINTS or not ground_points:
            return False, None
        
        for x, y, point_name in ground_points:
            if self.point_crosses_boundary(x, y):
                return True, point_name
        return False, None
    
    def point_crosses_boundary(self, x, y):
        """Check if point crosses boundary line"""
        for i in range(len(self.BOUNDARY_POINTS) - 1):
            p1 = self.BOUNDARY_POINTS[i]
            p2 = self.BOUNDARY_POINTS[i + 1]
            
            x1, y1 = min(p1[0], p2[0]), p1[1] if p1[0] < p2[0] else p2[1]
            x2, y2 = max(p1[0], p2[0]), p2[1] if p1[0] > p2[0] else p1[1]
            
            if x1 <= x <= x2:
                if x2 - x1 == 0:
                    continue
                m = (y2 - y1) / (x2 - x1)
                line_y = y1 + m * (x - x1)
                if y > line_y + 15:  # 15px threshold
                    return True
        return False
    
    def draw_skeleton(self, frame, skeleton, player_id=None, is_violation=False, is_out=False):
        """Draw complete skeleton with connections"""
        w, h = frame.shape[1], frame.shape[0]
        
        # Color coding
        if is_out:
            color = (0, 0, 139)  # Dark red for OUT
        elif is_violation:
            color = (0, 0, 255)  # Red for violation
        elif player_id:
            color = (0, 255, 0)  # Green for assigned
        else:
            color = (255, 255, 255)  # White for unassigned
        
        # Skeleton connections
        connections = [
            # Head
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            # Torso
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            # Arms
            ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'),
            # Legs
            ('left_hip', 'left_knee'),
            ('left_knee', 'left_ankle'),
            ('left_ankle', 'left_heel'),
            ('left_ankle', 'left_foot'),
            ('right_hip', 'right_knee'),
            ('right_knee', 'right_ankle'),
            ('right_ankle', 'right_heel'),
            ('right_ankle', 'right_foot')
        ]
        
        # Draw connections
        for start, end in connections:
            if start in skeleton and end in skeleton:
                start_point = (int(skeleton[start]['x'] * w), int(skeleton[start]['y'] * h))
                end_point = (int(skeleton[end]['x'] * w), int(skeleton[end]['y'] * h))
                cv2.line(frame, start_point, end_point, color, 2)
        
        # Draw key points
        key_points = ['nose', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
                     'left_heel', 'right_heel', 'left_foot', 'right_foot']
        
        for point in key_points:
            if point in skeleton:
                x = int(skeleton[point]['x'] * w)
                y = int(skeleton[point]['y'] * h)
                
                # Different sizes for different points
                if point in ['left_heel', 'right_heel', 'left_foot', 'right_foot']:
                    cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)  # Yellow for feet
                elif point == 'nose':
                    cv2.circle(frame, (x, y), 6, color, -1)
                else:
                    cv2.circle(frame, (x, y), 4, color, -1)
        
        # Draw player ID
        if player_id and 'nose' in skeleton:
            nose_x = int(skeleton['nose']['x'] * w)
            nose_y = int(skeleton['nose']['y'] * h)
            
            if is_out:
                text = f"P{player_id}-OUT"
                text_color = (0, 0, 255)
            else:
                text = f"P{player_id}"
                text_color = color
            
            cv2.putText(frame, text, (nose_x - 20, nose_y - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection"""
        if event == cv2.EVENT_LBUTTONDOWN and self.selection_mode:
            # Scale click coordinates back to original frame
            orig_x = int(x / self.display_scale)
            orig_y = int(y / self.display_scale)
            
            # Find closest pose
            min_distance = float('inf')
            closest_pose_idx = -1
            
            for i, pose_landmarks in enumerate(self.current_poses):
                if pose_landmarks and len(pose_landmarks) > 0:
                    # Use nose for head position
                    head_x = int(pose_landmarks[0].x * self.original_width)
                    head_y = int(pose_landmarks[0].y * self.original_height)
                    
                    distance = np.sqrt((orig_x - head_x)**2 + (orig_y - head_y)**2)
                    if distance < min_distance and distance < 100:
                        min_distance = distance
                        closest_pose_idx = i
            
            if closest_pose_idx != -1:
                # Check if already assigned
                already_assigned = any(data['pose_index'] == closest_pose_idx 
                                     for data in self.selected_players.values())
                
                if not already_assigned:
                    self.selected_players[self.next_player_id] = {
                        'pose_index': closest_pose_idx,
                        'last_seen': self.frame_count
                    }
                    self.violations[self.next_player_id] = 0
                    self.player_out[self.next_player_id] = False
                    self.last_violation_time[self.next_player_id] = 0
                    
                    print(f"Assigned P{self.next_player_id} to pose {closest_pose_idx}")
                    self.next_player_id += 1
    
    def process_violations(self, frame):
        """Process violations for all selected players"""
        w, h = frame.shape[1], frame.shape[0]
        violation_count = 0
        
        for player_id, player_data in self.selected_players.items():
            if self.player_out.get(player_id, False):
                continue  # Skip players who are out
            
            pose_idx = player_data['pose_index']
            if pose_idx < len(self.current_poses) and self.current_poses[pose_idx]:
                skeleton = self.get_skeleton_landmarks(self.current_poses[pose_idx])
                ground_points = self.get_ground_contact_points(skeleton, w, h)
                
                is_violation, violation_point = self.check_boundary_violation(ground_points)
                
                if is_violation:
                    current_time = time.time()
                    if (current_time - self.last_violation_time[player_id]) > 2.0:  # 2 second cooldown
                        self.violations[player_id] += 1
                        self.last_violation_time[player_id] = current_time
                        
                        # Save evidence
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"violations/Violation_P{player_id}_{timestamp}.jpg"
                        cv2.imwrite(filename, frame)
                        
                        print(f"VIOLATION: P{player_id} ({violation_point}) - Total: {self.violations[player_id]}")
                        
                        # Check if player should be out (3 violations)
                        if self.violations[player_id] >= 3:
                            self.player_out[player_id] = True
                            print(f"PLAYER OUT: P{player_id}")
                    
                    # Draw violation warning
                    cv2.putText(frame, f"VIOLATION! P{player_id}", 
                               (50, 50 + violation_count * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    violation_count += 1
    
    def draw_ui(self, frame):
        """Draw user interface elements"""
        h = frame.shape[0]
        
        # Draw boundary line
        if self.BOUNDARY_POINTS:
            for i in range(len(self.BOUNDARY_POINTS) - 1):
                cv2.line(frame, self.BOUNDARY_POINTS[i], self.BOUNDARY_POINTS[i+1], 
                        (0, 255, 255), 3)
        
        # Status information
        status_y = h - 200
        cv2.putText(frame, f"Players: {len(self.selected_players)}", 
                   (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Selection: {'ON' if self.selection_mode else 'OFF'}", 
                   (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if self.selection_mode else (0, 0, 255), 2)
        
        cv2.putText(frame, f"Show All: {'ON' if self.show_all_skeletons else 'OFF'}", 
                   (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                   (0, 255, 0) if self.show_all_skeletons else (0, 0, 255), 2)
        
        # Player status
        y_offset = 90
        for player_id in self.selected_players.keys():
            violations = self.violations.get(player_id, 0)
            is_out = self.player_out.get(player_id, False)
            
            if is_out:
                status_text = f"P{player_id}: OUT ({violations} violations)"
                color = (0, 0, 255)
            else:
                status_text = f"P{player_id}: {violations} violations"
                color = (255, 255, 0) if violations > 0 else (0, 255, 0)
            
            cv2.putText(frame, status_text, (10, status_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 25
        
        # Instructions
        instructions = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "S - Selection Mode",
            "A - Toggle Show All Skeletons", 
            "U - Toggle Unassigned",
            "R - Reset Players",
            "Q - Quit"
        ]
        
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, instruction, (10, 30 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def run(self):
        """Main application loop"""
        window_name = 'Multi-Person Skeleton Tracker'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)
        
        print("\\nMULTI-PERSON SKELETON TRACKER")
        print("- Tracks up to 10 people simultaneously")
        print("- Full video display without cropping")
        print("- Complete skeleton visualization")
        print("- Automatic violation detection")
        
        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.frame_count += 1
                
                # Process with MediaPipe
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                                   data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results = self.pose.detect_for_video(mp_image, int(time.time() * 1000))
                self.current_poses = results.pose_landmarks if results.pose_landmarks else []
            
            # Draw UI elements
            self.draw_ui(frame)
            
            # Draw all skeletons
            assigned_indices = set()
            
            # Draw assigned players
            for player_id, player_data in self.selected_players.items():
                pose_idx = player_data['pose_index']
                assigned_indices.add(pose_idx)
                
                if pose_idx < len(self.current_poses) and self.current_poses[pose_idx]:
                    skeleton = self.get_skeleton_landmarks(self.current_poses[pose_idx])
                    
                    # Check violation
                    ground_points = self.get_ground_contact_points(skeleton, frame.shape[1], frame.shape[0])
                    is_violation, _ = self.check_boundary_violation(ground_points)
                    is_out = self.player_out.get(player_id, False)
                    
                    self.draw_skeleton(frame, skeleton, player_id, is_violation, is_out)
            
            # Draw unassigned skeletons
            if self.show_unassigned:
                for i, pose_landmarks in enumerate(self.current_poses):
                    if i not in assigned_indices and pose_landmarks:
                        skeleton = self.get_skeleton_landmarks(pose_landmarks)
                        self.draw_skeleton(frame, skeleton)
            
            # Process violations
            if not self.paused:
                self.process_violations(frame)
            
            # Resize for display
            display_frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # Show frame
            cv2.imshow(window_name, display_frame)
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
            elif key == ord('a'):
                self.show_all_skeletons = not self.show_all_skeletons
                print(f"Show all skeletons {'ON' if self.show_all_skeletons else 'OFF'}")
            elif key == ord('u'):
                self.show_unassigned = not self.show_unassigned
                print(f"Show unassigned {'ON' if self.show_unassigned else 'OFF'}")
            elif key == ord('r'):
                self.selected_players.clear()
                self.violations.clear()
                self.player_out.clear()
                self.last_violation_time.clear()
                self.next_player_id = 1
                print("All players reset")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = MultiPersonSkeletonTracker()
    tracker.run()