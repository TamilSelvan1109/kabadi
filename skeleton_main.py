import cv2
import mediapipe as mp
import time
import os

from config_manager import config
from player_tracker import PlayerTracker
from skeleton_tracker import SkeletonTracker
from violation_detector import ViolationDetector
from visualizer import Visualizer

class SkeletonKabaddiTracker:
    def __init__(self):
        self.player_tracker = PlayerTracker()
        self.skeleton_tracker = SkeletonTracker(self.player_tracker)
        self.violation_detector = ViolationDetector(self.player_tracker)
        self.visualizer = Visualizer()
        
        # State variables
        self.frame_count = 0
        self.current_poses = []
        self.paused = False
        self.selection_mode = False
        self.manual_correction_mode = False
        self.correcting_player_id = None
        
        # Initialize MediaPipe
        self.pose = self._init_mediapipe()
        
        # Initialize video capture
        self.cap = cv2.VideoCapture(config.VIDEO_SOURCE)
        
        # Create violations directory
        if not os.path.exists('violations'):
            os.makedirs('violations')
    
    def _init_mediapipe(self):
        """Initialize MediaPipe pose detection"""
        try:
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
                running_mode=RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,  # Higher for better skeleton
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            pose = PoseLandmarker.create_from_options(options)
            print("MediaPipe Skeleton Tracker initialized")
            return pose
        except Exception as e:
            print(f"ERROR: Failed to initialize MediaPipe: {e}")
            exit()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection and manual correction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at ({x}, {y})")
            
            if self.selection_mode:
                print("Selection mode: Finding closest pose")
                self._handle_player_selection(x, y, param['width'], param['height'])
            elif self.manual_correction_mode and self.correcting_player_id:
                print(f"Manual correction: Setting ground point for P{self.correcting_player_id}")
                self._handle_manual_correction(x, y)
    
    def _handle_player_selection(self, x, y, w, h):
        """Handle player selection click"""
        closest_pose_idx = self.player_tracker.find_closest_pose(x, y, self.current_poses, w, h)
        
        if closest_pose_idx != -1:
            self.player_tracker.assign_player(closest_pose_idx, self.frame_count)
    
    def _handle_manual_correction(self, x, y):
        """Handle manual ground point correction"""
        if self.correcting_player_id in self.player_tracker.selected_players:
            self.player_tracker.selected_players[self.correcting_player_id]['manual_ground_point'] = (x, y)
            print(f"Set P{self.correcting_player_id} ground point to ({x}, {y})")
            self.correcting_player_id = None
    
    def process_frame(self, frame):
        """Process a single frame for pose detection"""
        if not self.paused:
            self.frame_count += 1
            
            # Convert frame for MediaPipe
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            timestamp_ms = int(time.time() * 1000)
            
            # Get pose results
            results = self.pose.detect_for_video(mp_image, timestamp_ms)
            self.current_poses = results.pose_landmarks if results.pose_landmarks else []
    
    def draw_frame(self, frame):
        """Draw all visual elements on the frame"""
        h, w, _ = frame.shape
        
        # Draw boundary line
        self.visualizer.draw_boundary_line(frame)
        
        # Get assigned pose indices
        assigned_indices = set()
        violation_count = 0
        
        # Process each selected player
        for player_id, player_data in self.player_tracker.get_players().items():
            pose_idx = player_data['pose_index']
            assigned_indices.add(pose_idx)
            
            if pose_idx < len(self.current_poses) and self.current_poses[pose_idx]:
                pose_landmarks = self.current_poses[pose_idx]
                
                # Get skeleton points
                skeleton_points = self.skeleton_tracker.get_skeleton_points(pose_landmarks, w, h)
                
                # Get ground contact point
                ground_point = self.skeleton_tracker.get_ground_contact_point(skeleton_points, player_id)
                
                # Check violation using ground contact point
                violation_detected = False
                if ground_point:
                    violation_detected = self.violation_detector.check_boundary_crossing(
                        ground_point[0], ground_point[1])
                
                is_player_out = self.violation_detector.is_player_out(player_id)
                
                # Log violation if detected
                if violation_detected and not self.paused:
                    if self.violation_detector.should_save_evidence(player_id):
                        from datetime import datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.violation_detector.violation_logger.log_violation(
                            player_id, "Ground_Contact", timestamp, self.frame_count)
                        
                        # Update violation count
                        if player_id not in self.violation_detector.player_violation_counts:
                            self.violation_detector.player_violation_counts[player_id] = 0
                        self.violation_detector.player_violation_counts[player_id] += 1
                        
                        # Check if player should be out
                        if self.violation_detector.player_violation_counts[player_id] >= 3:
                            self.violation_detector.mark_player_out(player_id)
                        
                        # Save evidence
                        self.violation_detector.save_evidence(frame, player_id, "Ground_Contact")
                
                # Draw skeleton
                self.skeleton_tracker.draw_skeleton(frame, skeleton_points, player_id, violation_detected)
                
                # Draw arrow above head
                head_pos = skeleton_points.get('nose')
                if head_pos:
                    self.visualizer.draw_arrow_above_head(
                        frame, head_pos, player_id, violation_detected, is_player_out)
                
                # Highlight ground contact point
                if ground_point:
                    color = (0, 0, 255) if violation_detected else (255, 255, 0)
                    cv2.circle(frame, ground_point, 10, color, 3)
                    cv2.putText(frame, "GROUND", (ground_point[0] - 20, ground_point[1] - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw violation warning
                if violation_detected:
                    cv2.putText(frame, f"BOUNDARY VIOLATION! P{player_id}", 
                               (50, 50 + violation_count * 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    violation_count += 1
        
        # Draw unassigned poses
        self.visualizer.draw_unassigned_poses(
            frame, self.current_poses, assigned_indices, w, h, self.selection_mode)
        
        # Draw status and instructions
        violation_counts = {pid: self.violation_detector.get_violation_count(pid) 
                           for pid in self.player_tracker.get_players().keys()}
        out_players = [pid for pid in self.player_tracker.get_players().keys() 
                      if self.violation_detector.is_player_out(pid)]
        
        self.visualizer.draw_status_info(
            frame, self.player_tracker.get_player_count(), self.selection_mode,
            self.manual_correction_mode, self.correcting_player_id, None, 
            self.paused, violation_counts, out_players)
        
        # Draw skeleton-specific instructions
        self._draw_skeleton_instructions(frame)
    
    def _draw_skeleton_instructions(self, frame):
        """Draw skeleton tracking specific instructions"""
        instructions = [
            "SKELETON TRACKING:",
            "SPACE - Pause/Resume",
            "S - Select players (click heads)",
            "M - Manual ground point correction",
            "1-9 - Select player for correction",
            "R - Reset all players",
            "Q - Quit"
        ]
        
        y_offset = 30
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def handle_key_input(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord(' '):
            self.paused = not self.paused
            print(f"Video {'paused' if self.paused else 'resumed'}")
        elif key == ord('s'):
            self.selection_mode = not self.selection_mode
            self.manual_correction_mode = False
            print(f"Selection mode {'ON' if self.selection_mode else 'OFF'}")
        elif key == ord('m'):
            self.manual_correction_mode = not self.manual_correction_mode
            self.selection_mode = False
            print(f"Manual correction mode {'ON' if self.manual_correction_mode else 'OFF'}")
        elif key == ord('r'):
            self.player_tracker.reset_all_players()
            self.correcting_player_id = None
            print("All players reset")
        elif key >= ord('1') and key <= ord('9'):
            player_id = key - ord('0')
            if player_id in self.player_tracker.get_players():
                self.correcting_player_id = player_id
                print(f"Selected P{player_id} for ground point correction. Click on ground contact point.")
        
        return True
    
    def run(self):
        """Main application loop"""
        window_name = 'Skeleton-Based Kabaddi Tracker'
        
        print("\\nSKELETON KABADDI TRACKER:")
        print("- More reliable than foot tracking")
        print("- Uses full skeleton for better detection")
        print("- Tracks ground contact point automatically")
        print("- Manual correction available")
        
        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
            
            h, w, _ = frame.shape
            
            # Process frame
            self.process_frame(frame)
            
            # Draw everything
            self.draw_frame(frame)
            
            # Show frame and set mouse callback
            cv2.imshow(window_name, frame)
            cv2.setMouseCallback(window_name, self.mouse_callback, {'width': w, 'height': h})
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            if not self.handle_key_input(key):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = SkeletonKabaddiTracker()
    tracker.run()