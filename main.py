import cv2
import mediapipe as mp
import time
import os

from config_manager import config
from player_tracker import PlayerTracker
from foot_detector import FootDetector
from violation_detector import ViolationDetector
from visualizer import Visualizer

class KabaddiTracker:
    def __init__(self):
        self.player_tracker = PlayerTracker()
        self.foot_detector = FootDetector(self.player_tracker)
        self.violation_detector = ViolationDetector(self.player_tracker)
        self.visualizer = Visualizer()
        
        # State variables
        self.frame_count = 0
        self.current_poses = []
        self.paused = False
        self.selection_mode = False
        self.manual_foot_mode = False
        self.correcting_player_id = None
        self.correcting_foot = None
        
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
                min_pose_detection_confidence=config.POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=config.POSE_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.TRACKING_CONFIDENCE
            )
            
            pose = PoseLandmarker.create_from_options(options)
            print("MediaPipe initialized successfully")
            return pose
        except Exception as e:
            print(f"ERROR: Failed to initialize MediaPipe: {e}")
            exit()
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for player selection and foot correction"""
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse clicked at ({x}, {y})")
            
            if self.selection_mode:
                print(f"Selection mode: Finding closest pose to ({x}, {y})")
                self._handle_player_selection(x, y, param['width'], param['height'])
            elif self.manual_foot_mode and self.correcting_player_id and self.correcting_foot:
                print(f"Manual foot mode: Setting {self.correcting_foot} foot for P{self.correcting_player_id}")
                self._handle_foot_correction(x, y)
    
    def _handle_player_selection(self, x, y, w, h):
        """Handle player selection click"""
        closest_pose_idx = self.player_tracker.find_closest_pose(x, y, self.current_poses, w, h)
        
        if closest_pose_idx != -1:
            self.player_tracker.assign_player(closest_pose_idx, self.frame_count)
    
    def _handle_foot_correction(self, x, y):
        """Handle manual foot correction click"""
        success = self.player_tracker.set_manual_foot(
            self.correcting_player_id, self.correcting_foot, (x, y)
        )
        if success:
            self.correcting_foot = None
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
                
                # Get head position
                head_x = int(pose_landmarks[0].x * w)
                head_y = int(pose_landmarks[0].y * h)
                head_position = (head_x, head_y)
                
                # Get foot positions
                left_foot = self.foot_detector.get_foot_position(
                    pose_landmarks, 'left', player_id, w, h)
                right_foot = self.foot_detector.get_foot_position(
                    pose_landmarks, 'right', player_id, w, h)
                
                # Check violations
                left_violation, right_violation = self.violation_detector.check_player_violation(
                    player_id, left_foot, right_foot, self.frame_count)
                
                violation_detected = left_violation or right_violation
                is_player_out = self.violation_detector.is_player_out(player_id)
                
                # Draw arrow above head
                self.visualizer.draw_arrow_above_head(
                    frame, head_position, player_id, violation_detected, is_player_out)
                
                # Draw feet
                self.visualizer.draw_foot(frame, left_foot, 'left', left_violation)
                self.visualizer.draw_foot(frame, right_foot, 'right', right_violation)
                
                # Handle violation evidence
                if violation_detected and not self.paused:
                    if self.violation_detector.should_save_evidence(player_id):
                        foot_type = "Left" if left_violation else "Right"
                        if left_violation and right_violation:
                            foot_type = "Both"
                        self.violation_detector.save_evidence(frame, player_id, foot_type)
                    
                    self.visualizer.draw_violation_warning(frame, player_id, violation_count)
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
            self.manual_foot_mode, self.correcting_player_id, self.correcting_foot, 
            self.paused, violation_counts, out_players)
        
        self.visualizer.draw_instructions(frame)
    
    def handle_key_input(self, key):
        """Handle keyboard input"""
        if key == ord('q'):
            return False
        elif key == ord(' '):
            self.paused = not self.paused
            print(f"Video {'paused' if self.paused else 'resumed'}")
        elif key == ord('s'):
            self.selection_mode = not self.selection_mode
            self.manual_foot_mode = False
            print(f"Selection mode {'ON' if self.selection_mode else 'OFF'}")
        elif key == ord('f'):
            self.manual_foot_mode = not self.manual_foot_mode
            self.selection_mode = False
            print(f"Manual foot mode {'ON' if self.manual_foot_mode else 'OFF'}")
        elif key == ord('r'):
            self.player_tracker.reset_all_players()
            self.correcting_player_id = None
            self.correcting_foot = None
        elif key >= ord('1') and key <= ord('9'):
            player_id = key - ord('0')
            if player_id in self.player_tracker.get_players():
                self.correcting_player_id = player_id
                self.correcting_foot = None
                print(f"Selected P{player_id} for foot correction. Press L or R.")
        elif key == ord('l') and self.correcting_player_id:
            self.correcting_foot = 'left'
            print(f"Click to set P{self.correcting_player_id} left foot position")
        elif key == ord('r') and self.correcting_player_id and self.manual_foot_mode:
            self.correcting_foot = 'right'
            print(f"Click to set P{self.correcting_player_id} right foot position")
        
        return True
    
    def run(self):
        """Main application loop"""
        window_name = 'Kabaddi Multi-Player Tracker'
        
        print("\nKABADDI TRACKER CONTROLS:")
        print("SPACE - Pause/Resume")
        print("S - Selection mode (click on players)")
        print("F - Manual foot correction mode")
        print("1-9 - Select player for correction")
        print("L/R - Select left/right foot")
        print("R - Reset all players")
        print("Q - Quit")
        
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
    tracker = KabaddiTracker()
    tracker.run()