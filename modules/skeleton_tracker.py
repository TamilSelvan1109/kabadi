import cv2
import numpy as np
import os

class SkeletonTracker:
    def __init__(self):
        self.mediapipe_working = False
        self.pose_landmarker = None
        
        # Try to initialize MediaPipe with local model file
        self._try_initialize_mediapipe()
    
    def _try_initialize_mediapipe(self):
        """Initialize MediaPipe with downloaded model file"""
        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            
            # Path to downloaded model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'pose_landmarker_lite.task')
            model_path = os.path.abspath(model_path)
            
            if os.path.exists(model_path):
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.IMAGE,
                    num_poses=1,
                    min_pose_detection_confidence=0.3,
                    min_pose_presence_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                
                self.pose_landmarker = vision.PoseLandmarker.create_from_options(options)
                self.mediapipe_working = True
                print("SUCCESS: MediaPipe initialized with local model")
                return
            else:
                print(f"WARNING: Model file not found at {model_path}")
                
        except Exception as e:
            print(f"WARNING: MediaPipe initialization failed: {e}")
        
        print("Using YOLO bounding box for foot tracking")
        self.mediapipe_working = False
    
    def get_foot_position(self, frame, bbox, player_id):
        """Extract foot position and draw skeleton for individual player"""
        x1, y1, x2, y2 = bbox
        skeleton_drawn = False
        
        if self.mediapipe_working and self.pose_landmarker is not None:
            # Crop player region
            pad = 20
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(frame.shape[1], x2 + pad)
            crop_y2 = min(frame.shape[0], y2 + pad)
            
            player_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            if player_crop.size > 0 and player_crop.shape[0] > 30 and player_crop.shape[1] > 20:
                try:
                    import mediapipe as mp
                    
                    # Convert to MediaPipe Image format
                    rgb_crop = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
                    
                    # Detect pose landmarks
                    detection_result = self.pose_landmarker.detect(mp_image)
                    
                    if detection_result.pose_landmarks and len(detection_result.pose_landmarks) > 0:
                        landmarks = detection_result.pose_landmarks[0]  # First person
                        crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
                        
                        # Convert landmarks to frame coordinates
                        pose_points = []
                        for landmark in landmarks:
                            x_coord = int(crop_x1 + landmark.x * crop_w)
                            y_coord = int(crop_y1 + landmark.y * crop_h)
                            pose_points.append((x_coord, y_coord, landmark.visibility))
                        
                        # Draw skeleton
                        self.draw_skeleton(frame, pose_points, player_id)
                        skeleton_drawn = True
                        
                        # Get foot position from ankles (landmarks 27, 28)
                        if len(landmarks) > 28:
                            left_ankle = landmarks[27]  # LEFT_ANKLE
                            right_ankle = landmarks[28]  # RIGHT_ANKLE
                            
                            if left_ankle.visibility > 0.1 or right_ankle.visibility > 0.1:
                                if left_ankle.visibility >= right_ankle.visibility:
                                    foot_x = int(crop_x1 + left_ankle.x * crop_w)
                                    foot_y = int(crop_y1 + left_ankle.y * crop_h)
                                else:
                                    foot_x = int(crop_x1 + right_ankle.x * crop_w)
                                    foot_y = int(crop_y1 + right_ankle.y * crop_h)
                                
                                return (foot_x, foot_y), skeleton_drawn
                
                except Exception as e:
                    pass
        
        # Fallback to bottom center of bounding box (YOLO-based foot tracking)
        return (int((x1 + x2) / 2), y2), skeleton_drawn
    
    def draw_skeleton(self, frame, pose_points, player_id):
        """Draw skeleton with unique color per player"""
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        player_color = colors[player_id % len(colors)]
        
        # Pose connections (key body parts)
        connections = [
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Arms
            (11, 23), (12, 24), (23, 24),  # Torso
            (23, 25), (24, 26), (25, 27), (26, 28)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if (start_idx < len(pose_points) and end_idx < len(pose_points) and
                pose_points[start_idx][2] > 0.1 and pose_points[end_idx][2] > 0.1):
                start_point = (pose_points[start_idx][0], pose_points[start_idx][1])
                end_point = (pose_points[end_idx][0], pose_points[end_idx][1])
                cv2.line(frame, start_point, end_point, player_color, 3)
        
        # Draw ankle points in red (most important for foot tracking)
        for idx in [27, 28]:  # Left and right ankle
            if idx < len(pose_points) and pose_points[idx][2] > 0.1:
                cv2.circle(frame, (pose_points[idx][0], pose_points[idx][1]), 8, (0, 0, 255), -1)