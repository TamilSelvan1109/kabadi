import numpy as np
from config_manager import config

class SkeletonTracker:
    def __init__(self, player_tracker):
        self.player_tracker = player_tracker
        
        # Key skeleton landmarks for tracking
        self.SKELETON_LANDMARKS = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_heel': 29,
            'right_heel': 30
        }
    
    def get_skeleton_points(self, pose_landmarks, w, h):
        """Extract key skeleton points with confidence filtering"""
        points = {}
        
        for name, idx in self.SKELETON_LANDMARKS.items():
            if (idx < len(pose_landmarks) and 
                pose_landmarks[idx].visibility > config.FOOT_DETECTION_CONFIDENCE):
                points[name] = (
                    int(pose_landmarks[idx].x * w),
                    int(pose_landmarks[idx].y * h)
                )
        
        return points
    
    def get_ground_contact_point(self, skeleton_points, player_id):
        """Get the lowest visible body part for ground contact detection"""
        # Check manual correction first
        if (player_id in self.player_tracker.selected_players and 
            'manual_ground_point' in self.player_tracker.selected_players[player_id]):
            return self.player_tracker.selected_players[player_id]['manual_ground_point']
        
        # Priority order for ground contact (lowest to highest priority)
        ground_candidates = []
        
        # Add available points with their Y coordinates
        if 'left_heel' in skeleton_points:
            ground_candidates.append(('left_heel', skeleton_points['left_heel']))
        if 'right_heel' in skeleton_points:
            ground_candidates.append(('right_heel', skeleton_points['right_heel']))
        if 'left_ankle' in skeleton_points:
            ground_candidates.append(('left_ankle', skeleton_points['left_ankle']))
        if 'right_ankle' in skeleton_points:
            ground_candidates.append(('right_ankle', skeleton_points['right_ankle']))
        if 'left_knee' in skeleton_points:
            ground_candidates.append(('left_knee', skeleton_points['left_knee']))
        if 'right_knee' in skeleton_points:
            ground_candidates.append(('right_knee', skeleton_points['right_knee']))
        
        if not ground_candidates:
            return None
        
        # Return the lowest point (highest Y value)
        lowest_point = max(ground_candidates, key=lambda x: x[1][1])
        return lowest_point[1]
    
    def get_player_center(self, skeleton_points):
        """Get player's center point for tracking"""
        # Use torso center (average of shoulders and hips)
        center_points = []
        
        for point_name in ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip']:
            if point_name in skeleton_points:
                center_points.append(skeleton_points[point_name])
        
        if center_points:
            avg_x = sum(p[0] for p in center_points) // len(center_points)
            avg_y = sum(p[1] for p in center_points) // len(center_points)
            return (avg_x, avg_y)
        
        # Fallback to nose if torso not available
        return skeleton_points.get('nose', None)
    
    def draw_skeleton(self, frame, skeleton_points, player_id, is_violation=False):
        """Draw skeleton connections"""
        import cv2
        
        # Define skeleton connections
        connections = [
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'),
            ('right_hip', 'right_knee'),
            ('left_knee', 'left_ankle'),
            ('right_knee', 'right_ankle'),
            ('left_ankle', 'left_heel'),
            ('right_ankle', 'right_heel')
        ]
        
        # Color based on violation status
        skeleton_color = (0, 0, 255) if is_violation else (0, 255, 0)
        
        # Draw connections
        for start, end in connections:
            if start in skeleton_points and end in skeleton_points:
                cv2.line(frame, skeleton_points[start], skeleton_points[end], skeleton_color, 2)
        
        # Draw key points
        for name, point in skeleton_points.items():
            if name in ['left_heel', 'right_heel', 'left_ankle', 'right_ankle']:
                # Highlight ground contact points
                cv2.circle(frame, point, 6, (255, 255, 0), -1)
            else:
                cv2.circle(frame, point, 4, skeleton_color, -1)
        
        # Draw player ID near center
        center = self.get_player_center(skeleton_points)
        if center:
            cv2.putText(frame, f"P{player_id}", (center[0] - 15, center[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, skeleton_color, 2)