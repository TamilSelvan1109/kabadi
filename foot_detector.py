import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import time

class SkeletonTracker:
    def __init__(self):
        # MediaPipe pose connections for full skeleton
        self.POSE_CONNECTIONS = [
            # Face
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            # Torso
            (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # Right arm  
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        
        # Color coding system for different player states
        self.COLORS = {
            'ACTIVE': (0, 255, 0),          # Green - Active player, no violations
            'WARNING': (0, 255, 255),       # Yellow - Warning state, approaching boundary
            'FIRST_VIOLATION': (0, 165, 255),  # Orange - First violation detected
            'MULTIPLE_VIOLATIONS': (0, 0, 255),  # Red - Multiple violations, critical state
            'ELIMINATED': (0, 0, 139),      # Dark Red - Player eliminated (3+ violations)
            'UNASSIGNED': (255, 255, 255),  # White - Unassigned/detected player
            'UNDER_REVIEW': (255, 0, 0),    # Blue - Player under manual review
            'BOUNDARY_LINE': (0, 255, 255), # Cyan - Boundary line
            'FOOT_HIGHLIGHT': (0, 255, 255) # Yellow - Foot landmarks
        }
        
        # Player violation tracking
        self.player_violations = {}
        self.player_states = {}
        self.violation_timestamps = {}
    
    def draw_skeleton(self, frame, pose_landmarks, player_id=None, is_violation=False, is_out=False):
        """Draw complete skeleton with enhanced visibility and status colors"""
        if not pose_landmarks:
            return
        
        h, w = frame.shape[:2]
        
        # Enhanced color coding with better visibility
        if is_out:
            color = (0, 0, 200)  # Bright red
            thickness = 4
            text_color = (255, 255, 255)
        elif is_violation:
            color = (0, 100, 255)  # Orange-red
            thickness = 3
            text_color = (255, 255, 255)
        elif player_id:
            color = (0, 255, 0)  # Bright green
            thickness = 3
            text_color = (0, 0, 0)
        else:
            color = (255, 255, 255)  # White
            thickness = 2
            text_color = (0, 0, 0)
        
        # Draw skeleton connections with enhanced visibility
        for start_idx, end_idx in self.POSE_CONNECTIONS:
            if (start_idx < len(pose_landmarks) and end_idx < len(pose_landmarks) and
                pose_landmarks[start_idx].visibility > 0.2 and  # Lowered threshold
                pose_landmarks[end_idx].visibility > 0.2):
                
                start_point = (
                    int(pose_landmarks[start_idx].x * w),
                    int(pose_landmarks[start_idx].y * h)
                )
                end_point = (
                    int(pose_landmarks[end_idx].x * w),
                    int(pose_landmarks[end_idx].y * h)
                )
                
                # Draw thicker lines with outline for better visibility
                cv2.line(frame, start_point, end_point, (0, 0, 0), thickness + 2)  # Black outline
                cv2.line(frame, start_point, end_point, color, thickness)
        
        # Draw enhanced landmarks
        for i, landmark in enumerate(pose_landmarks):
            if landmark.visibility > 0.2:  # Lowered threshold
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # Different sizes and colors for body parts
                if i == 0:  # Nose - head indicator
                    radius = 8
                    cv2.circle(frame, (x, y), radius + 2, (0, 0, 0), -1)  # Black outline
                    cv2.circle(frame, (x, y), radius, (255, 255, 0), -1)  # Yellow head
                elif i in [11, 12, 23, 24]:  # Shoulders, hips
                    radius = 6
                    cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), -1)
                    cv2.circle(frame, (x, y), radius, color, -1)
                elif i in [27, 28, 29, 30, 31, 32]:  # Feet - most important for tracking
                    radius = 10
                    cv2.circle(frame, (x, y), radius + 3, (0, 0, 0), 3)  # Black outline
                    cv2.circle(frame, (x, y), radius, (0, 255, 255), -1)  # Cyan feet
                    # Add foot labels
                    foot_label = "L" if i in [27, 29, 31] else "R"
                    cv2.putText(frame, foot_label, (x-5, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
                else:
                    radius = 4
                    cv2.circle(frame, (x, y), radius + 1, (0, 0, 0), -1)
                    cv2.circle(frame, (x, y), radius, color, -1)
        
        # Draw enhanced player ID with background
        if player_id and len(pose_landmarks) > 0:
            nose_x = int(pose_landmarks[0].x * w)
            nose_y = int(pose_landmarks[0].y * h)
            
            status_text = f"P{player_id}"
            if is_out:
                status_text += "-OUT"
            elif is_violation:
                status_text += "-VIOLATION"
            
            # Text background for better visibility
            text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (nose_x - text_size[0]//2 - 5, nose_y - 35), 
                         (nose_x + text_size[0]//2 + 5, nose_y - 10), (0, 0, 0), -1)
            cv2.rectangle(frame, (nose_x - text_size[0]//2 - 5, nose_y - 35), 
                         (nose_x + text_size[0]//2 + 5, nose_y - 10), color, 2)
            
            cv2.putText(frame, status_text, (nose_x - text_size[0]//2, nose_y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    def get_ground_contact_points(self, pose_landmarks, w, h):
        """Get ground contact points with confidence"""
        ground_points = []
        foot_landmarks = [29, 30, 31, 32, 27, 28]  # heels, toes, ankles
        
        for idx in foot_landmarks:
            if (idx < len(pose_landmarks) and 
                pose_landmarks[idx].visibility > 0.3):
                
                x = int(pose_landmarks[idx].x * w)
                y = int(pose_landmarks[idx].y * h)
                confidence = pose_landmarks[idx].visibility
                
                ground_points.append({
                    'position': (x, y),
                    'confidence': confidence,
                    'index': idx
                })
        
        ground_points.sort(key=lambda p: p['confidence'], reverse=True)
        return ground_points