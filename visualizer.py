import cv2
import numpy as np
from config_manager import config

class Visualizer:
    def __init__(self):
        pass
    
    def draw_arrow_above_head(self, frame, head_position, player_id, is_violation=False, is_out=False):
        """Draw arrow above player's head with status colors"""
        x, y = head_position
        
        # Color logic: Red if out, Orange if violation, Green if normal
        if is_out:
            arrow_color = (0, 0, 139)  # Dark red for OUT
            text_color = (0, 0, 255)   # Bright red text
        elif is_violation:
            arrow_color = (0, 0, 255)  # Red for violation
            text_color = (0, 0, 255)
        else:
            arrow_color = (0, 255, 0)  # Green for normal
            text_color = (0, 255, 0)
        
        # Arrow shape
        arrow_tip = (x, y - 40)
        arrow_left = (x - 15, y - 25)
        arrow_right = (x + 15, y - 25)
        arrow_base_left = (x - 8, y - 25)
        arrow_base_right = (x + 8, y - 25)
        arrow_base_bottom = (x, y - 15)
        
        arrow_points = np.array([arrow_tip, arrow_left, arrow_base_left, 
                                arrow_base_bottom, arrow_base_right, arrow_right], np.int32)
        cv2.fillPoly(frame, [arrow_points], arrow_color)
        
        # Player ID and status text
        if is_out:
            cv2.putText(frame, f"P{player_id}-OUT", (x - 25, y - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
        else:
            cv2.putText(frame, f"P{player_id}", (x - 15, y - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
    
    def draw_foot(self, frame, foot_position, foot_type, is_violation=False):
        """Draw foot position with label"""
        if not foot_position:
            return
        
        x, y = foot_position
        foot_color = (0, 0, 255) if is_violation else (255, 0, 0)
        
        # Draw foot circle
        cv2.circle(frame, (x, y), 8, foot_color, -1)
        
        # Draw foot label
        label = "L" if foot_type == 'left' else "R"
        cv2.putText(frame, label, (x - 5, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, foot_color, 1)
        
        # Draw violation highlight
        if is_violation:
            cv2.circle(frame, (x, y), 25, (0, 0, 255), 3)
    
    def draw_boundary_line(self, frame):
        """Draw the boundary line"""
        points = config.BOUNDARY_POINTS
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (0, 255, 255), 3)
    
    def draw_unassigned_poses(self, frame, poses, assigned_indices, w, h, selection_mode):
        """Draw circles for unassigned poses"""
        for i, pose_landmarks in enumerate(poses):
            if pose_landmarks and i not in assigned_indices:
                head_x = int(pose_landmarks[0].x * w)
                head_y = int(pose_landmarks[0].y * h)
                
                cv2.circle(frame, (head_x, head_y), 8, (255, 255, 255), -1)
                
                if selection_mode:
                    cv2.putText(frame, "Click", (head_x - 15, head_y - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def draw_status_info(self, frame, player_count, selection_mode, manual_foot_mode, 
                        correcting_player_id, correcting_foot, paused, violation_counts, out_players):
        """Draw status information including violations"""
        h = frame.shape[0]
        status_y = h - 200
        
        cv2.putText(frame, f"Players: {player_count}", (10, status_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(frame, f"Selection: {'ON' if selection_mode else 'OFF'}", 
                    (10, status_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if selection_mode else (0, 0, 255), 2)
        
        cv2.putText(frame, f"Manual Foot: {'ON' if manual_foot_mode else 'OFF'}", 
                    (10, status_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (0, 255, 0) if manual_foot_mode else (0, 0, 255), 2)
        
        cv2.putText(frame, f"Paused: {'YES' if paused else 'NO'}", 
                    (10, status_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                    (255, 0, 0) if paused else (0, 255, 0), 2)
        
        # Show violation counts
        y_offset = 100
        for player_id, count in violation_counts.items():
            color = (0, 0, 255) if player_id in out_players else (255, 255, 0)
            status = "OUT" if player_id in out_players else f"Violations: {count}"
            cv2.putText(frame, f"P{player_id}: {status}", (10, status_y + y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            y_offset += 20
        
        if correcting_player_id:
            foot_text = correcting_foot or "select L/R"
            cv2.putText(frame, f"Correcting P{correcting_player_id} {foot_text}", 
                       (10, status_y + y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def draw_instructions(self, frame):
        """Draw control instructions"""
        instructions = [
            "CONTROLS:",
            "SPACE - Pause/Resume",
            "S - Selection Mode",
            "F - Manual Foot Mode", 
            "1-9 - Select player",
            "L/R - Select foot",
            "R - Reset players",
            "Q - Quit"
        ]
        
        y_offset = 30
        for i, instruction in enumerate(instructions):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(frame, instruction, (10, y_offset + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def draw_violation_warning(self, frame, player_id, violation_count):
        """Draw violation warning text"""
        cv2.putText(frame, f"VIOLATION! P{player_id}", (50, 50 + violation_count * 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)