import cv2
import time
from datetime import datetime
from config_manager import config
from violation_logger import ViolationLogger

class ViolationDetector:
    def __init__(self, player_tracker):
        self.player_tracker = player_tracker
        self.violation_logger = ViolationLogger()
        self.player_violation_counts = {}  # Track violations per player
        self.player_out_status = {}  # Track which players are out
        self.MAX_VIOLATIONS = 3  # Player is out after 3 violations
    
    def check_boundary_crossing(self, foot_x, foot_y):
        """Check if foot crosses the boundary line"""
        points = config.BOUNDARY_POINTS
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            # Ensure p1 is left of p2
            x1, y1 = (p1[0], p1[1]) if p1[0] < p2[0] else (p2[0], p2[1])
            x2, y2 = (p2[0], p2[1]) if p1[0] < p2[0] else (p1[0], p1[1])
            
            if x1 <= foot_x <= x2:
                if x2 - x1 == 0:
                    continue
                
                # Calculate line equation
                m = (y2 - y1) / (x2 - x1)
                c = y1 - (m * x1)
                line_y = (m * foot_x) + c
                
                # Check if foot is below line (violation)
                if foot_y > (line_y + config.BOUNDARY_THRESHOLD):
                    return True
        return False
    
    def check_player_violation(self, player_id, left_foot, right_foot, frame_number):
        """Check if player has any foot violations and handle logging"""
        left_violation = False
        right_violation = False
        
        # Skip if player is already out
        if self.is_player_out(player_id):
            return left_violation, right_violation
        
        if left_foot:
            left_violation = self.check_boundary_crossing(left_foot[0], left_foot[1])
        
        if right_foot:
            right_violation = self.check_boundary_crossing(right_foot[0], right_foot[1])
        
        # Log violations if they occur and cooldown has passed
        if (left_violation or right_violation) and self.should_save_evidence(player_id):
            foot_type = "Left" if left_violation else "Right"
            if left_violation and right_violation:
                foot_type = "Both"
            
            # Log the violation
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.violation_logger.log_violation(player_id, foot_type, timestamp, frame_number)
            
            # Update violation count
            if player_id not in self.player_violation_counts:
                self.player_violation_counts[player_id] = 0
            self.player_violation_counts[player_id] += 1
            
            # Check if player should be marked as out
            if self.player_violation_counts[player_id] >= self.MAX_VIOLATIONS:
                self.mark_player_out(player_id)
        
        return left_violation, right_violation
    
    def save_evidence(self, frame, player_id, foot_type):
        """Save violation evidence with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"violations/LineOut_P{player_id}_{foot_type}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"SAVED: {filename}")
    
    def should_save_evidence(self, player_id):
        """Check if enough time has passed since last violation save"""
        current_time = time.time()
        last_time = self.player_tracker.last_violation_time.get(player_id, 0)
        
        if (current_time - last_time) > config.COOLDOWN_SECONDS:
            self.player_tracker.last_violation_time[player_id] = current_time
            return True
        return False
    
    def is_player_out(self, player_id):
        """Check if player is marked as out"""
        return self.player_out_status.get(player_id, False)
    
    def mark_player_out(self, player_id):
        """Mark player as out due to violations"""
        self.player_out_status[player_id] = True
        self.violation_logger.mark_player_out(player_id, f"Exceeded {self.MAX_VIOLATIONS} violations")
    
    def get_violation_count(self, player_id):
        """Get violation count for player"""
        return self.player_violation_counts.get(player_id, 0)
    
    def reset_player_violations(self, player_id):
        """Reset violations for a player"""
        self.player_violation_counts[player_id] = 0
        self.player_out_status[player_id] = False