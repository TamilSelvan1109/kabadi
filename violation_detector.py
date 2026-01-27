import cv2
import time
from datetime import datetime

class ViolationDetector:
    def __init__(self, player_tracker):
        self.player_tracker = player_tracker
        self.player_violation_counts = {}  # Track violations per player
        self.player_out_status = {}  # Track which players are out
        self.MAX_VIOLATIONS = 3  # Player is out after 3 violations
        self.BOUNDARY_THRESHOLD = 10  # Pixels threshold for boundary crossing
    
    def check_boundary_crossing(self, foot_x, foot_y):
        """Check if foot crosses the boundary line"""
        # This will be set from config.json boundary_points
        return False  # Placeholder - will be implemented in app.py
    
    def check_player_violation(self, player_id, ground_points, frame_number):
        """Check if player has ground contact violations using skeleton points"""
        violation_detected = False
        
        # Skip if player is already out
        if self.is_player_out(player_id):
            return violation_detected
        
        # Check top 2 most confident ground contact points
        for point in ground_points[:2]:
            if self.check_boundary_crossing(point['position'][0], point['position'][1]):
                violation_detected = True
                break
        
        # Update violation count
        if violation_detected and self.should_save_evidence(player_id):
            if player_id not in self.player_violation_counts:
                self.player_violation_counts[player_id] = 0
            self.player_violation_counts[player_id] += 1
            
            # Check if player should be marked as out
            if self.player_violation_counts[player_id] >= self.MAX_VIOLATIONS:
                self.mark_player_out(player_id)
        
        return violation_detected
    
    def should_save_evidence(self, player_id):
        """Check if enough time has passed since last violation save"""
        current_time = time.time()
        last_time = self.player_tracker.last_violation_time.get(player_id, 0)
        
        if (current_time - last_time) > 2:  # 2 second cooldown
            self.player_tracker.last_violation_time[player_id] = current_time
            return True
        return False
    
    def is_player_out(self, player_id):
        """Check if player is marked as out"""
        return self.player_out_status.get(player_id, False)
    
    def mark_player_out(self, player_id):
        """Mark player as out due to violations"""
        self.player_out_status[player_id] = True
        print(f"Player {player_id} is OUT - exceeded {self.MAX_VIOLATIONS} violations")
    
    def get_violation_count(self, player_id):
        """Get violation count for player"""
        return self.player_violation_counts.get(player_id, 0)
    
    def reset_player_violations(self, player_id):
        """Reset violations for a player"""
        self.player_violation_counts[player_id] = 0
        self.player_out_status[player_id] = False