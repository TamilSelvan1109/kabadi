import numpy as np
from config_manager import config

class PlayerTracker:
    def __init__(self):
        self.selected_players = {}
        self.next_player_id = 1
        self.foot_history = {}
        self.last_violation_time = {}
        
    def find_closest_pose(self, click_x, click_y, poses, frame_width, frame_height):
        """Find the closest pose to click position"""
        min_distance = float('inf')
        closest_pose_idx = -1
        
        for i, pose_landmarks in enumerate(poses):
            if pose_landmarks and len(pose_landmarks) > 0:
                # Use nose landmark (index 0) for head position
                head_x = int(pose_landmarks[0].x * frame_width)
                head_y = int(pose_landmarks[0].y * frame_height)
                
                distance = np.sqrt((click_x - head_x)**2 + (click_y - head_y)**2)
                print(f"Pose {i}: head at ({head_x}, {head_y}), click at ({click_x}, {click_y}), distance: {distance:.1f}")
                
                if distance < min_distance and distance < config.MAX_CLICK_DISTANCE:
                    min_distance = distance
                    closest_pose_idx = i
        
        print(f"Closest pose: {closest_pose_idx} (distance: {min_distance:.1f})")
        return closest_pose_idx
    
    def assign_player(self, pose_idx, frame_count):
        """Assign a new player ID to a pose"""
        # Check if pose already assigned
        for pid, pdata in self.selected_players.items():
            if pdata['pose_index'] == pose_idx:
                print(f"Pose already assigned to P{pid}")
                return False
        
        # Assign new player
        self.selected_players[self.next_player_id] = {
            'pose_index': pose_idx,
            'last_seen': frame_count,
            'manual_feet': {}
        }
        self.last_violation_time[self.next_player_id] = 0
        self.foot_history[self.next_player_id] = {'left': [], 'right': []}
        
        print(f"Assigned P{self.next_player_id}")
        self.next_player_id += 1
        return True
    
    def set_manual_foot(self, player_id, foot_type, position):
        """Set manual foot position for a player"""
        if player_id in self.selected_players:
            self.selected_players[player_id]['manual_feet'][foot_type] = position
            print(f"Set P{player_id} {foot_type} foot to {position}")
            return True
        return False
    
    def reset_all_players(self):
        """Reset all player selections"""
        self.selected_players.clear()
        self.foot_history.clear()
        self.last_violation_time.clear()
        self.next_player_id = 1
        print("All players reset")
    
    def get_player_count(self):
        return len(self.selected_players)
    
    def get_players(self):
        return self.selected_players