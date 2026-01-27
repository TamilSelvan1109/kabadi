import numpy as np

class PlayerTracker:
    def __init__(self):
        self.selected_players = {}
        self.next_player_id = 1
        self.last_violation_time = {}
        self.MAX_CLICK_DISTANCE = 50  # Maximum distance for click detection
        
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
                
                if distance < min_distance and distance < self.MAX_CLICK_DISTANCE:
                    min_distance = distance
                    closest_pose_idx = i
        
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
            'last_seen': frame_count
        }
        self.last_violation_time[self.next_player_id] = 0
        
        print(f"Assigned P{self.next_player_id}")
        self.next_player_id += 1
        return True
    
    def reset_all_players(self):
        """Reset all player selections"""
        self.selected_players.clear()
        self.last_violation_time.clear()
        self.next_player_id = 1
        print("All players reset")
    
    def get_player_count(self):
        return len(self.selected_players)
    
    def get_players(self):
        return self.selected_players