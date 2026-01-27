import numpy as np
from config_manager import config

class FootDetector:
    def __init__(self, player_tracker):
        self.player_tracker = player_tracker
    
    def smooth_foot_position(self, player_id, foot_type, new_position):
        """Apply smoothing to foot positions"""
        if player_id not in self.player_tracker.foot_history:
            self.player_tracker.foot_history[player_id] = {'left': [], 'right': []}
        
        history = self.player_tracker.foot_history[player_id][foot_type]
        history.append(new_position)
        
        # Keep only recent positions
        if len(history) > config.FOOT_SMOOTHING_FRAMES:
            history.pop(0)
        
        if len(history) == 1:
            return new_position
        
        # Weighted average (more weight to recent)
        weights = [0.1, 0.15, 0.2, 0.25, 0.3]
        weights = weights[-len(history):]
        
        weighted_x = sum(pos[0] * w for pos, w in zip(history, weights)) / sum(weights)
        weighted_y = sum(pos[1] * w for pos, w in zip(history, weights)) / sum(weights)
        
        return (int(weighted_x), int(weighted_y))
    
    def get_foot_position(self, pose_landmarks, foot_type, player_id, w, h):
        """Get improved foot position using multiple landmarks"""
        # Check manual correction first
        if (player_id in self.player_tracker.selected_players and 
            foot_type in self.player_tracker.selected_players[player_id]['manual_feet']):
            return self.player_tracker.selected_players[player_id]['manual_feet'][foot_type]
        
        # Define landmark indices
        if foot_type == 'left':
            heel_idx, toe_idx, ankle_idx = 29, 31, 27
        else:
            heel_idx, toe_idx, ankle_idx = 30, 32, 28
        
        # Get landmark positions with confidence check
        heel_pos = self._get_landmark_pos(pose_landmarks, heel_idx, w, h)
        toe_pos = self._get_landmark_pos(pose_landmarks, toe_idx, w, h)
        ankle_pos = self._get_landmark_pos(pose_landmarks, ankle_idx, w, h)
        
        # Determine best foot position
        foot_position = None
        
        if heel_pos and toe_pos:
            # Average of heel and toe (most accurate)
            foot_position = ((heel_pos[0] + toe_pos[0]) // 2, max(heel_pos[1], toe_pos[1]))
        elif heel_pos:
            foot_position = heel_pos
        elif toe_pos:
            foot_position = toe_pos
        elif ankle_pos:
            # Estimate from ankle
            foot_position = (ankle_pos[0], ankle_pos[1] + 30)
        
        if foot_position:
            foot_position = self.smooth_foot_position(player_id, foot_type, foot_position)
        
        return foot_position
    
    def _get_landmark_pos(self, pose_landmarks, idx, w, h):
        """Get landmark position if confidence is sufficient"""
        if (idx < len(pose_landmarks) and 
            pose_landmarks[idx].visibility > config.FOOT_DETECTION_CONFIDENCE):
            return (int(pose_landmarks[idx].x * w), int(pose_landmarks[idx].y * h))
        return None