import math

class PlayerIDManager:
    def __init__(self):
        self.stable_players = {}  # stable_id: {position, last_seen, yolo_id, bbox}
        self.next_stable_id = 1
        self.max_distance = 150
        self.max_frames_missing = 60
        
    def get_stable_id(self, center_pos, bbox, yolo_id, frame_count):
        """Assign stable ID with improved tracking logic"""
        x, y = center_pos
        x1, y1, x2, y2 = bbox
        
        # Check if YOLO ID already mapped
        for stable_id, player_data in self.stable_players.items():
            if player_data.get('yolo_id') == yolo_id:
                self.stable_players[stable_id].update({
                    'position': center_pos,
                    'last_seen': frame_count,
                    'bbox': bbox
                })
                return stable_id
        
        # Try position matching
        closest_id = None
        min_distance = float('inf')
        
        for stable_id, player_data in self.stable_players.items():
            px, py = player_data['position']
            distance = math.sqrt((x - px)**2 + (y - py)**2)
            
            if distance < self.max_distance and distance < min_distance:
                min_distance = distance
                closest_id = stable_id
        
        if closest_id is not None:
            self.stable_players[closest_id].update({
                'position': center_pos,
                'last_seen': frame_count,
                'yolo_id': yolo_id,
                'bbox': bbox
            })
            return closest_id
        
        # Check overlap
        for stable_id, player_data in self.stable_players.items():
            px1, py1, px2, py2 = player_data['bbox']
            
            overlap_x = max(0, min(x2, px2) - max(x1, px1))
            overlap_y = max(0, min(y2, py2) - max(y1, py1))
            overlap_area = overlap_x * overlap_y
            
            current_area = (x2 - x1) * (y2 - y1)
            existing_area = (px2 - px1) * (py2 - py1)
            
            if overlap_area > 0.3 * min(current_area, existing_area):
                self.stable_players[stable_id].update({
                    'position': center_pos,
                    'last_seen': frame_count,
                    'yolo_id': yolo_id,
                    'bbox': bbox
                })
                return stable_id
        
        # Create new player
        new_stable_id = self.next_stable_id
        self.stable_players[new_stable_id] = {
            'position': center_pos,
            'last_seen': frame_count,
            'yolo_id': yolo_id,
            'bbox': bbox
        }
        self.next_stable_id += 1
        return new_stable_id
    
    def cleanup_old_players(self, frame_count):
        """Remove players not seen for too long"""
        to_remove = []
        for stable_id, player_data in self.stable_players.items():
            if frame_count - player_data['last_seen'] > self.max_frames_missing:
                to_remove.append(stable_id)
        
        for stable_id in to_remove:
            del self.stable_players[stable_id]
        
        return to_remove