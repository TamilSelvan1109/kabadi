import json
import os
from datetime import datetime

class ViolationLogger:
    def __init__(self):
        self.violations_log = []
        self.log_file = "violations/violations_log.json"
        self.ensure_log_directory()
    
    def ensure_log_directory(self):
        """Ensure violations directory exists"""
        if not os.path.exists('violations'):
            os.makedirs('violations')
    
    def log_violation(self, player_id, foot_type, timestamp, frame_number):
        """Log a violation with details"""
        violation_entry = {
            "player_id": player_id,
            "foot_type": foot_type,
            "timestamp": timestamp,
            "frame_number": frame_number,
            "datetime": datetime.now().isoformat()
        }
        
        self.violations_log.append(violation_entry)
        
        # Save to file immediately
        self.save_log()
        
        print(f"VIOLATION LOGGED: P{player_id} {foot_type} foot at {timestamp}")
    
    def save_log(self):
        """Save violations log to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.violations_log, f, indent=2)
        except Exception as e:
            print(f"Error saving violations log: {e}")
    
    def get_player_violations(self, player_id):
        """Get all violations for a specific player"""
        return [v for v in self.violations_log if v['player_id'] == player_id]
    
    def get_violation_count(self, player_id=None):
        """Get total violation count or for specific player"""
        if player_id:
            return len(self.get_player_violations(player_id))
        return len(self.violations_log)
    
    def mark_player_out(self, player_id, reason="Multiple violations"):
        """Mark a player as out"""
        out_entry = {
            "player_id": player_id,
            "status": "OUT",
            "reason": reason,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "datetime": datetime.now().isoformat(),
            "total_violations": self.get_violation_count(player_id)
        }
        
        self.violations_log.append(out_entry)
        self.save_log()
        
        print(f"PLAYER OUT: P{player_id} - {reason} (Total violations: {out_entry['total_violations']})")
        return out_entry