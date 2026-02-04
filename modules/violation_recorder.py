import cv2
import os
from datetime import datetime

class ViolationRecorder:
    def __init__(self):
        self.violation_records = {}
        self.active_violations = set()
        
        # Create output directories
        os.makedirs('violations/screenshots', exist_ok=True)
        os.makedirs('violations/videos', exist_ok=True)
    
    def handle_violations(self, frame, current_violations, frame_count):
        """Handle violation recording - one screenshot per violation"""
        # New violations
        new_violations = current_violations - self.active_violations
        
        # Ended violations
        ended_violations = self.active_violations - current_violations
        
        # Handle new violations
        for player_id in new_violations:
            print(f"🚨 NEW VIOLATION: Player {player_id} at frame {frame_count}")
            
            # Take screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"violations/screenshots/player_{player_id}_violation_{frame_count}_{timestamp}.jpg"
            cv2.imwrite(screenshot_path, frame)
            print(f"📸 Screenshot saved: {screenshot_path}")
            
            # Start video recording
            self.violation_records[player_id] = {
                'screenshot_taken': True,
                'frames': [frame.copy()],
                'start_frame': frame_count
            }
        
        # Continue recording for ongoing violations
        for player_id in current_violations:
            if player_id in self.violation_records:
                self.violation_records[player_id]['frames'].append(frame.copy())
                # Limit frames (max 5 seconds at 30fps)
                if len(self.violation_records[player_id]['frames']) > 150:
                    self.violation_records[player_id]['frames'].pop(0)
        
        # Handle ended violations
        for player_id in ended_violations:
            self.save_violation_video(player_id)
        
        # Update active violations
        self.active_violations = current_violations.copy()
    
    def save_violation_video(self, player_id):
        """Save violation video when violation ends"""
        if player_id not in self.violation_records:
            return
            
        frames = self.violation_records[player_id]['frames']
        start_frame = self.violation_records[player_id]['start_frame']
        
        if len(frames) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = f"violations/videos/player_{player_id}_violation_{start_frame}_{timestamp}.mp4"
            
            h, w, _ = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            duration = len(frames) / 30.0
            print(f"🎥 Video saved: {video_path} ({duration:.1f}s, {len(frames)} frames)")
        
        # Cleanup
        del self.violation_records[player_id]
        print(f"✅ Violation recording completed for Player {player_id}")
    
    def cleanup_player_violations(self, removed_player_ids):
        """Save videos for players that disappeared"""
        for player_id in removed_player_ids:
            if player_id in self.violation_records:
                print(f"⚠️ Player {player_id} disappeared during violation - saving video")
                self.save_violation_video(player_id)
            self.active_violations.discard(player_id)