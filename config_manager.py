import json
import os

class Config:
    def __init__(self):
        # Video settings
        self.VIDEO_SOURCE = 'assets/back_angle_video1.mp4'
        
        # Detection settings
        self.FOOT_DETECTION_CONFIDENCE = 0.3
        self.POSE_DETECTION_CONFIDENCE = 0.5
        self.TRACKING_CONFIDENCE = 0.5
        
        # Tracking settings
        self.MAX_CLICK_DISTANCE = 80
        self.FOOT_SMOOTHING_FRAMES = 5
        self.COOLDOWN_SECONDS = 2.0
        self.BOUNDARY_THRESHOLD = 10
        
        # Load boundary points
        self.BOUNDARY_POINTS = self.load_boundary_points()
    
    def load_boundary_points(self):
        if os.path.exists("config.json"):
            try:
                with open("config.json", "r") as f:
                    data = json.load(f)
                    points = data["boundary_points"]
                    return [tuple(p) for p in points]
            except Exception as e:
                print(f"ERROR reading config.json: {e}")
                exit()
        else:
            print("ERROR: config.json not found!")
            exit()

config = Config()