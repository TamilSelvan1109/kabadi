
# Video file paths
VIDEO_PATHS = {
    'line_detection': 'assets/video2.mp4',
    'player_tracking': 'assets/video2.mp4',
    'skeleton_test': 'assets/video2.mp4'
}

# Alternative video sources (fallback options)
FALLBACK_VIDEOS = [
    'assets/video1.mp4',
    'assets/video2.mp4', 
    'assets/video3.mp4',
    'assets/back_angle_video1.mp4'
]

# Frame processing configuration
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_RATE = 30
BUFFER_SIZE = 1  # Minimal buffer for real-time processing

# Webcam settings
WEBCAM_ID = 0  # Default webcam (0 = first camera)

def get_video_path(component):
    return VIDEO_PATHS.get(component, VIDEO_PATHS['player_tracking'])

def get_line_detection_video():
    return VIDEO_PATHS['line_detection']

def get_player_tracking_video():
    return VIDEO_PATHS['player_tracking']

def get_skeleton_test_video():
    return VIDEO_PATHS['skeleton_test']

def get_fallback_videos():
    return FALLBACK_VIDEOS

def get_webcam_id():
    return WEBCAM_ID

def get_frame_config():
    return {
        'width': FRAME_WIDTH,
        'height': FRAME_HEIGHT,
        'fps': FRAME_RATE,
        'buffer_size': BUFFER_SIZE
    }