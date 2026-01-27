# Enhanced Multi-Player Skeleton Tracking System
## Final Year B.Tech Project - Computer Vision & Sports Analytics

### Project Overview
This project implements an advanced computer vision system for tracking multiple players in Kabaddi matches using skeleton detection. The system automatically detects boundary violations, highlights violating players with red skeletons, and records violation clips for analysis.

### Key Features

#### 1. Advanced Multi-Player Detection
- **Simultaneous Tracking**: Tracks up to 8 players simultaneously
- **Robust Skeleton Detection**: Uses MediaPipe's latest pose estimation model
- **Backside View Optimization**: Specifically tuned for rear-view player tracking
- **Real-time Processing**: Processes video at 30 FPS with minimal latency

#### 2. Enhanced Skeleton Visualization
- **Color-Coded Status**: 
  - Green: Normal players
  - Red: Players violating boundary
  - Dark Red: Players who are out (3+ violations)
  - White: Undetected/unassigned players
- **Comprehensive Skeleton**: 33-point skeleton including hands and feet
- **Ground Contact Highlighting**: Yellow circles on feet for boundary detection

#### 3. Intelligent Violation Detection
- **Multi-Point Analysis**: Checks heels, toes, ankles, and knees for ground contact
- **Confidence-Based Filtering**: Uses landmark confidence scores for accuracy
- **Temporal Smoothing**: Reduces false positives from detection noise
- **Automatic Clip Recording**: Records 3-second violation clips automatically

#### 4. Professional Data Management
- **Violation Clips**: Automatically saved as MP4 files
- **Screenshots**: High-quality violation evidence images
- **Structured Logging**: Comprehensive event logging system
- **JSON Configuration**: Easy parameter tuning without code changes

### Technical Architecture

#### Core Components

1. **EnhancedSkeletonTracker**: Main tracking engine
2. **PlayerData**: Individual player state management
3. **ViolationRecorder**: Automatic clip recording system
4. **BoundaryDetector**: Geometric violation analysis
5. **SkeletonVisualizer**: Advanced rendering system

#### MediaPipe Integration
```python
# Optimized for multi-person detection
PoseLandmarkerOptions(
    num_poses=8,                    # Track up to 8 players
    min_pose_detection_confidence=0.4,
    min_pose_presence_confidence=0.4,
    min_tracking_confidence=0.4
)
```

#### Skeleton Landmark Mapping
- **33 Key Points**: Complete body mapping including fingers and toes
- **Ground Contact Points**: Heels, toes, ankles prioritized for boundary detection
- **Confidence Filtering**: Only uses landmarks with >30% confidence
- **Temporal Consistency**: Maintains tracking across frames

### Installation & Setup

#### Prerequisites
```bash
pip install opencv-python>=4.5.0
pip install mediapipe>=0.10.0
pip install numpy>=1.21.0
```

#### Model Download
The system automatically downloads the MediaPipe pose model:
- **File**: `pose_landmarker_lite.task`
- **Size**: ~12MB
- **Accuracy**: Optimized for real-time performance

#### Directory Structure
```
Project/
├── enhanced_skeleton_tracker.py    # Main application
├── enhanced_config.json           # Configuration file
├── assets/                        # Video files
├── violations/                    # Violation screenshots
├── violation_clips/              # Violation video clips
├── logs/                         # System logs
└── screenshots/                  # Manual screenshots
```

### Usage Instructions

#### Basic Operation
1. **Start Application**: `python enhanced_skeleton_tracker.py`
2. **Player Selection**: Press 'S' to enter selection mode, click on player heads
3. **Monitor Violations**: System automatically detects and records violations
4. **Review Clips**: Check `violation_clips/` folder for recorded violations

#### Controls
- **SPACE**: Pause/Resume video
- **S**: Toggle selection mode (click to assign players)
- **R**: Toggle violation recording on/off
- **C**: Clear all assigned players
- **Q**: Quit application

#### Player Assignment Process
1. Pause video at clear frame showing players
2. Press 'S' to activate selection mode
3. Click on each player's head to assign unique ID (P1, P2, etc.)
4. Resume video to start tracking

### Violation Detection Algorithm

#### Ground Contact Analysis
```python
def check_boundary_violation(ground_points):
    # Priority order for ground contact
    foot_landmarks = [
        'left_heel', 'right_heel',      # Primary contact points
        'left_foot_index', 'right_foot_index',  # Toe contact
        'left_ankle', 'right_ankle',    # Backup points
        'left_knee', 'right_knee'       # Emergency fallback
    ]
```

#### Boundary Mathematics
- **Line Equation**: Uses parametric line equations for precise boundary detection
- **Threshold**: 10-pixel buffer zone to account for detection uncertainty
- **Multi-Segment**: Supports complex boundary shapes with multiple segments

#### Violation Workflow
1. **Detection**: Continuous monitoring of all ground contact points
2. **Validation**: 2-second cooldown prevents duplicate violations
3. **Recording**: Automatic 3-second clip recording with pre-buffer
4. **Counting**: Tracks total violations per player
5. **Elimination**: Players with 3+ violations marked as "OUT"

### Output Files

#### Violation Clips
- **Format**: MP4 (H.264 codec)
- **Duration**: 3 seconds (configurable)
- **Naming**: `P{ID}_violation_{timestamp}.mp4`
- **Quality**: Original video resolution maintained

#### Screenshots
- **Format**: JPEG (high quality)
- **Timing**: Captured at exact violation moment
- **Naming**: `P{ID}_violation_{timestamp}.jpg`
- **Usage**: Quick reference and evidence

#### Log Files
- **Format**: JSON structured logs
- **Content**: Timestamps, player IDs, violation types, coordinates
- **Usage**: Statistical analysis and system debugging

### Performance Optimization

#### Real-Time Processing
- **Frame Rate**: Maintains 30 FPS on modern hardware
- **Memory Usage**: Efficient circular buffers for video clips
- **CPU Optimization**: Multi-threaded violation recording
- **GPU Acceleration**: MediaPipe uses GPU when available

#### Accuracy Improvements
- **Confidence Thresholding**: Filters unreliable detections
- **Temporal Smoothing**: Reduces jitter in skeleton tracking
- **Multi-Point Validation**: Uses multiple body parts for violation detection
- **Manual Override**: Allows manual player assignment for difficult cases

### Configuration Options

#### Tracking Parameters
```json
{
    "max_players": 8,
    "min_pose_confidence": 0.4,
    "tracking_history_size": 10,
    "violation_cooldown_seconds": 2.0
}
```

#### Visualization Settings
```json
{
    "normal_color": [0, 255, 0],
    "violation_color": [0, 0, 255],
    "skeleton_thickness": 2,
    "foot_radius": 8
}
```

### Academic Contributions

#### Computer Vision Techniques
1. **Multi-Person Pose Estimation**: Advanced MediaPipe integration
2. **Temporal Tracking**: Consistent player identification across frames
3. **Geometric Analysis**: Precise boundary violation detection
4. **Real-Time Processing**: Optimized for live sports analysis

#### Software Engineering Practices
1. **Modular Architecture**: Clean separation of concerns
2. **Configuration Management**: JSON-based parameter control
3. **Error Handling**: Robust exception management
4. **Documentation**: Comprehensive code documentation

#### Sports Analytics Applications
1. **Automated Officiating**: Reduces human error in violation detection
2. **Performance Analysis**: Detailed violation statistics per player
3. **Video Evidence**: Automatic clip generation for review
4. **Real-Time Feedback**: Immediate violation notifications

### Future Enhancements

#### Planned Features
1. **Multi-Camera Support**: Combine multiple viewing angles
2. **Machine Learning**: Train custom models for Kabaddi-specific poses
3. **3D Tracking**: Depth estimation for more accurate positioning
4. **Live Streaming**: Real-time analysis of live matches
5. **Mobile App**: Companion app for coaches and referees

#### Research Opportunities
1. **Action Recognition**: Detect specific Kabaddi moves and techniques
2. **Player Identification**: Automatic jersey number recognition
3. **Strategy Analysis**: Team formation and movement patterns
4. **Injury Prevention**: Biomechanical analysis for safety

### Conclusion

This Enhanced Multi-Player Skeleton Tracking System represents a significant advancement in sports technology, combining cutting-edge computer vision with practical sports officiating needs. The system's ability to automatically detect violations, record evidence, and provide real-time feedback makes it suitable for professional Kabaddi matches while serving as an excellent demonstration of modern AI capabilities in sports analytics.

The project showcases advanced programming skills, computer vision expertise, and practical problem-solving abilities expected in a final year B.Tech project, with clear applications in the growing field of sports technology.

### Contact & Support
For technical support, feature requests, or academic collaboration, please refer to the project documentation or contact the development team.

---
**Project Status**: Production Ready  
**Last Updated**: January 2025  
**Version**: 2.0  
**License**: Academic Use