# Kabaddi Foot Tracking System - Project Structure

## 🎯 Core System Files

### Main Application
- **`main.py`** - Main tracking application with skeleton visualization
- **`line_detection.py`** - Interactive boundary line setup with 3 methods

### Core Components
- **`player_tracker.py`** - Multi-player tracking and assignment
- **`foot_detector.py`** - Skeleton tracking with 33-point pose detection
- **`violation_detector.py`** - Boundary crossing detection and validation
- **`violation_logger.py`** - Evidence collection and logging
- **`visualizer.py`** - Visual rendering and UI components
- **`config_manager.py`** - Configuration management

### Configuration
- **`config.json`** - Boundary points and system settings
- **`requirements.txt`** - Python dependencies
- **`pose_landmarker_lite.task`** - MediaPipe model file

## 📁 Directory Structure

```
kabadi/
├── assets/                     # Video files
│   ├── back_angle_video.MP4
│   └── back_angle_video1.MP4
├── violations/                 # Evidence storage
│   ├── violations_log.json
│   └── [violation screenshots]
├── main.py                     # 🚀 START HERE
├── line_detection.py           # 🎯 Boundary setup
├── config.json                 # ⚙️ System config
└── [core components]
```

## 🚀 Quick Start

1. **Setup Boundary Line**:
   ```bash
   python line_detection.py
   ```
   - Choose detection method (Two Points/Multi Points/Hough Transform)
   - Define the kabaddi court boundary
   - Line saved to config.json

2. **Start Tracking**:
   ```bash
   python main.py
   ```
   - Full skeleton tracking with 33 landmarks
   - Real-time violation detection
   - Evidence collection

## 🎮 Controls

### Line Detection
- **1, 2, 3** - Select detection method
- **S** - Start detection
- **Mouse Click** - Place points or select lines
- **Right Click** - Finish multi-point drawing

### Main Tracking
- **SPACE** - Pause/Resume
- **S** - Selection mode (click on players)
- **R** - Reset all players
- **Q** - Quit

## 🔧 Features

### Advanced Skeleton Tracking
- 33-point MediaPipe pose detection
- Full body skeleton visualization
- Color-coded player status (Green/Red/Dark Red)
- Ground contact point analysis

### Smart Violation Detection
- Multiple foot landmark analysis (heels, toes, ankles)
- Confidence-based detection
- Temporal smoothing to reduce false positives
- Automatic evidence collection

### Professional UI
- Real-time status display
- Player assignment system
- Violation counter
- Evidence logging

## 📊 Technical Specifications

- **Framework**: OpenCV + MediaPipe
- **Detection**: 33-landmark pose estimation
- **Performance**: Real-time processing
- **Accuracy**: Multi-point ground contact validation
- **Evidence**: Automatic screenshot capture
- **Logging**: JSON-based violation records

## 🎯 Project Highlights

This system demonstrates:
- **Computer Vision**: Advanced pose estimation
- **Real-time Processing**: Live video analysis
- **Sports Technology**: Practical kabaddi application
- **User Interface**: Interactive boundary setup
- **Data Management**: Evidence collection and logging
- **Performance Optimization**: Efficient skeleton tracking

Perfect for final year computer science projects showcasing practical AI applications in sports technology.