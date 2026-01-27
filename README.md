# Enhanced Kabaddi Foot Tracking System

## Problem Solved
The original MediaPipe `mp.solutions` API was deprecated. This solution provides:
1. **Updated MediaPipe API** - Compatible with MediaPipe 0.10+
2. **Manual Player Selection** - Click to assign unique IDs to players
3. **Enhanced Foot Tracking** - Multiple methods for accurate foot detection
4. **Manual Foot Correction** - Override MediaPipe when it's inaccurate

## How It Works

### 1. MediaPipe API Fix
- Updated from `mp.solutions.pose` to `mp.tasks.vision.PoseLandmarker`
- Fixed Unicode encoding issues for Windows
- Added proper model file download

### 2. Multi-Player Tracking
- **Manual Selection**: Pause video, press 'S', click on players to assign IDs
- **Unique IDs**: Each player gets P1, P2, P3, etc.
- **Green/Red Arrows**: Green arrows above heads, turn red when violating

### 3. Enhanced Foot Detection
The system uses multiple approaches for accurate foot tracking:

#### A. MediaPipe Landmarks
- **Heel** (landmarks 29, 30): Primary foot contact points
- **Toe** (landmarks 31, 32): Front of foot
- **Ankle** (landmarks 27, 28): Backup when heel/toe not visible

#### B. Improved Algorithm
```python
# Uses multiple landmarks for better accuracy
if heel_pos and toe_pos:
    foot_position = average(heel_pos, toe_pos)  # Most accurate
elif heel_pos:
    foot_position = heel_pos  # Good for ground contact
elif toe_pos:
    foot_position = toe_pos   # Alternative
else:
    foot_position = ankle_pos + offset  # Estimate from ankle
```

#### C. Smoothing & Filtering
- **Temporal Smoothing**: Averages last 5 positions to reduce jitter
- **Confidence Filtering**: Only uses landmarks with >30% confidence
- **Manual Override**: Click to manually set foot positions when MediaPipe fails

## Usage Instructions

### Basic Operation
1. **Start**: `python main.py`
2. **Pause**: Press `SPACE` to pause video
3. **Select Players**: Press `S` to enter selection mode, click on player heads
4. **Watch**: Green arrows show tracked players, red indicates violations

### Manual Foot Correction
When MediaPipe foot detection is inaccurate:
1. Press `F` to enter manual foot mode
2. Press `1-9` to select player ID
3. Press `L` or `R` to select left/right foot
4. Click on the frame where the foot actually is

### Controls
- `SPACE` - Pause/Resume
- `S` - Selection mode (click on players)
- `F` - Manual foot correction mode
- `1-9` - Select player for correction
- `L/R` - Select left/right foot
- `R` - Reset all players
- `Q` - Quit

## Testing Foot Accuracy

Run the test script to evaluate foot detection:
```bash
python test_foot_tracking.py
```

This shows:
- All detected foot landmarks
- Confidence scores for each landmark
- Color coding: Yellow=Ankle, Red=Heel, Green=Toe

## Configuration

Edit `foot_config.json` to fine-tune:
- Detection confidence thresholds
- Smoothing parameters
- Visualization settings
- Tracking parameters

## When to Use Manual vs Automatic

### Use MediaPipe Automatic When:
- Players are clearly visible
- Good lighting conditions
- Players wearing contrasting colors
- Minimal occlusion

### Use Manual Correction When:
- Players partially hidden behind others
- Poor lighting or shadows
- Similar colored clothing
- Fast movements causing detection errors
- Critical moments requiring 100% accuracy

## Files Created
- `main.py` - Enhanced tracking system
- `test_foot_tracking.py` - Accuracy testing tool
- `foot_config.json` - Configuration parameters
- `pose_landmarker_lite.task` - MediaPipe model file
- `requirements.txt` - Python dependencies

## Recommendation
For professional Kabaddi refereeing, use a **hybrid approach**:
1. Start with automatic MediaPipe detection
2. Monitor accuracy using the test script
3. Use manual correction for critical moments
4. Consider multiple camera angles for better coverage

The manual correction feature ensures 100% accuracy when needed, while automatic detection handles the majority of cases efficiently.