# MANUAL IMPLEMENTATION GUIDE

## How Player Selection Works

### Step 1: Enable Selection Mode
- Press `S` to toggle selection mode ON
- You'll see "Selection: ON" in the status area
- White circles appear on unassigned player heads with "Click" text

### Step 2: Click on Player Heads
- Click directly on a player's head (white circle)
- System finds closest pose within 80 pixels
- Assigns unique ID (P1, P2, P3, etc.)
- Green arrow appears above selected player

### Step 3: Verify Selection
- Selected players show green arrows with "P1", "P2" labels
- Status shows "Players: X" count
- Each player gets unique tracking

## How Foot Correction Works

### Step 1: Select Player for Correction
- Press `F` to enable manual foot mode
- Press number key `1-9` to select player ID
- Status shows "Correcting PX select L/R"

### Step 2: Select Foot Type
- Press `L` for left foot correction
- Press `R` for right foot correction
- Status shows "Correcting PX left/right"

### Step 3: Click Foot Position
- Click exactly where the foot should be
- System saves manual position for that player's foot
- Manual position overrides MediaPipe detection

## Violation System

### Boundary Crossing Detection
- System checks if foot Y-position > boundary line Y + 10px threshold
- Uses geometric line intersection calculation
- Checks both left and right feet independently

### Violation Logging
- Each violation logged with timestamp, player ID, foot type
- Saved to `violations/violations_log.json`
- Screenshot saved to `violations/LineOut_PX_FootType_Timestamp.jpg`

### Player Out System
- Player marked OUT after 3 violations
- Arrow turns dark red, shows "PX-OUT" label
- No further violations counted for OUT players
- Details logged in violations file

## Color Coding System

### Arrow Colors
- **Green**: Normal player, no violations
- **Red**: Player currently violating boundary
- **Dark Red**: Player is OUT (3+ violations)

### Foot Colors
- **Blue**: Normal foot position
- **Red**: Foot currently crossing boundary
- **Red Circle**: Violation highlight (25px radius)

## Troubleshooting

### Player Selection Not Working
1. Ensure selection mode is ON (press `S`)
2. Click directly on player's head (nose area)
3. Check distance - must be within 80 pixels
4. Look for debug output in console

### Foot Correction Not Working
1. Enable manual foot mode (press `F`)
2. Select player number (press `1-9`)
3. Select foot type (press `L` or `R`)
4. Click on exact foot position
5. Check console for confirmation message

### Violations Not Detected
1. Check boundary line is visible (yellow line)
2. Verify foot positions are being detected
3. Check threshold settings in config_manager.py
4. Look for foot circles (blue/red dots)

## File Structure
- `main.py` - Main application coordinator
- `config_manager.py` - Settings and boundary points
- `player_tracker.py` - Player selection and tracking
- `foot_detector.py` - Foot position detection
- `violation_detector.py` - Boundary crossing detection
- `violation_logger.py` - Violation logging and player out
- `visualizer.py` - Drawing arrows, feet, UI elements

## Key Settings (config_manager.py)
- `MAX_CLICK_DISTANCE = 80` - Click detection radius
- `FOOT_DETECTION_CONFIDENCE = 0.3` - MediaPipe confidence
- `BOUNDARY_THRESHOLD = 10` - Violation detection threshold
- `COOLDOWN_SECONDS = 2.0` - Time between violation saves

## Log Files
- `violations/violations_log.json` - All violations and player status
- `violations/LineOut_*.jpg` - Violation screenshots
- Console output - Real-time debug information