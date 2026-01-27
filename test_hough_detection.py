import cv2
import numpy as np
import os

def test_hough_detection(video_path):
    """Test Hough line detection on a video frame"""
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load first frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video frame")
        return
    
    print(f"Original frame size: {frame.shape}")
    
    # Resize for processing
    target_width = 1600
    orig_h, orig_w = frame.shape[:2]
    scale_factor = target_width / float(orig_w)
    new_dim = (target_width, int(orig_h * scale_factor))
    resized_frame = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
    
    print(f"Resized frame size: {resized_frame.shape}")
    
    # Enhanced Hough line detection
    def detect_hough_lines(image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"Gray image shape: {gray.shape}")
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Enhanced edge detection
            edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
            print(f"Edge pixels detected: {np.sum(edges > 0)}")
            
            # Show edges for debugging
            cv2.imshow('Edges', edges)
            
            # Improved Hough line detection with lower threshold
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
            
            if lines is not None:
                print(f"Lines detected: {len(lines)}")
                return lines[:10]
            else:
                print("No lines detected")
                return []
                
        except Exception as e:
            print(f"Hough line detection error: {e}")
            return []
    
    # Detect lines
    lines = detect_hough_lines(resized_frame)
    
    # Draw detected lines
    result_frame = resized_frame.copy()
    h, w = result_frame.shape[:2]
    
    for i, line in enumerate(lines):
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        
        if abs(b) > 0.001:
            x1, y1 = 0, int(y0 - (x0 * a / b))
            x2, y2 = w, int(y0 + ((w - x0) * a / b))
        else:
            x1, y1 = int(x0), 0
            x2, y2 = int(x0), h
        
        x1, y1 = max(0, min(w, x1)), max(0, min(h, y1))
        x2, y2 = max(0, min(w, x2)), max(0, min(h, y2))
        
        # Draw line
        cv2.line(result_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw center point
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.circle(result_frame, (center_x, center_y), 20, (255, 255, 0), 3)
        cv2.putText(result_frame, str(i+1), (center_x-8, center_y+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        print(f"Line {i+1}: ({x1},{y1}) to ({x2},{y2})")
    
    # Display results
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Hough Lines', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Original', 800, 600)
    cv2.resizeWindow('Hough Lines', 800, 600)
    
    cv2.imshow('Original', resized_frame)
    cv2.imshow('Hough Lines', result_frame)
    
    print("\nPress any key to close windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with available videos
    test_videos = [
        "assets/back_angle_video1.mp4",
        "assets/video1.mp4",
        "uploads/video4.mp4"
    ]
    
    for video_path in test_videos:
        if os.path.exists(video_path):
            print(f"\n=== Testing Hough detection on {video_path} ===")
            test_hough_detection(video_path)
            break
    else:
        print("No test videos found. Please ensure video files exist in assets/ or uploads/ folder.")