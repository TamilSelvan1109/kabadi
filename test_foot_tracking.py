import cv2
import mediapipe as mp
import json

# Simple foot tracking test
def test_foot_tracking():
    # Load MediaPipe
    try:
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode
        
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3
        )
        pose = PoseLandmarker.create_from_options(options)
        print("MediaPipe loaded successfully")
    except Exception as e:
        print(f"Error loading MediaPipe: {e}")
        return
    
    # Test with video
    cap = cv2.VideoCapture('assets/back_angle_video1.mp4')
    
    print("\nFOOT TRACKING TEST")
    print("This will show all detected foot landmarks")
    print("Press Q to quit, SPACE to pause")
    
    paused = False
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
        
        h, w, _ = frame.shape
        
        if not paused:
            # Process frame
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                               data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = pose.detect_for_video(mp_image, int(cv2.getTickCount()))
        
        # Draw all poses and foot landmarks
        if results.pose_landmarks:
            for i, pose_landmarks in enumerate(results.pose_landmarks):
                # Draw pose connections (skeleton)
                # Note: We'll draw key points manually since drawing utils changed
                
                # Head
                head = (int(pose_landmarks[0].x * w), int(pose_landmarks[0].y * h))
                cv2.circle(frame, head, 5, (255, 255, 0), -1)
                cv2.putText(frame, f"Person {i+1}", (head[0]-20, head[1]-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # Foot landmarks with confidence scores
                foot_landmarks = [
                    (27, "L_Ankle", (0, 255, 255)),
                    (28, "R_Ankle", (0, 255, 255)),
                    (29, "L_Heel", (255, 0, 0)),
                    (30, "R_Heel", (255, 0, 0)),
                    (31, "L_Toe", (0, 255, 0)),
                    (32, "R_Toe", (0, 255, 0))
                ]
                
                for landmark_idx, name, color in foot_landmarks:
                    if landmark_idx < len(pose_landmarks):
                        landmark = pose_landmarks[landmark_idx]
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        confidence = landmark.visibility
                        
                        # Draw landmark
                        cv2.circle(frame, (x, y), 6, color, -1)
                        cv2.putText(frame, f"{name}:{confidence:.2f}", 
                                   (x-30, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Instructions
        cv2.putText(frame, "Foot Tracking Test - Check landmark accuracy", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Yellow=Ankle, Red=Heel, Green=Toe", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Numbers show confidence (higher = better)", 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Foot Tracking Test', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_foot_tracking()