import numpy as np
import cv2

class KalmanTracker:
    def __init__(self, initial_position):
        """Initialize Kalman filter for player tracking"""
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 states (x, y, vx, vy), 2 measurements (x, y)
        
        # State transition matrix (constant velocity model)
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], dtype=np.float32)
        
        # Process noise covariance
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise covariance
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Error covariance
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)
        
        # Initial state [x, y, vx, vy]
        x, y = initial_position
        self.kalman.statePre = np.array([x, y, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([x, y, 0, 0], dtype=np.float32)
        
        self.last_prediction = initial_position
        
    def predict(self):
        """Predict next position"""
        prediction = self.kalman.predict()
        self.last_prediction = (int(prediction[0]), int(prediction[1]))
        return self.last_prediction
    
    def update(self, measurement):
        """Update with new measurement"""
        measurement = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kalman.correct(measurement)
        
    def get_predicted_position(self):
        """Get last predicted position"""
        return self.last_prediction