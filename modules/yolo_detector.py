import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')
        
    def detect_players(self, frame, conf_threshold=0.5):
        """Detect players using YOLO and return bounding boxes with tracking IDs"""
        results = self.model.track(
            frame, 
            persist=True, 
            classes=[0],  # Only detect persons
            conf=conf_threshold,
            iou=0.7,
            tracker="bytetrack.yaml"
        )
        
        detections = []
        
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            if boxes.id is not None:
                xyxy_boxes = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.int().cpu().tolist()
                confidences = boxes.conf.float().cpu().tolist()
                
                for bbox, yolo_id, conf in zip(xyxy_boxes, track_ids, confidences):
                    if conf >= conf_threshold:
                        x1, y1, x2, y2 = map(int, bbox)
                        
                        # Ensure bbox is within frame
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(frame.shape[1], x2)
                        y2 = min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:  # Valid bbox
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'center': (center_x, center_y),
                                'yolo_id': yolo_id,
                                'confidence': conf
                            })
        
        return detections