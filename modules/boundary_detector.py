import json
import numpy as np
import cv2

class BoundaryDetector:
    def __init__(self):
        self.boundary_points = []
        self.original_boundary_points = []
        
    def load_boundary_config(self, config_path='config.json'):
        """Load boundary points from config file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.original_boundary_points = config['boundary_points']
                print(f"✅ Loaded {len(self.original_boundary_points)} boundary points")
                return True
        except Exception as e:
            print(f"❌ Error loading boundary points: {e}")
            return False
    
    def scale_boundary_points(self, scale_factor):
        """Scale boundary points for display resolution"""
        self.boundary_points = []
        for point in self.original_boundary_points:
            scaled_x = int(point[0] * scale_factor)
            scaled_y = int(point[1] * scale_factor)
            self.boundary_points.append([scaled_x, scaled_y])
    
    def is_point_below_boundary(self, point, debug=False):
        """Check if point violates boundary with improved logic and debugging"""
        if len(self.boundary_points) < 2:
            if debug:
                print(f"⚠️ No boundary points available")
            return False
            
        x, y = point
        
        # Sort boundary points by x-coordinate for proper interpolation
        sorted_points = sorted(self.boundary_points, key=lambda p: p[0])
        
        # Find boundary y-coordinate at given x
        min_x = sorted_points[0][0]
        max_x = sorted_points[-1][0]
        
        boundary_y = None
        
        if x < min_x:
            # Extrapolate from first point
            boundary_y = sorted_points[0][1]
            if debug:
                print(f"🔍 Point ({x},{y}) left of boundary, using y={boundary_y}")
        elif x > max_x:
            # Extrapolate from last point
            boundary_y = sorted_points[-1][1]
            if debug:
                print(f"🔍 Point ({x},{y}) right of boundary, using y={boundary_y}")
        else:
            # Interpolate between boundary points
            for i in range(len(sorted_points) - 1):
                x1, y1 = sorted_points[i]
                x2, y2 = sorted_points[i + 1]
                
                if x1 <= x <= x2:
                    if x2 != x1:
                        boundary_y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
                    else:
                        boundary_y = y1
                    
                    if debug:
                        print(f"🔍 Interpolated between ({x1},{y1}) and ({x2},{y2}): boundary_y={boundary_y:.1f}")
                    break
        
        if boundary_y is None:
            if debug:
                print(f"❌ Could not determine boundary_y for point ({x},{y})")
            return False
        
        # Check violation: foot below boundary line (higher y value)
        violation = y > boundary_y
        
        if debug:
            print(f"🎯 Point ({x},{y}) vs boundary_y={boundary_y:.1f} → violation={violation}")
        
        return violation
    
    def draw_boundary(self, frame, show_debug=False):
        """Draw boundary line on frame with optional debug info"""
        if len(self.boundary_points) > 1:
            pts = np.array(self.boundary_points, np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 255), 2)
            
            # Draw boundary points
            for i, point in enumerate(self.boundary_points):
                cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
                if show_debug:
                    cv2.putText(frame, f"{i}", (point[0]+5, point[1]-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Boundary label
            cv2.putText(frame, "BOUNDARY", (self.boundary_points[0][0], self.boundary_points[0][1]-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if show_debug:
                # Show boundary point coordinates
                for i, (x, y) in enumerate(self.boundary_points):
                    cv2.putText(frame, f"({x},{y})", (x, y+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        return frame