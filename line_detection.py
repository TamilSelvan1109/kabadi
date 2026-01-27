import cv2
import json
import numpy as np
import webbrowser
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse
import os

video_path = 'assets/back_angle_video1.mp4'
points = []
scale_factor = 1.0
mode = "IDLE"
detection_method = "TWO_POINTS"
mouse_x, mouse_y = 0, 0
hough_lines = []
selected_line_idx = -1

def detect_hough_lines(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Enhanced edge detection
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        # Improved Hough line detection with lower threshold
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        return lines[:10] if lines is not None else []
    except Exception as e:
        print(f"Hough line detection error: {e}")
        return []

def is_inside_button(x, y, btn_x1, btn_y1, btn_x2, btn_y2):
    return btn_x1 < x < btn_x2 and btn_y1 < y < btn_y2

def draw_ui(image, curr_x, curr_y):
    h, w, _ = image.shape
    
    # Enhanced UI with better visibility
    # Background for buttons
    cv2.rectangle(image, (10, 10), (350, 80), (0, 0, 0), -1)
    cv2.rectangle(image, (10, 10), (350, 80), (255, 255, 255), 2)
    
    # SAVE button - larger and more visible
    cv2.rectangle(image, (20, 20), (150, 70), (0, 255, 0), -1)
    cv2.rectangle(image, (20, 20), (150, 70), (0, 0, 0), 2)
    cv2.putText(image, "SAVE", (55, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)
    
    # RESET button - larger and more visible
    cv2.rectangle(image, (170, 20), (300, 70), (0, 0, 255), -1)
    cv2.rectangle(image, (170, 20), (300, 70), (255, 255, 255), 2)
    cv2.putText(image, "RESET", (195, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
    
    # Method-specific instructions with background
    if detection_method == "TWO_POINTS":
        if len(points) == 0:
            msg = "Click first point"
        elif len(points) == 1:
            msg = "Click second point"
        else:
            msg = "Line complete. Click SAVE."
    elif detection_method == "MULTIPOINTS":
        if mode == "DRAWING":
            msg = "Left click: add points | Right click: finish"
        elif mode == "DONE":
            msg = "Line complete. Click SAVE."
        else:
            msg = "Click to start drawing"
    else:  # HOUGH
        if mode == "HOUGH_SELECT":
            msg = "Click yellow circles to select line"
        else:
            msg = "Line selected. Click SAVE."
    
    # Message background
    cv2.rectangle(image, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(image, (10, h - 50), (w - 10, h - 10), (255, 255, 255), 2)
    cv2.putText(image, msg, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global points, mode, img_display, img_clean, mouse_x, mouse_y, selected_line_idx, hough_lines
    
    mouse_x, mouse_y = x, y
    h, w, _ = img_display.shape

    if event == cv2.EVENT_LBUTTONDOWN:
        # SAVE button
        if is_inside_button(x, y, 20, 20, 150, 70):
            if (detection_method in ["TWO_POINTS", "MULTIPOINTS"] and len(points) >= 2) or \
               (detection_method == "HOUGH" and selected_line_idx >= 0):
                save_line()
            return
        
        # RESET button
        if is_inside_button(x, y, 170, 20, 300, 70):
            reset_detection()
            return
        
        # Line drawing logic
        if detection_method == "TWO_POINTS":
            points.append((x, y))
            cv2.circle(img_display, (x, y), 4, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(img_display, points[0], points[1], (0, 255, 0), 2)
                mode = "DONE"
        
        elif detection_method == "MULTIPOINTS":
            if mode != "DONE":
                mode = "DRAWING"
                points.append((x, y))
                cv2.circle(img_display, (x, y), 4, (0, 0, 255), -1)
                if len(points) > 1:
                    cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)
        
        elif detection_method == "HOUGH" and mode == "HOUGH_SELECT":
            select_hough_line(x, y)
        
        draw_ui(img_display, x, y)

    elif event == cv2.EVENT_RBUTTONDOWN and detection_method == "MULTIPOINTS":
        if len(points) > 1:
            mode = "DONE"
            redraw_multipoints()

def select_hough_line(x, y):
    global selected_line_idx, mode
    h, w = img_display.shape[:2]
    
    for i, line in enumerate(hough_lines[:5]):
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
        center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
        
        if ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5 <= 20:
            selected_line_idx = i
            mode = "DONE"
            redraw_hough_lines()
            break

def redraw_multipoints():
    global img_display
    img_display = img_clean.copy()
    for i in range(len(points) - 1):
        cv2.line(img_display, points[i], points[i+1], (0, 0, 255), 2)
        cv2.circle(img_display, points[i], 4, (0, 255, 0), -1)
    cv2.circle(img_display, points[-1], 4, (0, 255, 0), -1)
    draw_ui(img_display, mouse_x, mouse_y)

def redraw_hough_lines():
    global img_display
    img_display = img_clean.copy()
    h, w = img_display.shape[:2]
    
    for i, line in enumerate(hough_lines[:5]):
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
        
        if i == selected_line_idx:
            cv2.line(img_display, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.circle(img_display, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 20, (0, 255, 0), -1)
        else:
            cv2.line(img_display, (x1, y1), (x2, y2), (100, 100, 100), 1)
            cv2.circle(img_display, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 20, (255, 255, 0), 3)
        
        cv2.putText(img_display, str(i+1), (int((x1 + x2) / 2)-8, int((y1 + y2) / 2)+8), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    draw_ui(img_display, mouse_x, mouse_y)

def reset_detection():
    global points, mode, img_display, selected_line_idx
    points = []
    selected_line_idx = -1
    mode = "IDLE"
    img_display = img_clean.copy()
    draw_ui(img_display, mouse_x, mouse_y)

def save_line():
    global points, selected_line_idx, hough_lines
    
    if detection_method == "HOUGH" and selected_line_idx >= 0:
        line = hough_lines[selected_line_idx]
        rho, theta = line[0]
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1 = int((x0 + 500 * (-b)) / scale_factor)
        y1 = int((y0 + 500 * (a)) / scale_factor)
        x2 = int((x0 - 500 * (-b)) / scale_factor)
        y2 = int((y0 - 500 * (a)) / scale_factor)
        real_points = [(x1, y1), (x2, y2)]
    else:
        real_points = [(int(p[0] / scale_factor), int(p[1] / scale_factor)) for p in points]
    
    data = {"boundary_points": real_points, "method": detection_method}
    with open("config.json", "w") as f:
        json.dump(data, f)
    
    print(f"\n✅ SUCCESS: {detection_method} line saved!")
    print("\n📋 NEXT STEPS:")
    print("1. Run 'python main.py' to start tracking")
    print("2. Boundary line will be loaded automatically")
    cv2.destroyAllWindows()
    exit()

def start_detection(method):
    global detection_method, mode, img_display, img_clean, hough_lines, selected_line_idx
    
    detection_method = method
    mode = "DRAWING" if method != "HOUGH" else "HOUGH_SELECT"
    
    # Load video frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        return False
    
    # Resize frame - much bigger display
    target_width = 1600  # Increased from 1000
    orig_h, orig_w = frame.shape[:2]
    global scale_factor
    scale_factor = target_width / float(orig_w)
    new_dim = (target_width, int(orig_h * scale_factor))
    
    img_clean = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
    img_display = img_clean.copy()
    
    if method == "HOUGH":
        hough_lines = detect_hough_lines(img_clean)
        h, w = img_display.shape[:2]
        for i, line in enumerate(hough_lines[:5]):
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
            
            cv2.line(img_display, (x1, y1), (x2, y2), (0, 0, 255), 2)
            center_x, center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(img_display, (center_x, center_y), 20, (255, 255, 0), 3)
            cv2.putText(img_display, str(i+1), (center_x-8, center_y+8), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.namedWindow('Line Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Line Detection', target_width, int(orig_h * scale_factor))
    cv2.setMouseCallback('Line Detection', mouse_callback)
    draw_ui(img_display, 0, 0)
    
    while True:
        cv2.imshow('Line Detection', img_display)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return True

class RequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/start_detection':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            method = data.get('method', 'TWO_POINTS')
            
            # Start detection in separate thread
            threading.Thread(target=start_detection, args=(method,), daemon=True).start()
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True}).encode())
        else:
            self.send_response(404)
            self.end_headers()

def start_web_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    server = HTTPServer(('localhost', 8000), RequestHandler)
    print("Web server started at http://localhost:8000")
    server.serve_forever()

if __name__ == "__main__":
    # Start web server in background
    threading.Thread(target=start_web_server, daemon=True).start()
    
    # Open dashboard in browser
    webbrowser.open('http://localhost:8000/dashboard.html')
    
    print("\n🌐 Dashboard opened in browser")
    print("Select detection method and click Start")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nShutting down...")