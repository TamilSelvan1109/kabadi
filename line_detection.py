import cv2
import json
import numpy as np
import os
import tkinter as tk
from tkinter import ttk
from video_config import get_line_detection_video

video_path = get_line_detection_video()  # Centralized video configuration
points = []
scale_factor = 1.0
mode = "IDLE"
detection_method = "TWO_POINTS"
mouse_x, mouse_y = 0, 0
hough_lines = []
selected_line_idx = -1
back_pressed = False

def detect_hough_lines(image):
    try:
        # Focus on bottom 60% of image for better line detection
        h, w = image.shape[:2]
        bottom_region = image[int(h*0.4):, :]
        
        gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
        # Enhanced preprocessing
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120, apertureSize=3)
        
        # Display preprocessing steps
        cv2.imshow('Grayscale', cv2.resize(gray, (400, 300)))
        cv2.imshow('Canny Edges', cv2.resize(edges, (400, 300)))
        
        # More precise Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            adjusted_lines = []
            for line in lines[:8]:  # Limit to 8 lines
                rho, theta = line[0]
                # Adjust rho for bottom region offset
                adjusted_rho = rho + (int(h*0.4)) * np.sin(theta)
                adjusted_lines.append([[adjusted_rho, theta]])
            return adjusted_lines
        return []
    except Exception as e:
        print(f"Hough line detection error: {e}")
        return []

def is_inside_button(x, y, btn_x1, btn_y1, btn_x2, btn_y2):
    return btn_x1 < x < btn_x2 and btn_y1 < y < btn_y2

def draw_ui(image, curr_x, curr_y):
    h, w, _ = image.shape
    
    # Background for buttons
    cv2.rectangle(image, (10, 10), (550, 80), (0, 0, 0), -1)
    cv2.rectangle(image, (10, 10), (550, 80), (255, 255, 255), 2)
    
    # BACK button
    cv2.rectangle(image, (20, 20), (120, 70), (128, 128, 128), -1)
    cv2.rectangle(image, (20, 20), (120, 70), (255, 255, 255), 2)
    cv2.putText(image, "BACK", (45, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # SAVE button
    cv2.rectangle(image, (140, 20), (270, 70), (0, 255, 0), -1)
    cv2.rectangle(image, (140, 20), (270, 70), (0, 0, 0), 2)
    cv2.putText(image, "SAVE", (175, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
    
    # RESET button
    cv2.rectangle(image, (290, 20), (420, 70), (0, 0, 255), -1)
    cv2.rectangle(image, (290, 20), (420, 70), (255, 255, 255), 2)
    cv2.putText(image, "RESET", (315, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    
    # RESELECT button for Hough method
    if detection_method == "HOUGH" and selected_line_idx >= 0:
        cv2.rectangle(image, (440, 20), (540, 70), (255, 165, 0), -1)
        cv2.rectangle(image, (440, 20), (540, 70), (255, 255, 255), 2)
        cv2.putText(image, "RESEL", (460, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    # Method-specific instructions
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
            msg = "Press number keys (1-8) to select line or click circles"
        else:
            msg = f"Line {selected_line_idx+1} selected. Click SAVE or RESELECT."
    
    # Message background
    cv2.rectangle(image, (10, h - 50), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.rectangle(image, (10, h - 50), (w - 10, h - 10), (255, 255, 255), 2)
    cv2.putText(image, msg, (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

def mouse_callback(event, x, y, flags, param):
    global points, mode, img_display, img_clean, mouse_x, mouse_y, selected_line_idx, hough_lines, back_pressed
    
    mouse_x, mouse_y = x, y
    h, w, _ = img_display.shape

    if event == cv2.EVENT_LBUTTONDOWN:
        # BACK button
        if is_inside_button(x, y, 20, 20, 120, 70):
            back_pressed = True
            return
        
        # SAVE button
        if is_inside_button(x, y, 140, 20, 270, 70):
            if (detection_method in ["TWO_POINTS", "MULTIPOINTS"] and len(points) >= 2) or \
               (detection_method == "HOUGH" and selected_line_idx >= 0):
                save_line()
            return
        
        # RESET button
        if is_inside_button(x, y, 290, 20, 420, 70):
            reset_detection()
            return
        
        # RESELECT button for Hough
        if detection_method == "HOUGH" and selected_line_idx >= 0 and is_inside_button(x, y, 440, 20, 540, 70):
            selected_line_idx = -1
            mode = "HOUGH_SELECT"
            redraw_hough_lines()
            return
        
        # Line drawing logic
        if detection_method == "TWO_POINTS":
            if len(points) < 2:  # Only allow 2 points maximum
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
    
    for i, line in enumerate(hough_lines[:8]):  # Show up to 8 lines
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
            cv2.circle(img_display, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 25, (0, 255, 0), -1)
        else:
            cv2.line(img_display, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.circle(img_display, (int((x1 + x2) / 2), int((y1 + y2) / 2)), 25, (0, 255, 255), 3)
        
        # Display number on circle
        cv2.putText(img_display, str(i+1), (int((x1 + x2) / 2)-10, int((y1 + y2) / 2)+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
    
    draw_ui(img_display, mouse_x, mouse_y)

def reset_detection():
    global points, mode, img_display, selected_line_idx
    points = []
    selected_line_idx = -1
    mode = "IDLE" if detection_method != "HOUGH" else "HOUGH_SELECT"
    img_display = img_clean.copy()
    if detection_method == "HOUGH":
        redraw_hough_lines()
    else:
        draw_ui(img_display, mouse_x, mouse_y)

def save_line():
    global points, selected_line_idx, hough_lines, scale_factor
    
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
        # Convert display coordinates back to original video coordinates
        real_points = [(int(p[0] / scale_factor), int(p[1] / scale_factor)) for p in points]
    
    data = {"boundary_points": real_points, "method": detection_method}
    with open("config.json", "w") as f:
        json.dump(data, f)
    
    print(f"✅ SUCCESS: {detection_method} line saved!")
    print(f"Original coordinates: {real_points}")
    print(f"Scale factor used: {scale_factor:.3f}")
    cv2.destroyAllWindows()
    exit()

def start_detection(method):
    global detection_method, mode, img_display, img_clean, hough_lines, selected_line_idx, back_pressed
    
    detection_method = method
    mode = "DRAWING" if method != "HOUGH" else "HOUGH_SELECT"
    back_pressed = False
    
    # Load video frame
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"Error: Could not load video from {video_path}")
        return False
    
    target_width = 1280  # Standard display width
    orig_h, orig_w = frame.shape[:2]
    global scale_factor
    scale_factor = target_width / float(orig_w)
    new_dim = (target_width, int(orig_h * scale_factor))
    
    img_clean = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
    img_display = img_clean.copy()
    
    if method == "HOUGH":
        print("Detecting Hough lines...")
        hough_lines = detect_hough_lines(img_clean)
        print(f"Found {len(hough_lines)} lines")
        if len(hough_lines) == 0:
            print("No lines detected. Try adjusting detection parameters.")
        redraw_hough_lines()
    
    # Create resizable window that fits screen
    cv2.namedWindow('Line Detection', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('Line Detection', mouse_callback)
    draw_ui(img_display, 0, 0)
    
    while True:
        cv2.imshow('Line Detection', img_display)
        key = cv2.waitKey(10) & 0xFF
        
        # Handle number key selection for Hough method
        if method == "HOUGH" and mode == "HOUGH_SELECT" and key >= ord('1') and key <= ord('8'):
            line_num = key - ord('1')
            if line_num < len(hough_lines):
                selected_line_idx = line_num
                mode = "DONE"
                redraw_hough_lines()
        
        if key == ord('q') or back_pressed:
            break
    
    # Close preprocessing windows
    cv2.destroyWindow('Grayscale')
    cv2.destroyWindow('Canny Edges')
    cv2.destroyAllWindows()
    return True

class LineDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Line Detection Tool")
        self.root.geometry("400x300")
        self.root.configure(bg='#2c3e50')
        
        # Title
        title = tk.Label(self.root, text="LINE DETECTION TOOL", 
                        font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title.pack(pady=20)
        
        # Buttons
        btn_style = {'font': ('Arial', 12), 'width': 25, 'height': 2, 'relief': 'raised'}
        
        tk.Button(self.root, text="Two Points Detection", 
                 command=lambda: self.start_detection("TWO_POINTS"),
                 bg='#3498db', fg='white', **btn_style).pack(pady=10)
        
        tk.Button(self.root, text="Multi Points Detection", 
                 command=lambda: self.start_detection("MULTIPOINTS"),
                 bg='#e74c3c', fg='white', **btn_style).pack(pady=10)
        
        tk.Button(self.root, text="Hough Lines Detection", 
                 command=lambda: self.start_detection("HOUGH"),
                 bg='#f39c12', fg='white', **btn_style).pack(pady=10)
        
        tk.Button(self.root, text="Exit", command=self.root.quit,
                 bg='#95a5a6', fg='white', **btn_style).pack(pady=20)
    
    def start_detection(self, method):
        self.root.withdraw()  # Hide GUI window
        start_detection(method)
        self.root.deiconify()  # Show GUI window again
    
    def run(self):
        self.root.mainloop()

def main():
    gui = LineDetectionGUI()
    gui.run()

if __name__ == "__main__":
    main()