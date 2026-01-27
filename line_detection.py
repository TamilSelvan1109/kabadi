import cv2
import json  # <--- NEW: Library to save data

# --- CONFIGURATION ---
video_path = 'assets/back_angle_video1.mp4' 
# ---------------------

points = []         
scale_factor = 1.0  
mode = "IDLE"       
mouse_x, mouse_y = 0, 0 

def is_inside_button(x, y, btn_x1, btn_y1, btn_x2, btn_y2):
    return btn_x1 < x < btn_x2 and btn_y1 < y < btn_y2

def draw_ui(image, curr_x, curr_y):
    h, w, _ = image.shape
    
    # RESET BUTTON
    btn1_color = (200, 200, 200)
    if is_inside_button(curr_x, curr_y, 20, 20, 160, 70):
        btn1_color = (255, 255, 0)
    cv2.rectangle(image, (20, 20), (160, 70), btn1_color, -1)
    cv2.putText(image, "RESET", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
    
    # SAVE BUTTON
    btn2_color = (0, 0, 255) 
    txt2 = "EXIT"
    if mode == "DONE":
        btn2_color = (0, 255, 0) 
        txt2 = "SAVE"
        
    if is_inside_button(curr_x, curr_y, w - 160, 20, w - 20, 70):
        btn2_color = (0, 200, 0) if mode == "DONE" else (0, 0, 200)
            
    cv2.rectangle(image, (w - 160, 20), (w - 20, 70), btn2_color, -1)
    cv2.putText(image, txt2, (w - 130, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Instructions
    if mode == "DRAWING":
        msg = "Left Click to add points. RIGHT CLICK to finish."
        col = (0, 255, 255)
    elif mode == "DONE":
        msg = "Line Complete. Click SAVE."
        col = (0, 255, 0)
    else:
        msg = "Click anywhere to start drawing."
        col = (200, 200, 200)
    
    cv2.putText(image, msg, (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

def mouse_callback(event, x, y, flags, param):
    global points, mode, img_display, img_clean, mouse_x, mouse_y, scale_factor

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        # CLICKED RESET
        if is_inside_button(x, y, 20, 20, 160, 70): 
            points = []
            mode = "IDLE"
            img_display = img_clean.copy()
            print("Reset.")
            return

        # CLICKED SAVE / EXIT
        h, w, _ = img_display.shape
        if is_inside_button(x, y, w - 160, 20, w - 20, 70): 
            if mode == "DONE" and len(points) > 1:
                # --- AUTO SAVE LOGIC ---
                real_points = []
                for p in points:
                    rx = int(p[0] / scale_factor)
                    ry = int(p[1] / scale_factor)
                    real_points.append((rx, ry))
                
                # Write to config.json
                data = {"boundary_points": real_points}
                with open("config.json", "w") as f:
                    json.dump(data, f)
                    
                print("\n✅ SUCCESS: Points saved to 'config.json'!")
                print("You can now run main.py immediately.")
                cv2.destroyAllWindows()
                exit()
            else:
                print("Exited without saving.")
                cv2.destroyAllWindows()
                exit()
            return

        # DRAWING
        if mode != "DONE":
            mode = "DRAWING"
            points.append((x, y))
            cv2.circle(img_display, (x, y), 4, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.line(img_display, points[-2], points[-1], (0, 255, 0), 2)
            draw_ui(img_display, x, y)

    elif event == cv2.EVENT_RBUTTONDOWN:
        # FINISH DRAWING
        if len(points) > 1:
            mode = "DONE"
            img_display = img_clean.copy()
            for i in range(len(points) - 1):
                cv2.line(img_display, points[i], points[i+1], (0, 0, 255), 2)
                cv2.circle(img_display, points[i], 4, (0, 255, 0), -1)
            cv2.circle(img_display, points[-1], 4, (0, 255, 0), -1)
            draw_ui(img_display, x, y)
            print("Drawing finished. Click SAVE.")

    elif event == cv2.EVENT_MOUSEMOVE:
        if mode == "DRAWING":
            temp = img_display.copy()
            if len(points) > 0:
                cv2.line(temp, points[-1], (x, y), (0, 255, 255), 1)
            draw_ui(temp, x, y)
            cv2.imshow('Polyline Setup', temp)
        else:
            temp = img_display.copy()
            draw_ui(temp, x, y)
            cv2.imshow('Polyline Setup', temp)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    target_width = 1000
    orig_h, orig_w = frame.shape[:2]
    scale_factor = target_width / float(orig_w)
    new_dim = (target_width, int(orig_h * scale_factor))
    
    img_clean = cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)
    img_display = img_clean.copy()

    cv2.namedWindow('Polyline Setup')
    cv2.setMouseCallback('Polyline Setup', mouse_callback)
    
    draw_ui(img_display, 0, 0)

    while True:
        cv2.imshow('Polyline Setup', img_display)
        if cv2.waitKey(10) & 0xFF == ord('q'): break
            
    cv2.destroyAllWindows()
else:
    print("Error reading video.")