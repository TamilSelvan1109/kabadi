import urllib.request
import os

def download_model():
    model_url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    model_path = "pose_landmarker_lite.task"
    
    if os.path.exists(model_path):
        print(f"Model file already exists: {model_path}")
        return True
    
    try:
        print("Downloading MediaPipe pose model...")
        urllib.request.urlretrieve(model_url, model_path)
        print(f"Downloaded model to: {model_path}")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False

if __name__ == "__main__":
    download_model()