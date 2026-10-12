import os
import urllib.request
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

class AutomatedPotholeDetector:
    def __init__(self, model_path: str = "pothole_yolov8.pt"):
        """
        Initializes the YOLOv8 pothole model.
        Auto-downloads dedicated pre-trained pothole weights from Hugging Face if not found.
        Falls back to base yolov8n.pt if offline.
        """
        self.model_path = model_path
        
        # Auto-download pre-trained pothole model if not present locally
        if not os.path.exists(self.model_path):
            print(f"Pothole model '{self.model_path}' not found. Downloading pre-trained weights from Hugging Face...")
            huggingface_url = "https://huggingface.co/peterhdd/pothole-detection-yolov8/resolve/main/best.pt"
            try:
                req = urllib.request.Request(huggingface_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=30) as response, open(self.model_path, 'wb') as out_file:
                    out_file.write(response.read())
                print("Download complete: pothole_yolov8.pt")
            except Exception as e:
                print(f"Could not download custom weights ({e}). Falling back to yolov8n.pt...")
                self.model_path = "yolov8n.pt"

        self.model = YOLO(self.model_path)

    def detect(self, image: Image.Image, conf_threshold: float = 0.25):
        """
        Runs automated detection on a PIL Image.
        Returns:
            annotated_img (PIL.Image): Image with drawn bounding boxes.
            count (int): Number of detected potholes.
            avg_conf (float): Average confidence score.
            severity (float): Calculated severity metric (1.0 to 10.0).
        """
        img_np = np.array(image.convert("RGB"))
        results = self.model.predict(source=img_np, conf=conf_threshold, verbose=False)
        result = results[0]

        boxes = result.boxes
        count = len(boxes)

        if count == 0:
            return None, 0, 0.0, 0.0

        confidences = boxes.conf.cpu().numpy()
        avg_conf = float(np.mean(confidences))

        img_h, img_w, _ = img_np.shape
        total_pothole_area = 0.0
        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            total_pothole_area += (x2 - x1) * (y2 - y1)

        area_ratio = total_pothole_area / max(1.0, (img_h * img_w))
        severity = min(10.0, (count * 2.0) + (area_ratio * 20.0))

        # Ultralytics plot() returns BGR numpy array
        annotated_np = result.plot()
        annotated_img = Image.fromarray(cv2.cvtColor(annotated_np, cv2.COLOR_BGR2RGB))

        return annotated_img, count, round(avg_conf, 2), round(severity, 2)
