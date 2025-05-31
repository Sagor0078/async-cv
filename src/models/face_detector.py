import asyncio
import numpy as np
from typing import Dict, List
import cv2
from PIL import Image
import io

from .base import BaseModel

class FaceDetector(BaseModel):
    """OpenCV-based face detector"""
    
    def __init__(self):
        super().__init__()
        self.face_cascade = None
        self.min_face_size = (30, 30)
    
    async def load_model(self):
        def _load():
            # Load OpenCV's pre-trained face detector
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        async with self.lock:
            if not self.is_loaded:
                loop = asyncio.get_event_loop()
                self.face_cascade = await loop.run_in_executor(None, _load)
                self.is_loaded = True
    
    async def predict(self, image_bytes: bytes) -> Dict[str, any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        def _detect_faces():
            # Load and convert image
            image = np.array(Image.open(io.BytesIO(image_bytes)))
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.min_face_size
            )
            
            height, width = image.shape[:2]
            detections = []
            
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': [int(x), int(y), int(w), int(h)],
                    'bbox_normalized': [x/width, y/height, w/width, h/height],
                    'confidence': 0.9  # OpenCV doesn't provide confidence scores
                })
            
            return {
                'faces': detections,
                'face_count': len(detections),
                'image_size': {'width': width, 'height': height},
                'model_type': 'face_detector'
            }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _detect_faces)