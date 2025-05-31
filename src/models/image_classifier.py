import asyncio
import numpy as np
from typing import Dict, List
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import io

from .base import BaseModel

class ImageClassifier(BaseModel):
    """ResNet-based image classifier"""
    
    def __init__(self):
        super().__init__()
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    async def load_model(self):
        def _load():
            model = models.resnet50(weights='IMAGENET1K_V1')
            model.eval()
            return model
        
        async with self.lock:
            if not self.is_loaded:
                loop = asyncio.get_event_loop()
                self.model = await loop.run_in_executor(None, _load)
                self.is_loaded = True
    
    async def predict(self, image_bytes: bytes) -> Dict[str, any]:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        def _predict():
            # Load and preprocess image
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            input_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                
                # Get top 5 predictions
                top5_prob, top5_catid = torch.topk(probabilities, 5)
                
                results = []
                for i in range(5):
                    results.append({
                        'class_id': top5_catid[i].item(),
                        'class_name': f'class_{top5_catid[i].item()}',
                        'confidence': top5_prob[i].item()
                    })
                
                return {
                    'predictions': results,
                    'model_type': 'image_classifier'
                }
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _predict)