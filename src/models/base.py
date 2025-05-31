import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np
from PIL import Image
import io

class BaseModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.lock = asyncio.Lock()
    
    @classmethod
    async def create(cls):
        """Async factory method"""
        instance = cls()
        await instance.load_model()
        return instance
    
    @abstractmethod
    async def load_model(self):
        """Load the ML model"""
        pass
    
    @abstractmethod
    async def predict(self, data: Any) -> Dict[str, Any]:
        """Make prediction"""
        pass
    
    async def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Common image preprocessing"""
        def _process():
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _process)
    
    def resize_image(self, image: np.ndarray, target_size: tuple) -> np.ndarray:
        """Resize image to target size"""
        pil_image = Image.fromarray(image)
        resized = pil_image.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(resized)