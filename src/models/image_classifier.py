import asyncio
from typing import Dict
import tensorflow as tf
import numpy as np
from PIL import Image
import io

from .base import BaseModel


class ImageClassifier(BaseModel):
    """ResNet-based image classifier (TensorFlow version)"""

    def __init__(self):
        super().__init__()
        self.transform_size = (224, 224)
        self.model = None
        self.classes = None  # TensorFlow's model returns ImageNet class names
        self.decode_predictions = tf.keras.applications.resnet50.decode_predictions

    async def load_model(self):
        def _load():
            model = tf.keras.applications.ResNet50(weights="imagenet")
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
            if image.mode != "RGB":
                image = image.convert("RGB")

            image = image.resize(self.transform_size)
            image_array = tf.keras.preprocessing.image.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = tf.keras.applications.resnet50.preprocess_input(image_array)

            # Make prediction
            predictions = self.model.predict(image_array)
            decoded = self.decode_predictions(predictions, top=5)[0]

            results = []
            for class_id, class_name, confidence in decoded:
                results.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": float(confidence),
                    }
                )

            return {"predictions": results, "model_type": "image_classifier"}

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _predict)
