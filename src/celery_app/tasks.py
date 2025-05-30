import asyncio
import base64
import json
from typing import Dict, List, Any
from celery import current_task
from celery.exceptions import Retry
import redis

from app.celery_app.celery import celery_app
from app.models.image_classifier import ImageClassifier
from app.models.object_detector import ObjectDetector
from app.models.face_detector import FaceDetector
from app.core.config import settings

# Redis client for caching
redis_client = redis.from_url(settings.REDIS_URL)

# Global model instances for workers
models = {}

def get_or_create_model(model_type: str):
    """Get or create model instance"""
    if model_type not in models:
        if model_type == 'classifier':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models[model_type] = loop.run_until_complete(ImageClassifier.create())
        elif model_type == 'detector':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models[model_type] = loop.run_until_complete(ObjectDetector.create())
        elif model_type == 'face_detector':
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            models[model_type] = loop.run_until_complete(FaceDetector.create())
    
    return models[model_type]

@celery_app.task(bind=True, name='app.celery_app.tasks.classify_image_task')
def classify_image_task(self, image_data: str, task_id: str = None):
    """Celery task for image classification"""
    try:
        # Update task state
        self.update_state(state='PROCESSING', meta={'status': 'Loading model...'})
        
        # Get model
        model = get_or_create_model('classifier')
        
        # Decode image
        image_bytes = base64.b64decode(image_data)
        
        # Update task state
        self.update_state(state='PROCESSING', meta={'status': 'Running inference...'})
        
        # Run prediction
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(model.predict(image_bytes))
        
        # Cache result
        if task_id:
            redis_client.setex(
                f"result:{task_id}", 
                settings.REDIS_CACHE_TTL, 
                json.dumps(result)
            )
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'task_id': task_id
        }
        
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'task_id': task_id}
        )
        raise exc

@celery_app.task(bind=True, name='app.celery_app.tasks.detect_objects_task')
def detect_objects_task(self, image_data: str, task_id: str = None):
    """Celery task for object detection"""
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Loading model...'})
        
        model = get_or_create_model('detector')
        image_bytes = base64.b64decode(image_data)
        
        self.update_state(state='PROCESSING', meta={'status': 'Running detection...'})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(model.predict(image_bytes))
        
        if task_id:
            redis_client.setex(
                f"result:{task_id}", 
                settings.REDIS_CACHE_TTL, 
                json.dumps(result)
            )
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'task_id': task_id
        }
        
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'task_id': task_id}
        )
        raise exc

@celery_app.task(bind=True, name='app.celery_app.tasks.detect_faces_task')
def detect_faces_task(self, image_data: str, task_id: str = None):
    """Celery task for face detection"""
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Loading model...'})
        
        model = get_or_create_model('face_detector')
        image_bytes = base64.b64decode(image_data)
        
        self.update_state(state='PROCESSING', meta={'status': 'Detecting faces...'})
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(model.predict(image_bytes))
        
        if task_id:
            redis_client.setex(
                f"result:{task_id}", 
                settings.REDIS_CACHE_TTL, 
                json.dumps(result)
            )
        
        return {
            'status': 'SUCCESS',
            'result': result,
            'task_id': task_id
        }
        
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'task_id': task_id}
        )
        raise exc

@celery_app.task(bind=True, name='app.celery_app.tasks.analyze_image_task')
def analyze_image_task(self, image_data: str, task_id: str = None):
    """Celery task for complete image analysis"""
    try:
        self.update_state(state='PROCESSING', meta={'status': 'Loading models...'})
        
        # Load all models
        classifier = get_or_create_model('classifier')
        detector = get_or_create_model('detector')
        face_detector = get_or_create_model('face_detector')
        
        image_bytes = base64.b64decode(image_data)
        
        self.update_state(state='PROCESSING', meta={'status': 'Running analysis...'})
        
        # Run all models
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        tasks = [
            classifier.predict(image_bytes),
            detector.predict(image_bytes),
            face_detector.predict(image_bytes)
        ]
        
        results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        
        analysis = {}
        model_names = ['classifier', 'detector', 'face_detector']
        
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                analysis[model_names[i]] = result
        
        final_result = {'analysis': analysis}
        
        if task_id:
            redis_client.setex(
                f"result:{task_id}", 
                settings.REDIS_CACHE_TTL, 
                json.dumps(final_result)
            )
        
        return {
            'status': 'SUCCESS',
            'result': final_result,
            'task_id': task_id
        }
        
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'task_id': task_id}
        )
        raise exc

@celery_app.task(bind=True, name='app.celery_app.tasks.batch_process_task')
def batch_process_task(self, images_data: List[Dict], task_type: str, task_id: str = None):
    """Celery task for batch processing"""
    try:
        self.update_state(state='PROCESSING', meta={'status': f'Processing {len(images_data)} images...'})
        
        results = []
        total = len(images_data)
        
        for i, image_info in enumerate(images_data):
            try:
                # Update progress
                progress = (i + 1) / total * 100
                self.update_state(
                    state='PROCESSING', 
                    meta={'status': f'Processing image {i+1}/{total}', 'progress': progress}
                )
                
                image_data = image_info['data']
                filename = image_info['filename']
                
                # Process based on task type
                if task_type == 'classify':
                    model = get_or_create_model('classifier')
                elif task_type == 'detect':
                    model = get_or_create_model('detector')
                elif task_type == 'faces':
                    model = get_or_create_model('face_detector')
                else:
                    raise ValueError(f"Unknown task type: {task_type}")
                
                image_bytes = base64.b64decode(image_data)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(model.predict(image_bytes))
                
                results.append({
                    'filename': filename,
                    'success': True,
                    'result': result
                })
                
            except Exception as e:
                results.append({
                    'filename': image_info.get('filename', f'image_{i}'),
                    'success': False,
                    'error': str(e)
                })
        
        final_result = {
            'results': results,
            'total_processed': len(results),
            'successful': len([r for r in results if r['success']]),
            'failed': len([r for r in results if not r['success']])
        }
        
        if task_id:
            redis_client.setex(
                f"result:{task_id}", 
                settings.REDIS_CACHE_TTL, 
                json.dumps(final_result)
            )
        
        return {
            'status': 'SUCCESS',
            'result': final_result,
            'task_id': task_id
        }
        
    except Exception as exc:
        self.update_state(
            state='FAILURE',
            meta={'error': str(exc), 'task_id': task_id}
        )
        raise exc

# Health check task
@celery_app.task(name='app.celery_app.tasks.health_check')
def health_check():
    """Health check task for workers"""
    return {'status': 'healthy', 'timestamp': current_task.request.id}