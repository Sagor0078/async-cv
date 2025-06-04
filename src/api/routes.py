import base64
import json
import uuid
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
import redis

from src.main import models
from src.api.schemas import PredictionResponse, TaskResponse, TaskStatusResponse
from src.celery_app.celery import celery_app
from src.celery_app import tasks
from src.core.config import settings

router = APIRouter()

# Redis client for task results
redis_client = redis.from_url(settings.REDIS_URL)


# Sync endpoints (original functionality)
@router.post("/sync/classify", response_model=PredictionResponse)
async def classify_image_sync(file: UploadFile = File(...)):
    """Classify a single image synchronously"""
    if "classifier" not in models:
        raise HTTPException(status_code=503, detail="Image classifier not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        result = await models["classifier"].predict(contents)
        return PredictionResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/detect", response_model=PredictionResponse)
async def detect_objects_sync(file: UploadFile = File(...)):
    """Detect objects in an image synchronously"""
    if "detector" not in models:
        raise HTTPException(status_code=503, detail="Object detector not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        result = await models["detector"].predict(contents)
        return PredictionResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sync/detect-faces", response_model=PredictionResponse)
async def detect_faces_sync(file: UploadFile = File(...)):
    """Detect faces in an image synchronously"""
    if "face_detector" not in models:
        raise HTTPException(status_code=503, detail="Face detector not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        result = await models["face_detector"].predict(contents)
        return PredictionResponse(success=True, data=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Async endpoints using Celery
@router.post("/async/classify", response_model=TaskResponse)
async def classify_image_async(file: UploadFile = File(...)):
    """Classify a single image asynchronously using Celery"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_data = base64.b64encode(contents).decode("utf-8")
        task_id = str(uuid.uuid4())

        # Submit task to Celery
        task = tasks.classify_image_task.delay(image_data, task_id)

        return TaskResponse(
            task_id=task.id, status="PENDING", message="Task submitted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async/detect", response_model=TaskResponse)
async def detect_objects_async(file: UploadFile = File(...)):
    """Detect objects in an image asynchronously using Celery"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_data = base64.b64encode(contents).decode("utf-8")
        task_id = str(uuid.uuid4())

        task = tasks.detect_objects_task.delay(image_data, task_id)

        return TaskResponse(
            task_id=task.id, status="PENDING", message="Task submitted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async/detect-faces", response_model=TaskResponse)
async def detect_faces_async(file: UploadFile = File(...)):
    """Detect faces in an image asynchronously using Celery"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_data = base64.b64encode(contents).decode("utf-8")
        task_id = str(uuid.uuid4())

        task = tasks.detect_faces_task.delay(image_data, task_id)

        return TaskResponse(
            task_id=task.id, status="PENDING", message="Task submitted successfully"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async/analyze", response_model=TaskResponse)
async def analyze_image_async(file: UploadFile = File(...)):
    """Run all available models on an image asynchronously"""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image_data = base64.b64encode(contents).decode("utf-8")
        task_id = str(uuid.uuid4())

        task = tasks.analyze_image_task.delay(image_data, task_id)

        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message="Analysis task submitted successfully",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/async/batch-classify", response_model=TaskResponse)
async def batch_classify_async(files: List[UploadFile] = File(...)):
    """Classify multiple images asynchronously"""
    if len(files) > 50:
        raise HTTPException(
            status_code=400, detail="Maximum 50 files allowed for batch processing"
        )

    try:
        images_data = []
        for file in files:
            if not file.content_type.startswith("image/"):
                continue

            contents = await file.read()
            image_data = base64.b64encode(contents).decode("utf-8")
            images_data.append({"filename": file.filename, "data": image_data})

        task_id = str(uuid.uuid4())
        task = tasks.batch_process_task.delay(images_data, "classify", task_id)

        return TaskResponse(
            task_id=task.id,
            status="PENDING",
            message=f"Batch classification task submitted for {len(images_data)} images",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Task status endpoints
@router.get("/tasks/{task_id}/status", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """Get status of a Celery task"""
    try:
        # Check if task exists in Celery
        task_result = celery_app.AsyncResult(task_id)

        if task_result.state == "PENDING":
            response = {
                "task_id": task_id,
                "status": "PENDING",
                "message": "Task is waiting to be processed",
            }
        elif task_result.state == "PROCESSING":
            response = {
                "task_id": task_id,
                "status": "PROCESSING",
                "message": task_result.info.get("status", "Processing..."),
                "progress": task_result.info.get("progress"),
            }
        elif task_result.state == "SUCCESS":
            # Try to get result from Redis cache first
            cached_result = redis_client.get(f"result:{task_id}")
            if cached_result:
                result_data = json.loads(cached_result)
            else:
                result_data = task_result.result.get("result", {})

            response = {
                "task_id": task_id,
                "status": "SUCCESS",
                "message": "Task completed successfully",
                "result": result_data,
            }
        elif task_result.state == "FAILURE":
            response = {
                "task_id": task_id,
                "status": "FAILURE",
                "message": "Task failed",
                "error": str(task_result.info),
            }
        else:
            response = {
                "task_id": task_id,
                "status": task_result.state,
                "message": f"Task state: {task_result.state}",
            }

        return TaskStatusResponse(**response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """Cancel a running task"""
    try:
        celery_app.control.revoke(task_id, terminate=True)

        # Remove from Redis cache
        redis_client.delete(f"result:{task_id}")

        return {"message": f"Task {task_id} cancelled successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks")
async def list_active_tasks():
    """List all active tasks"""
    try:
        # Get active tasks from Celery
        inspect = celery_app.control.inspect()
        active_tasks = inspect.active()

        if not active_tasks:
            return {"active_tasks": [], "total": 0}

        all_tasks = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                all_tasks.append(
                    {
                        "task_id": task["id"],
                        "name": task["name"],
                        "worker": worker,
                        "args": task.get("args", []),
                        "kwargs": task.get("kwargs", {}),
                    }
                )

        return {"active_tasks": all_tasks, "total": len(all_tasks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# System monitoring endpoints
@router.get("/workers/status")
async def get_workers_status():
    """Get status of all Celery workers"""
    try:
        inspect = celery_app.control.inspect()

        # Get worker stats
        stats = inspect.stats()
        active = inspect.active()
        registered = inspect.registered()

        workers = {}
        if stats:
            for worker_name, worker_stats in stats.items():
                workers[worker_name] = {
                    "status": "online",
                    "active_tasks": len(active.get(worker_name, [])) if active else 0,
                    "processed_tasks": worker_stats.get("total", {}).get(
                        "tasks.completed", 0
                    ),
                    "registered_tasks": len(registered.get(worker_name, []))
                    if registered
                    else 0,
                    "load_avg": worker_stats.get("rusage", {}).get("stime", 0),
                    "memory_info": worker_stats.get("rusage", {}),
                }

        return {
            "workers": workers,
            "total_workers": len(workers),
            "online_workers": len(
                [w for w in workers.values() if w["status"] == "online"]
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/queues/status")
async def get_queues_status():
    """Get status of all queues"""
    try:
        inspect = celery_app.control.inspect()

        # Get queue lengths (this requires RabbitMQ management plugin)
        queues = {
            "classification": {"length": 0, "consumers": 0},
            "detection": {"length": 0, "consumers": 0},
            "faces": {"length": 0, "consumers": 0},
            "analysis": {"length": 0, "consumers": 0},
            "batch": {"length": 0, "consumers": 0},
        }

        # Get active tasks per queue
        active_tasks = inspect.active()
        if active_tasks:
            for worker, tasks in active_tasks.items():
                for task in tasks:
                    task_name = task.get("name", "")
                    if "classify" in task_name:
                        queues["classification"]["consumers"] += 1
                    elif "detect_objects" in task_name:
                        queues["detection"]["consumers"] += 1
                    elif "detect_faces" in task_name:
                        queues["faces"]["consumers"] += 1
                    elif "analyze" in task_name:
                        queues["analysis"]["consumers"] += 1
                    elif "batch" in task_name:
                        queues["batch"]["consumers"] += 1

        return {"queues": queues}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/status")
async def get_model_status():
    """Get status of all loaded models"""
    status = {}
    for name, model in models.items():
        status[name] = {
            "loaded": model.is_loaded if hasattr(model, "is_loaded") else True,
            "type": type(model).__name__,
        }
    return {"models": status, "total_models": len(models)}


# Health check endpoint
@router.get("/health/celery")
async def celery_health_check():
    """Check if Celery workers are healthy"""
    try:
        # Send a simple health check task
        task = tasks.health_check.delay()
        result = task.get(timeout=10)  # Wait up to 10 seconds

        return {"celery_status": "healthy", "task_id": task.id, "result": result}
    except Exception as e:
        return {"celery_status": "unhealthy", "error": str(e)}
