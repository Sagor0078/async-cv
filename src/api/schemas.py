from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    success: bool
    results: List[Dict[str, Any]]
    total_processed: int
    error: Optional[str] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    message: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None

class ModelStatus(BaseModel):
    loaded: bool
    type: str

class ModelsStatusResponse(BaseModel):
    models: Dict[str, ModelStatus]
    total_models: int

class WorkerInfo(BaseModel):
    status: str
    active_tasks: int
    processed_tasks: int
    registered_tasks: int
    load_avg: float
    memory_info: Dict[str, Any]

class WorkersStatusResponse(BaseModel):
    workers: Dict[str, WorkerInfo]
    total_workers: int
    online_workers: int

class QueueInfo(BaseModel):
    length: int
    consumers: int

class QueuesStatusResponse(BaseModel):
    queues: Dict[str, QueueInfo]