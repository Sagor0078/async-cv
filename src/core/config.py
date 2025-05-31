import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model settings
    MODEL_CACHE_DIR: str = "./models"
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024  # 10MB
    BATCH_SIZE_LIMIT: int = 10
    
    # Performance settings
    MAX_WORKERS: int = 4
    REQUEST_TIMEOUT: int = 30
    
    # Celery settings
    CELERY_BROKER_URL: str = "pyamqp://guest:guest@rabbitmq:5672//"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/0"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: list = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 300  # 5 minutes
    CELERY_TASK_SOFT_TIME_LIMIT: int = 240  # 4 minutes
    
    # Redis settings
    REDIS_URL: str = "redis://redis:6379/1"
    REDIS_CACHE_TTL: int = 3600  # 1 hour
    
    # RabbitMQ settings
    RABBITMQ_DEFAULT_USER: str = "guest"
    RABBITMQ_DEFAULT_PASS: str = "guest"
    RABBITMQ_HOST: str = "rabbitmq"
    RABBITMQ_PORT: int = 5672
    
    class Config:
        env_file = ".env"

settings = Settings()