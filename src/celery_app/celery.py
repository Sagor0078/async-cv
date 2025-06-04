from celery import Celery
from src.core.config import settings


def create_celery_app() -> Celery:
    celery_app = Celery("cv_inference")

    celery_app.conf.update(
        broker_url=settings.CELERY_BROKER_URL,
        result_backend=settings.CELERY_RESULT_BACKEND,
        task_serializer=settings.CELERY_TASK_SERIALIZER,
        result_serializer=settings.CELERY_RESULT_SERIALIZER,
        accept_content=settings.CELERY_ACCEPT_CONTENT,
        timezone=settings.CELERY_TIMEZONE,
        enable_utc=settings.CELERY_ENABLE_UTC,
        task_track_started=settings.CELERY_TASK_TRACK_STARTED,
        task_time_limit=settings.CELERY_TASK_TIME_LIMIT,
        task_soft_time_limit=settings.CELERY_TASK_SOFT_TIME_LIMIT,
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=50,
        task_routes={
            "src.celery_app.tasks.classify_image_task": {"queue": "classification"},
            "src.celery_app.tasks.detect_objects_task": {"queue": "detection"},
            "src.celery_app.tasks.detect_faces_task": {"queue": "faces"},
            "src.celery_app.tasks.analyze_image_task": {"queue": "analysis"},
            "src.celery_app.tasks.batch_process_task": {"queue": "batch"},
        },
    )

    return celery_app


celery_app = create_celery_app()
