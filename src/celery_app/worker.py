"""
Celery worker startup script
"""
import os
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.celery_app.celery import celery_app

if __name__ == '__main__':
    celery_app.start()