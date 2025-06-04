import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router
from src.models.image_classifier import ImageClassifier
from src.models.object_detector import ObjectDetector
from src.models.face_detector import FaceDetector
from src.core.config import settings

models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup and cleanup on shutdown"""
    print("Loading ML models...")

    # Load models asynchronously
    tasks = [load_classifier(), load_object_detector(), load_face_detector()]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Failed to load model {i}: {result}")
        else:
            print(f"Model {i} loaded successfully")

    print("All models loaded!")
    yield

    # Cleanup
    print("Shutting down models...")
    models.clear()


async def load_classifier():
    models["classifier"] = await ImageClassifier.create()


async def load_object_detector():
    models["detector"] = await ObjectDetector.create()


async def load_face_detector():
    models["face_detector"] = await FaceDetector.create()


app = FastAPI(
    title="Async Computer Vision API",
    description="Asynchronous ML inference API for computer vision tasks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {"message": "Async Computer Vision API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "models_loaded": len(models)}


# if __name__ == "__main__":
#     uvicorn.run(
#         "app.main:app",
#         host=settings.HOST,
#         port=settings.PORT,
#         reload=settings.DEBUG,
#         workers=1 if settings.DEBUG else 4,
#     )
