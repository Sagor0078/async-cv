"""
Async CV Inference System Architecture Diagram

Includes:
- Ingress (Nginx)
- FastAPI service
- RabbitMQ broker with volume
- Redis backend with volume
- PostgreSQL database
- Multiple Celery workers
- Monitoring (Flower via Prometheus)
- Healthcheck endpoint
"""

from diagrams import Diagram, Cluster
from diagrams.onprem.inmemory import Redis
from diagrams.onprem.queue import RabbitMQ
from diagrams.programming.language import Python
from diagrams.onprem.monitoring import Prometheus
from diagrams.onprem.client import Users
from diagrams.onprem.network import Nginx
from diagrams.onprem.database import PostgreSQL
from diagrams.generic.storage import Storage
from diagrams.custom import Custom

# Graph attributes
graph_attr = {
    "pad": "0.3",
    "splines": "curved",
    "nodesep": "0.5",
    "ranksep": "0.5",
    "fontcolor": "#2c3e50"
}

# Node attributes
node_attr = {
    "fontsize": "10",
    "width": "0.7",
    "height": "0.7",
    "fontcolor": "#2c3e50"
}

with Diagram(
    name="Async CV Inference System Architecture",
    show=False,
    filename="docs/diagram/architecture",
    graph_attr=graph_attr,
    node_attr=node_attr,
    edge_attr={"color": "#5D6D7E"}
):

    # Client
    client = Users("Clients")

    # Ingress Layer
    with Cluster("Ingress"):
        ingress = Nginx("Nginx\n(Load Balancer)")

    # API Layer
    with Cluster("API Layer"):
        api = Python("cv-api\n(FastAPI)")
        healthcheck = Custom("Healthcheck\n/health", "./icons/heartbeat.png")

    # Message Broker
    with Cluster("Message Queue"):
        broker = RabbitMQ("RabbitMQ\n(Broker)")
        broker_vol = Storage("rabbitmq_data")
        broker >> broker_vol

    # Caching Backend
    with Cluster("Cache/Backend"):
        cache = Redis("Redis\n(Result Store)")
        cache_vol = Storage("redis_data")
        cache >> cache_vol

    # Metadata Database
    with Cluster("Database"):
        db = PostgreSQL("PostgreSQL\n(Task Metadata)")

    # Celery Workers
    with Cluster("Celery Workers"):
        worker_cls = Python("Classification\nQueue")
        worker_det = Python("Object Detection\nQueue")
        worker_face = Python("Face Detection\nQueue")
        worker_analysis = Python("Analysis & Batch\nQueues")

    # Monitoring
    with Cluster("Monitoring"):
        monitoring = Prometheus("Celery Flower")

    # Workflow connections
    client >> ingress >> api >> healthcheck
    api >> broker
    api >> cache
    api >> db

    # Worker connections
    broker >> [worker_cls, worker_det, worker_face, worker_analysis]
    cache >> [worker_cls, worker_det, worker_face, worker_analysis]

    # Monitoring connections
    broker >> monitoring
    cache >> monitoring
