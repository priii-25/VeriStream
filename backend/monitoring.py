# backend/monitoring.py
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('monitoring')

class MetricsCollector:
    def __init__(self, port: int = 9090):
        logger.info("MetricsCollector initialized (disabled)")
    
    def record_metric(self, name: str, value: float):
        pass
    
    def get_system_metrics(self):
        return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'system_healthy': 0.0}

if __name__ == "__main__":
    metrics = MetricsCollector()