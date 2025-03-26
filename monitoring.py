import logging
import psutil
import threading
import time
from prometheus_client import Counter, Gauge, Histogram, start_http_server, CollectorRegistry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('monitoring')

class MetricsCollector:
    _instance = None
    _registry = None
    _metrics_initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, port: int = 8000):
        if MetricsCollector._metrics_initialized:
            return
            
        if MetricsCollector._registry is None:
            MetricsCollector._registry = CollectorRegistry()
            
        try:
            self.frames_processed = Counter(
                'frames_processed_total', 
                'Total number of frames processed',
                registry=MetricsCollector._registry
            )
            
            self.processing_time = Histogram(
                'frame_processing_seconds',
                'Time spent processing frames',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
                registry=MetricsCollector._registry
            )
            
            self.cpu_usage = Gauge(
                'cpu_usage_percent', 
                'CPU usage percentage',
                registry=MetricsCollector._registry
            )
            
            self.memory_usage = Gauge(
                'memory_usage_percent', 
                'Memory usage percentage',
                registry=MetricsCollector._registry
            )
            
            self.system_healthy = Gauge(
                'system_healthy',
                'Overall system health status',
                registry=MetricsCollector._registry
            )

            try:
                start_http_server(port, registry=MetricsCollector._registry)
            except OSError:
                pass

            self._start_collection()
            MetricsCollector._metrics_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize metrics: {e}")
            raise

    def _start_collection(self):
        def collect_metrics():
            while True:
                try:
                    cpu_percent = float(psutil.cpu_percent(interval=None))
                    memory_percent = float(psutil.virtual_memory().percent)

                    self.cpu_usage.set(cpu_percent)
                    self.memory_usage.set(memory_percent)
                    logger.info(f"CPU Usage: {cpu_percent}%, Memory Usage: {memory_percent}%")

                    time.sleep(5)  
                except Exception as e:
                    logger.error(f"Error collecting metrics: {e}", exc_info=True)
                    time.sleep(5)

        thread = threading.Thread(target=collect_metrics, daemon=True)
        thread.start()


    def record_metric(self, name: str, value: float):
        try:
            value = float(value) 
            if name == 'processing_time':
                self.processing_time.observe(value)
            elif name == 'frames_processed':
                self.frames_processed.inc(value)
        except Exception as e:
            logger.error(f"Error recording metric {name}: {e}", exc_info=True)

    def get_system_metrics(self):
        try:
            return {
                'cpu_usage': float(self.cpu_usage._value.get()),
                'memory_usage': float(self.memory_usage._value.get()),
                'system_healthy': float(self.system_healthy._value.get())
            }
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}", exc_info=True)
            return {
                'cpu_usage': 0.0,
                'memory_usage': 0.0,
                'system_healthy': 0.0
            }