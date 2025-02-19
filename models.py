import whisper
import torch
from optimized_deepfake_detector import OptimizedDeepfakeDetector
from typing import Tuple
from monitoring import MetricsCollector
import logging
from typing import Any

logger = logging.getLogger('veristream')
metrics = MetricsCollector(port=8000)

@torch.no_grad()
def load_models() -> Tuple[Any, OptimizedDeepfakeDetector]:
    try:
        whisper_model = whisper.load_model("base")
        detector = OptimizedDeepfakeDetector()
        return whisper_model, detector
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        metrics.system_healthy.set(0)
        raise