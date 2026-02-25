import logging
import time
from typing import Dict, Any
from print_color import print

# Setup basic logging to file/terminal
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
_standard_logger = logging.getLogger("RAG-Pipeline")

class PipelineLogger:
    def __init__(self):
        self.metrics = []

    def info(self, msg: str):
        _standard_logger.info(msg)
        print(msg, tag='INFO', tag_color='cyan', color='white')

    def success(self, msg: str):
        _standard_logger.info(f"SUCCESS: {msg}")
        print(msg, tag='SUCCESS', tag_color='green', color='white', format='bold')

    def warning(self, msg: str):
        _standard_logger.warning(msg)
        print(msg, tag='WAIT', tag_color='yellow', color='white')

    def error(self, msg: str):
        _standard_logger.error(msg)
        print(msg, tag='ERROR', tag_color='red', color='white', format='bold')

    def log_performance(self, model_name: str, metrics: Dict[str, Any]):
        self.metrics.append({
            "model": model_name,
            **metrics
        })
        msg = f"Performance logged for {model_name} (Latency: {metrics['total_latency']:.2f}s)"
        _standard_logger.info(msg)
        print(msg, tag='METRIC', tag_color='magenta', color='white')

    def get_comparison_table(self):
        return self.metrics

    def clear(self):
        self.metrics = []

# Global instances
logger = PipelineLogger()
metrics_logger = logger # For backward compatibility
