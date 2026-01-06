"""VL-JEPA: Vision-Language Joint Embedding Predictive Architecture for Jetson Orin Nano"""

__version__ = "0.1.0"

from .models.vl_jepa import VLJEPAModel
from .models.vision_encoder import VisionEncoder
from .models.text_encoder import TextEncoder
from .models.predictor import PredictorMLP

__all__ = [
    "VLJEPAModel",
    "VisionEncoder", 
    "TextEncoder",
    "PredictorMLP",
]
