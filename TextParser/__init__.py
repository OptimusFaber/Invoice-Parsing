"""
TextParser - модуль для извлечения и организации текста из изображений.
"""

from .parser import process_image

__version__ = "0.1.0"
__all__ = [
    "process_image", 
    "preprocess_image_for_ocr", 
    "sauvola_threshold",
    "niblack_threshold",
    "adaptive_threshold",
    "otsu_threshold"
] 