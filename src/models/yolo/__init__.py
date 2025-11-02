"""
YOLO Model Package
=================
YOLO-based classification for fracture detection
"""

# Import YOLO model if file exists
try:
    from .yolo_model import *
except ImportError:
    pass