"""
Surgery Phase Detection
========================

Automatic detection of surgical phases from video data using deep learning.
Uses a ResNet + LSTM architecture to classify frames from the Cholec80 dataset
into one of 7 surgical phases.
"""

__version__ = "0.1.0"

PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

NUM_CLASSES = len(PHASE_NAMES)
