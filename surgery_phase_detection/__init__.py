"""
Surgery Phase Detection
========================

Automatic detection of surgical phases from operating room video data using
deep learning. Supports both the Cholec80 endoscopic dataset and the MVOR
multi-view operating room dataset.

Uses a ResNet + LSTM architecture to classify frames into surgical phases.
For MVOR, phases are derived from scene context (people, roles, poses).
"""

__version__ = "0.2.0"

# Cholec80 phases (endoscopic surgery)
CHOLEC80_PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

# MVOR phases (operating room activity)
MVOR_PHASE_NAMES = [
    "Idle/Empty",
    "Preparation",
    "Procedure Active",
    "Closure/Cleanup",
]

# Default to MVOR
PHASE_NAMES = MVOR_PHASE_NAMES
NUM_CLASSES = len(PHASE_NAMES)
