"""
Surgical Phase Labeler for MVOR
================================

Derives surgical activity phase labels from MVOR scene annotations.
Since MVOR does not include explicit phase labels, this module infers
the current OR (operating room) phase from observable scene features:

    - Number of people present
    - Person roles (clinician / patient)
    - Spatial positions (bounding box locations)
    - Pose features (keypoint patterns)
    - Temporal context (time of day, sequence within a day)

Defined Phases:
    0 — Idle/Empty        : No people or only background activity
    1 — Preparation       : Patient positioned, few clinicians arriving
    2 — Procedure Active  : Multiple clinicians actively engaged
    3 — Closure/Cleanup   : Activity winding down, clinicians departing

These phases map to the general surgical workflow:
    Preparation → Incision/Procedure → Suturing/Closure
"""

from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np


# Phase definitions
MVOR_PHASE_NAMES = [
    "Idle/Empty",
    "Preparation",
    "Procedure Active",
    "Closure/Cleanup",
]

NUM_PHASES = len(MVOR_PHASE_NAMES)


class MVORPhaseLabeler:
    """Derives surgical phase labels from MVOR annotations.

    Uses a rule-based approach combining scene occupancy, person roles,
    spatial features, and temporal smoothing to assign one of 4 phases
    to each multi-view frame.

    Args:
        num_clinician_threshold: Min clinicians for "Procedure Active".
        temporal_smooth_window: Window size for temporal label smoothing.
        use_temporal_context: Whether to apply temporal smoothing.
    """

    def __init__(
        self,
        num_clinician_threshold: int = 2,
        temporal_smooth_window: int = 3,
        use_temporal_context: bool = True,
    ):
        self.num_clinician_threshold = num_clinician_threshold
        self.temporal_smooth_window = temporal_smooth_window
        self.use_temporal_context = use_temporal_context

    def label_all_frames(
        self,
        multiview_frames: List[dict],
        anns_by_image: Dict[int, List[dict]],
    ) -> Dict[str, int]:
        """Assign phase labels to all multi-view frames.

        Args:
            multiview_frames: List of multi-view frame dicts from MVOR JSON.
            anns_by_image: Mapping from image_id to list of annotations.

        Returns:
            Dictionary mapping multiview frame ID to phase label (0-3).
        """
        # Group frames by day for temporal processing
        day_frames: Dict[int, List[Tuple[str, str, dict]]] = defaultdict(list)

        for mv_frame in multiview_frames:
            mv_id = mv_frame["id"]
            day_id = mv_frame["images"][0].get("day_id", 0)
            timestamp = mv_frame["images"][0].get("date_captured", "")

            # Aggregate annotations across all views
            all_anns = []
            for img_info in mv_frame["images"]:
                all_anns.extend(anns_by_image.get(img_info["id"], []))

            scene_features = self._extract_scene_features(all_anns)
            day_frames[day_id].append((mv_id, timestamp, scene_features))

        # Process each day independently
        all_labels: Dict[str, int] = {}

        for day_id in sorted(day_frames.keys()):
            frames = sorted(day_frames[day_id], key=lambda x: x[1])
            day_labels = self._label_day(frames)
            all_labels.update(day_labels)

        # Log distribution
        dist = defaultdict(int)
        for label in all_labels.values():
            dist[label] += 1
        print(f"Phase label distribution: {dict(sorted(dist.items()))}")
        for phase_id, count in sorted(dist.items()):
            name = MVOR_PHASE_NAMES[phase_id] if phase_id < len(MVOR_PHASE_NAMES) else "?"
            print(f"  {phase_id} ({name}): {count} frames")

        return all_labels

    def _extract_scene_features(self, annotations: List[dict]) -> dict:
        """Extract scene-level features from a set of annotations.

        Args:
            annotations: List of person annotations across all views.

        Returns:
            Dictionary of scene features.
        """
        # Deduplicate by person_id (same person seen in multiple views)
        unique_persons: Dict[int, List[dict]] = defaultdict(list)
        for ann in annotations:
            unique_persons[ann["person_id"]].append(ann)

        num_people = len(unique_persons)
        num_clinicians = 0
        num_patients = 0
        has_patient = False
        person_roles = []

        # Pose activity features
        bbox_areas = []
        wrist_positions = []
        head_positions = []

        for person_id, person_anns in unique_persons.items():
            # Use the first annotation for this person (any view)
            ann = person_anns[0]
            role = ann.get("person_role", "unknown")
            person_roles.append(role)

            if role == "clinician":
                num_clinicians += 1
            elif role == "patient":
                num_patients += 1
                has_patient = True

            # Bounding box area as proxy for proximity
            bbox = ann.get("bbox", [0, 0, 0, 0])
            bbox_area = bbox[2] * bbox[3] if len(bbox) >= 4 else 0
            bbox_areas.append(bbox_area)

            # Extract keypoints if available
            kps = ann.get("keypoints", [])
            if kps and not ann.get("only_bbox", 0):
                kps_array = np.array(kps).reshape(-1, 3)
                # Wrist positions (indices 8, 9 in CAMMA format)
                for wrist_idx in [8, 9]:
                    if wrist_idx < len(kps_array) and kps_array[wrist_idx, 2] > 0:
                        wrist_positions.append(kps_array[wrist_idx, :2])
                # Head position (index 0)
                if kps_array[0, 2] > 0:
                    head_positions.append(kps_array[0, :2])

        # Compute activity indicators
        avg_bbox_area = np.mean(bbox_areas) if bbox_areas else 0.0
        wrist_spread = 0.0
        if len(wrist_positions) >= 2:
            wrists = np.array(wrist_positions)
            wrist_spread = np.std(wrists, axis=0).mean()

        return {
            "num_people": num_people,
            "num_clinicians": num_clinicians,
            "num_patients": num_patients,
            "has_patient": has_patient,
            "person_roles": person_roles,
            "avg_bbox_area": avg_bbox_area,
            "wrist_spread": wrist_spread,
            "num_visible_wrists": len(wrist_positions),
        }

    def _label_single_frame(self, features: dict, position_ratio: float) -> int:
        """Assign a phase label to a single frame based on its scene features.

        Args:
            features: Scene features dictionary.
            position_ratio: Temporal position within the day (0=start, 1=end).

        Returns:
            Phase label (0-3).
        """
        num_people = features["num_people"]
        num_clinicians = features["num_clinicians"]
        has_patient = features["has_patient"]

        # Phase 0: Idle/Empty — no one in the room
        if num_people == 0:
            return 0

        # Phase 2: Procedure Active — multiple clinicians + patient
        if (
            num_clinicians >= self.num_clinician_threshold
            and has_patient
        ):
            return 2

        # Phase 1 vs 3: Preparation vs Closure
        # Distinguished by temporal position within the day
        if has_patient and num_clinicians >= 1:
            # Multiple clinicians with patient but below threshold
            # could be preparation (early) or closure (late)
            if position_ratio < 0.5:
                return 1  # Preparation
            else:
                return 3  # Closure

        if num_clinicians >= 1 and not has_patient:
            # Clinicians only, no patient
            if position_ratio < 0.3:
                return 1  # Preparation (setting up)
            elif position_ratio > 0.7:
                return 3  # Cleanup
            else:
                return 0  # Idle between procedures

        # Default: patient alone or edge case
        if has_patient and num_clinicians == 0:
            return 1  # Patient waiting = preparation

        return 0  # Fallback

    def _label_day(
        self, frames: List[Tuple[str, str, dict]]
    ) -> Dict[str, int]:
        """Label all frames within a single day.

        Applies per-frame labeling followed by optional temporal smoothing.

        Args:
            frames: List of (mv_id, timestamp, features) sorted by time.

        Returns:
            Dictionary mapping mv_id to phase label.
        """
        n = len(frames)
        if n == 0:
            return {}

        # Per-frame labeling with temporal position context
        raw_labels = []
        for i, (mv_id, timestamp, features) in enumerate(frames):
            position_ratio = i / max(n - 1, 1)
            label = self._label_single_frame(features, position_ratio)
            raw_labels.append(label)

        # Temporal smoothing via majority vote
        if self.use_temporal_context and n > 1:
            labels = self._temporal_smooth(raw_labels)
        else:
            labels = raw_labels

        # Build output
        result: Dict[str, int] = {}
        for i, (mv_id, _, _) in enumerate(frames):
            result[mv_id] = labels[i]

        return result

    def _temporal_smooth(self, labels: List[int]) -> List[int]:
        """Smooth labels using a majority-vote sliding window.

        Args:
            labels: Raw per-frame labels.

        Returns:
            Smoothed labels.
        """
        n = len(labels)
        half = self.temporal_smooth_window // 2
        smoothed = labels.copy()

        for i in range(n):
            start = max(0, i - half)
            end = min(n, i + half + 1)
            window = labels[start:end]

            # Majority vote
            counts = defaultdict(int)
            for lbl in window:
                counts[lbl] += 1
            smoothed[i] = max(counts, key=counts.get)

        return smoothed

    @staticmethod
    def get_phase_names() -> List[str]:
        """Return the list of phase names."""
        return MVOR_PHASE_NAMES.copy()

    @staticmethod
    def get_num_phases() -> int:
        """Return the number of phases."""
        return NUM_PHASES
