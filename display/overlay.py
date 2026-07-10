"""OpenCV HUD overlay renderer for the Safety Copilot live view.

Draws bounding boxes, pose skeletons, danger zones, hazard indicators,
alert banners, and FPS counter on each frame.
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional

import cv2
import numpy as np

from core.models import (
    DangerZone,
    Detection,
    FrameResult,
    HazardState,
    PoseData,
    Severity,
    KEYPOINT_NOSE,
    KEYPOINT_LEFT_EYE,
    KEYPOINT_RIGHT_EYE,
    KEYPOINT_LEFT_EAR,
    KEYPOINT_RIGHT_EAR,
    KEYPOINT_LEFT_SHOULDER,
    KEYPOINT_RIGHT_SHOULDER,
    KEYPOINT_LEFT_ELBOW,
    KEYPOINT_RIGHT_ELBOW,
    KEYPOINT_LEFT_WRIST,
    KEYPOINT_RIGHT_WRIST,
    KEYPOINT_LEFT_HIP,
    KEYPOINT_RIGHT_HIP,
    KEYPOINT_LEFT_KNEE,
    KEYPOINT_RIGHT_KNEE,
    KEYPOINT_LEFT_ANKLE,
    KEYPOINT_RIGHT_ANKLE,
)

# ── COCO skeleton connections ────────────────────────────────
# Each tuple is (keypoint_a, keypoint_b, side).
# side: 'L' = left (blue), 'R' = right (red), 'C' = center (green)

SKELETON_CONNECTIONS: list[tuple[int, int, str]] = [
    # Face
    (KEYPOINT_NOSE, KEYPOINT_LEFT_EYE, "C"),
    (KEYPOINT_NOSE, KEYPOINT_RIGHT_EYE, "C"),
    (KEYPOINT_LEFT_EYE, KEYPOINT_LEFT_EAR, "L"),
    (KEYPOINT_RIGHT_EYE, KEYPOINT_RIGHT_EAR, "R"),
    # Torso
    (KEYPOINT_LEFT_SHOULDER, KEYPOINT_RIGHT_SHOULDER, "C"),
    (KEYPOINT_LEFT_SHOULDER, KEYPOINT_LEFT_HIP, "L"),
    (KEYPOINT_RIGHT_SHOULDER, KEYPOINT_RIGHT_HIP, "R"),
    (KEYPOINT_LEFT_HIP, KEYPOINT_RIGHT_HIP, "C"),
    # Left arm
    (KEYPOINT_LEFT_SHOULDER, KEYPOINT_LEFT_ELBOW, "L"),
    (KEYPOINT_LEFT_ELBOW, KEYPOINT_LEFT_WRIST, "L"),
    # Right arm
    (KEYPOINT_RIGHT_SHOULDER, KEYPOINT_RIGHT_ELBOW, "R"),
    (KEYPOINT_RIGHT_ELBOW, KEYPOINT_RIGHT_WRIST, "R"),
    # Left leg
    (KEYPOINT_LEFT_HIP, KEYPOINT_LEFT_KNEE, "L"),
    (KEYPOINT_LEFT_KNEE, KEYPOINT_LEFT_ANKLE, "L"),
    # Right leg
    (KEYPOINT_RIGHT_HIP, KEYPOINT_RIGHT_KNEE, "R"),
    (KEYPOINT_RIGHT_KNEE, KEYPOINT_RIGHT_ANKLE, "R"),
]

# Side → BGR colour
_SIDE_COLORS = {
    "L": (255, 170, 0),   # Blue-ish
    "R": (0, 85, 255),    # Red-ish
    "C": (0, 200, 0),     # Green
}

# Class-name → BGR colour for bounding boxes
_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "person": (0, 255, 0),
    "car": (255, 128, 0),
    "truck": (255, 128, 0),
    "forklift": (255, 128, 0),
    "hardhat": (0, 200, 200),
    "no-hardhat": (0, 0, 255),
    "safety-vest": (0, 200, 200),
    "no-safety-vest": (0, 0, 255),
    "vest": (0, 200, 200),
    "no-vest": (0, 0, 255),
    "goggles": (0, 200, 200),
    "no-goggles": (0, 0, 255),
    "gloves": (0, 200, 200),
    "no-gloves": (0, 0, 255),
    "boots": (0, 200, 200),
    "no-boots": (0, 0, 255),
    "mask": (0, 200, 200),
    "no-mask": (0, 0, 255),
    "safety-cone": (0, 165, 255),
    "machinery": (255, 128, 0),
    "utility-pole": (255, 128, 0),
    "hammer": (180, 105, 255),
    "drill": (0, 165, 255),
    "saw": (147, 20, 255),
    "measuring_tape": (238, 130, 238),
}
_DEFAULT_CLASS_COLOR = (200, 200, 200)

# Hazard state → border BGR colour
_STATE_COLORS: dict[HazardState, tuple[int, int, int]] = {
    HazardState.DETECTED: (200, 200, 200),
    HazardState.PASSIVE: (0, 255, 255),       # Yellow
    HazardState.UNNOTICED: (0, 165, 255),     # Orange
    HazardState.ESCALATED: (0, 0, 255),       # Red
    HazardState.ACKNOWLEDGED: (0, 255, 0),    # Green
    HazardState.RESOLVED: (200, 200, 200),
}


class OverlayRenderer:
    """Renders the HUD overlay onto each video frame.

    Parameters
    ----------
    config:
        Display configuration dict.  Expected keys:

        - ``show_keypoints`` (bool, default True)
        - ``show_gaze_lines`` (bool, default True)
        - ``show_danger_zones`` (bool, default True)
        - ``show_fps`` (bool, default True)
        - ``zone_alpha`` (float, default 0.25) — danger-zone transparency
        - ``banner_height`` (int, default 50)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.show_keypoints: bool = cfg.get("show_keypoints", True)
        self.show_gaze_lines: bool = cfg.get("show_gaze_lines", True)
        self.show_danger_zones: bool = cfg.get("show_danger_zones", True)
        self.show_fps: bool = cfg.get("show_fps", True)
        self.zone_alpha: float = cfg.get("zone_alpha", 0.25)
        self.banner_height: int = cfg.get("banner_height", 50)

    # ── Main entry point ────────────────────────────────────

    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw all HUD elements onto *frame* and return the annotated copy.

        Parameters
        ----------
        frame:
            BGR image from OpenCV (will be copied, original is not modified).
        result:
            Complete pipeline output for this frame.

        Returns
        -------
        np.ndarray
            Annotated frame with all overlays drawn.
        """
        out = frame.copy()

        # 1. Danger zones (semi-transparent polygons)
        if self.show_danger_zones:
            for zone in result.active_zones:
                self._draw_zone(out, zone)

        # 2. Bounding boxes with class labels and track IDs
        for det in result.detections:
            self._draw_bbox(out, det, result.tracked_objects)

        # 3. Pose skeletons
        if self.show_keypoints:
            for pose in result.poses:
                self._draw_skeleton(out, pose.keypoints)

                # 4. Gaze direction lines
                if self.show_gaze_lines and pose.head_yaw is not None:
                    self._draw_gaze_line(out, pose)

        # 5. Hazard state indicators
        for hazard in result.hazards:
            self._draw_hazard_indicator(out, hazard)

        # 6. Alert banner
        if result.alerts:
            most_severe = max(result.alerts, key=lambda a: list(Severity).index(a.severity))
            self._draw_alert_banner(out, most_severe)

        # 7. FPS counter
        if self.show_fps:
            self._draw_fps(out, result.fps)

        # 8. Tool carrying links (wrist to tool lines)
        self._draw_tool_carrying_links(out, result)

        return out

    def _draw_tool_carrying_links(self, frame: np.ndarray, result: FrameResult) -> None:
        """Draw lines connecting worker wrists to tools in close proximity."""
        workers = [
            obj for obj in result.tracked_objects.values()
            if obj.class_name == "person" and obj.is_active
        ]
        tools = [
            obj for obj in result.tracked_objects.values()
            if obj.class_name in ("hammer", "drill", "saw", "measuring_tape") and obj.is_active
        ]
        poses_by_track = {pose.person_track_id: pose for pose in result.poses}

        for worker in workers:
            pose = poses_by_track.get(worker.track_id)
            if not pose or pose.keypoints is None or len(pose.keypoints) < 11:
                continue
            for tool in tools:
                for index in (KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST):
                    if index < len(pose.keypoints):
                        wrist = pose.keypoints[index]
                        if len(wrist) >= 3 and wrist[2] >= 0.2:
                            dist = math.hypot(tool.center[0] - wrist[0], tool.center[1] - wrist[1])
                            if dist <= 120:
                                wx, wy = int(wrist[0]), int(wrist[1])
                                tx, ty = int(tool.center[0]), int(tool.center[1])
                                # Draw a violet line from hand to tool
                                cv2.line(frame, (wx, wy), (tx, ty), (238, 130, 238), 2, cv2.LINE_AA)
                                cv2.putText(
                                    frame,
                                    "carrying",
                                    (int((wx + tx) / 2), int((wy + ty) / 2) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.35,
                                    (238, 130, 238),
                                    1,
                                    cv2.LINE_AA,
                                )

    # ── Drawing primitives ──────────────────────────────────

    def _draw_zone(self, frame: np.ndarray, zone: DangerZone) -> None:
        """Draw a semi-transparent filled polygon for a danger zone."""
        pts = np.array(zone.polygon, dtype=np.int32)
        overlay = frame.copy()
        color = self._severity_color(zone.severity_base)
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, self.zone_alpha, frame, 1 - self.zone_alpha, 0, frame)
        # Outline and label
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        # Label at centroid
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = f"{zone.name} ({zone.zone_type})"
        cv2.putText(frame, label, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def _draw_bbox(
        self,
        frame: np.ndarray,
        det: Detection,
        tracked_objects: dict[int, TrackedObject] | None = None,
    ) -> None:
        """Draw a colour-coded bounding box with class label and track ID."""
        x1, y1, x2, y2 = det.bbox
        color = _CLASS_COLORS.get(det.class_name, _DEFAULT_CLASS_COLOR)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{det.class_name} {det.confidence:.0%}"
        if det.track_id is not None:
            label += f" #{det.track_id}"
            if tracked_objects and det.track_id in tracked_objects:
                dist = tracked_objects[det.track_id].distance_meters
                if dist is not None:
                    label += f" [{dist}m]"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )

    def _draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> None:
        """Draw COCO-17 skeleton with left/right side colours.

        Parameters
        ----------
        keypoints:
            Shape ``(17, 3)`` — x, y, confidence per keypoint.
        """
        if keypoints is None or len(keypoints) < 17:
            return

        min_conf = 0.3

        # Draw limb connections
        for kp_a, kp_b, side in SKELETON_CONNECTIONS:
            xa, ya, ca = keypoints[kp_a]
            xb, yb, cb = keypoints[kp_b]
            if ca > min_conf and cb > min_conf:
                color = _SIDE_COLORS.get(side, (200, 200, 200))
                cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2)

        # Draw keypoint dots
        for i, (x, y, c) in enumerate(keypoints):
            if c > min_conf:
                cv2.circle(frame, (int(x), int(y)), 3, (255, 255, 255), -1)

    def _draw_gaze_line(self, frame: np.ndarray, pose: PoseData) -> None:
        """Draw a 50px line from the nose in the head_yaw direction."""
        if pose.keypoints is None or len(pose.keypoints) < 1:
            return

        nx, ny, nc = pose.keypoints[KEYPOINT_NOSE]
        if nc < 0.3 or pose.head_yaw is None:
            return

        # head_yaw: 0° = right, 90° = camera, 180° = left
        # Convert to radians for line drawing (0° = right in image coords)
        angle_rad = math.radians(pose.head_yaw)
        length = 50
        ex = int(nx + length * math.cos(angle_rad))
        ey = int(ny - length * math.sin(angle_rad))  # y-axis inverted in image

        cv2.arrowedLine(
            frame, (int(nx), int(ny)), (ex, ey),
            (0, 255, 255), 2, tipLength=0.3,
        )

    def _draw_hazard_indicator(self, frame: np.ndarray, hazard: Any) -> None:
        """Draw hazard state info: coloured border, dwell timer, state label."""
        if hazard.hazard_bbox is None:
            return

        x1, y1, x2, y2 = hazard.hazard_bbox
        color = _STATE_COLORS.get(hazard.state, (200, 200, 200))
        thickness = 2

        # Pulsing effect for ESCALATED state
        if hazard.state == HazardState.ESCALATED:
            pulse = int(3 + 2 * abs(math.sin(time.time() * 4)))
            thickness = pulse

        cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), color, thickness)

        # Dwell timer label below bbox
        if hazard.dwell_seconds > 0.5:
            dwell_label = f"{hazard.dwell_seconds:.1f}s"
            cv2.putText(
                frame, dwell_label,
                (x1, y2 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

        # State label above bbox
        state_label = hazard.state.value.upper()
        cv2.putText(
            frame, state_label,
            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )

    def _draw_alert_banner(self, frame: np.ndarray, alert: Any) -> None:
        """Draw a full-width coloured banner at the top of the frame."""
        h, w = frame.shape[:2]
        color = self._severity_color(alert.severity)

        # Semi-transparent banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, self.banner_height), color, -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # White text centred in the banner
        text = alert.message
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        (tw, th), _ = cv2.getTextSize(text, font, scale, 2)
        tx = (w - tw) // 2
        ty = (self.banner_height + th) // 2
        cv2.putText(frame, text, (tx, ty), font, scale, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_fps(self, frame: np.ndarray, fps: float) -> None:
        """Draw FPS counter in the top-right corner."""
        h, w = frame.shape[:2]
        label = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        x = w - tw - 10
        y = 20
        # Background rectangle for readability
        cv2.rectangle(frame, (x - 4, y - th - 4), (x + tw + 4, y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (x, y), font, scale, (0, 255, 0), 1, cv2.LINE_AA)

    # ── Helpers ─────────────────────────────────────────────

    @staticmethod
    def _severity_color(severity: Severity) -> tuple[int, int, int]:
        """Map severity to a BGR colour tuple."""
        return {
            Severity.INFO: (180, 180, 0),       # Teal-ish
            Severity.WARNING: (0, 200, 255),     # Orange
            Severity.DANGER: (0, 80, 255),       # Red-Orange
            Severity.CRITICAL: (0, 0, 255),      # Red
        }.get(severity, (200, 200, 200))
