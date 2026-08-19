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
    # Persons & Vehicles
    "person": (0, 255, 0),
    "car": (255, 128, 0),
    "truck": (255, 128, 0),
    "forklift": (255, 128, 0),
    "excavator": (255, 140, 0),
    "bulldozer": (255, 140, 0),
    "backhoe_loader": (255, 140, 0),
    "skid_steer_loader": (255, 140, 0),
    "wheel_loader": (255, 140, 0),
    "road_roller": (255, 140, 0),
    "scissor_lift": (255, 140, 0),
    "boom_lift": (255, 140, 0),
    "concrete_pump_truck": (255, 128, 0),
    "flatbed_trailer": (255, 128, 0),
    "crane": (255, 128, 0),
    "tower_crane": (255, 128, 0),
    "mobile_crane": (255, 128, 0),
    "cement_mixer": (255, 128, 0),
    "concrete_mixer": (255, 128, 0),
    "cement_mixer_truck": (255, 128, 0),
    "dump_truck": (255, 128, 0),
    "pickup_truck": (255, 128, 0),
    "delivery_truck": (255, 128, 0),
    "machinery": (255, 128, 0),
    "utility_pole": (255, 128, 0),

    # PPE OK / Missing
    "hardhat": (0, 200, 200),
    "hard_hat": (0, 200, 200),
    "hard_hat_with_attached_radio": (0, 200, 200),
    "safety_helmet": (0, 200, 200),
    "no-hardhat": (0, 0, 255),
    "no_hardhat": (0, 0, 255),
    "vest": (0, 200, 200),
    "safety_vest": (0, 200, 200),
    "reflective_vest": (0, 200, 200),
    "high_visibility_jacket": (0, 200, 200),
    "no-vest": (0, 0, 255),
    "no_vest": (0, 0, 255),
    "goggles": (0, 200, 200),
    "safety_goggles": (0, 200, 200),
    "safety_glasses": (0, 200, 200),
    "face_shield": (0, 200, 200),
    "welding_helmet": (0, 200, 200),
    "no-goggles": (0, 0, 255),
    "no_goggles": (0, 0, 255),
    "gloves": (0, 200, 200),
    "work_gloves": (0, 200, 200),
    "cut_resistant_gloves": (0, 200, 200),
    "rubber_gloves": (0, 200, 200),
    "no-gloves": (0, 0, 255),
    "no_gloves": (0, 0, 255),
    "boots": (0, 200, 200),
    "safety_boots": (0, 200, 200),
    "steel_toe_boots": (0, 200, 200),
    "no-boots": (0, 0, 255),
    "no_boots": (0, 0, 255),
    "mask": (0, 200, 200),
    "dust_mask": (0, 200, 200),
    "respirator": (0, 200, 200),
    "respirator_mask": (0, 200, 200),
    "no-mask": (0, 0, 255),
    "no_mask": (0, 0, 255),
    "ear_protection": (0, 200, 200),
    "ear_muffs": (0, 200, 200),
    "ear_plugs": (0, 200, 200),
    "knee_pads": (0, 200, 200),
    "safety_harness": (0, 220, 220),
    "fall_arrest_harness": (0, 220, 220),

    # Hand & Power Tools
    "hammer": (180, 105, 255),
    "claw_hammer": (180, 105, 255),
    "sledgehammer": (180, 105, 255),
    "rubber_mallet": (180, 105, 255),
    "drill": (0, 165, 255),
    "cordless_drill": (0, 165, 255),
    "power_drill": (0, 165, 255),
    "impact_driver": (0, 165, 255),
    "saw": (147, 20, 255),
    "circular_saw": (147, 20, 255),
    "reciprocating_saw": (147, 20, 255),
    "handsaw": (147, 20, 255),
    "hand_saw": (147, 20, 255),
    "hacksaw": (147, 20, 255),
    "jigsaw": (147, 20, 255),
    "table_saw": (147, 20, 255),
    "miter_saw": (147, 20, 255),
    "chainsaw": (147, 20, 255),
    "angle_grinder": (0, 140, 255),
    "bench_grinder": (0, 140, 255),
    "rotary_hammer": (180, 105, 255),
    "rotary_hammer_drill": (180, 105, 255),
    "jackhammer": (180, 105, 255),
    "nail_gun": (0, 215, 255),
    "heat_gun": (0, 165, 255),
    "soldering_iron": (255, 69, 0),
    "welding_machine": (255, 69, 0),
    "welding_torch": (255, 69, 0),
    "belt_sander": (0, 165, 255),
    "orbital_sander": (0, 165, 255),
    "power_sander": (0, 165, 255),
    "air_compressor": (100, 149, 237),
    "pressure_washer": (100, 149, 237),
    "measuring_tape": (238, 130, 238),
    "spirit_level": (50, 205, 50),
    "level": (50, 205, 50),
    "screwdriver": (255, 191, 0),
    "phillips_screwdriver": (255, 191, 0),
    "wrench": (255, 140, 0),
    "adjustable_wrench": (255, 140, 0),
    "pipe_wrench": (255, 140, 0),
    "socket_wrench": (255, 140, 0),
    "socket_wrench_set": (255, 140, 0),
    "allen_key_set": (255, 140, 0),
    "pliers": (218, 112, 214),
    "needle_nose_pliers": (218, 112, 214),
    "wire_cutters": (218, 112, 214),
    "bolt_cutter": (218, 112, 214),
    "bolt_cutters": (218, 112, 214),
    "pipe_cutter": (218, 112, 214),
    "staple_gun": (218, 112, 214),
    "c_clamp": (218, 112, 214),
    "chisel": (255, 165, 0),
    "hand_file": (205, 133, 63),
    "putty_knife": (205, 133, 63),
    "hand_trowel": (205, 133, 63),
    "pry_bar": (205, 133, 63),
    "crowbar": (205, 133, 63),
    "utility_knife": (0, 69, 255),
    "knife": (0, 69, 255),
    "scissors": (255, 105, 180),
    "toolbox": (128, 0, 128),
    "tool_cart": (128, 0, 128),
    "storage_bin": (128, 0, 128),
    "tool": (200, 150, 255),
    "power_tool": (200, 150, 255),

    # Site Infrastructure & Access
    "step_ladder": (255, 215, 0),
    "extension_ladder": (255, 215, 0),
    "ladder": (255, 215, 0),
    "scaffolding": (255, 165, 0),
    "scaffold_platform": (255, 165, 0),
    "barricade": (0, 165, 255),
    "safety_cone": (0, 165, 255),
    "caution_tape": (0, 255, 255),
    "guardrail": (0, 165, 255),
    "warning_sign": (0, 255, 255),
    "asbestos_warning_sign": (0, 0, 255),
    "safety_barrier": (0, 165, 255),
    "temporary_railing": (0, 165, 255),
    "bucket": (255, 192, 203),
    "wheelbarrow": (205, 133, 63),
    "dumpster": (128, 128, 128),
    "dolly_cart": (205, 133, 63),
    "workbench": (139, 69, 19),
    "work_platform": (255, 165, 0),
    "shipping_container": (70, 130, 180),
    "port_a_potty": (70, 130, 180),

    # Hazards & Materials
    "exposed_wire": (0, 0, 255),
    "loose_cable": (0, 0, 255),
    "electrical_cable": (30, 144, 255),
    "extension_cord": (0, 165, 255),
    "extension_cord_reel": (0, 165, 255),
    "power_strip": (0, 165, 255),
    "electrical_panel": (255, 215, 0),
    "electrical_outlet": (255, 215, 0),
    "circuit_breaker": (255, 215, 0),
    "gas_cylinder": (0, 69, 255),
    "propane_cylinder": (0, 69, 255),
    "propane_tank": (0, 69, 255),
    "fire_extinguisher": (0, 0, 255),
    "spill": (0, 0, 255),
    "puddle": (0, 0, 255),
    "spill_puddle": (0, 0, 255),
    "chemical_container": (0, 0, 255),
    "open_manhole": (0, 0, 255),
    "sharp_metal_shard": (0, 0, 255),
    "debris_pile": (128, 128, 128),
    "sandbag": (189, 183, 107),
    "cinder_block": (169, 169, 169),
    "concrete_block": (169, 169, 169),
    "concrete_slab": (169, 169, 169),
    "brick": (178, 34, 34),
    "rebar": (192, 192, 192),
    "metal_pipe": (192, 192, 192),
    "steel_pipe": (192, 192, 192),
    "wooden_plank": (139, 69, 19),
    "lumber_stack": (139, 69, 19),
    "plywood_sheet": (139, 69, 19),
    "metal_sheet": (192, 192, 192),
    "glass_sheet": (224, 255, 255),
    "wooden_pallet": (139, 69, 19),
    "pallet": (139, 69, 19),
    "paint_can": (255, 20, 147),
    "oil_container": (255, 140, 0),

    # Common Workplace Items
    "cell_phone": (255, 255, 0),
    "two_way_radio": (255, 255, 0),
    "laptop": (0, 255, 255),
    "tablet": (0, 255, 255),
    "clipboard": (255, 215, 0),
    "notebook": (255, 255, 255),
    "pen": (255, 255, 0),
    "water_bottle": (255, 200, 100),
    "coffee_cup": (255, 200, 100),
    "bottle": (255, 200, 100),
    "cup": (255, 200, 100),
    "lunch_box": (255, 165, 0),
    "backpack": (100, 200, 255),
    "folding_chair": (180, 180, 180),
    "chair": (180, 180, 180),
    "desk": (139, 69, 19),
    "storage_cabinet": (169, 169, 169),
    "first_aid_kit": (0, 255, 0),
    "trash_can": (128, 128, 128),
    "trash_bin": (128, 128, 128),
    "broom": (210, 180, 140),
    "dustpan": (210, 180, 140),
}
_DEFAULT_CLASS_COLOR = (200, 200, 200)

_TOOL_CLASSES = frozenset({
    "hammer", "claw_hammer", "sledgehammer", "rubber_mallet",
    "drill", "cordless_drill", "power_drill", "impact_driver",
    "saw", "circular_saw", "reciprocating_saw", "handsaw", "hand_saw", "hacksaw", "jigsaw",
    "table_saw", "miter_saw", "chainsaw",
    "angle_grinder", "bench_grinder", "rotary_hammer", "rotary_hammer_drill", "jackhammer",
    "nail_gun", "heat_gun", "soldering_iron", "welding_torch",
    "belt_sander", "orbital_sander", "power_sander",
    "measuring_tape", "spirit_level", "level",
    "screwdriver", "phillips_screwdriver",
    "wrench", "adjustable_wrench", "pipe_wrench", "socket_wrench", "socket_wrench_set", "allen_key_set",
    "pliers", "needle_nose_pliers", "wire_cutters", "bolt_cutter", "bolt_cutters", "pipe_cutter", "staple_gun",
    "c_clamp", "chisel", "hand_file", "putty_knife", "hand_trowel", "pry_bar", "crowbar",
    "utility_knife", "knife", "scissors",
    "flashlight", "shovel", "toolbox", "tool_cart", "tool", "power_tool", "bucket",
    "two_way_radio", "broom", "dustpan",
})

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
        if not workers:
            return

        # Gather tools from tracked objects or active tool detections
        tool_targets: list[tuple[int, int]] = []
        for obj in result.tracked_objects.values():
            if obj.class_name in _TOOL_CLASSES and obj.is_active:
                tool_targets.append((int(obj.center[0]), int(obj.center[1])))

        # Also check detections if tracked_objects did not capture them
        if not tool_targets:
            for det in result.detections:
                if det.class_name in _TOOL_CLASSES:
                    tool_targets.append(det.center)

        if not tool_targets:
            return

        poses_by_track = {pose.person_track_id: pose for pose in result.poses}

        for worker in workers:
            pose = poses_by_track.get(worker.track_id)
            if not pose or pose.keypoints is None or len(pose.keypoints) < 11:
                continue
            for tx, ty in tool_targets:
                for index in (KEYPOINT_LEFT_WRIST, KEYPOINT_RIGHT_WRIST):
                    if index < len(pose.keypoints):
                        wrist = pose.keypoints[index]
                        if len(wrist) >= 3 and wrist[2] >= 0.2:
                            dist = math.hypot(tx - wrist[0], ty - wrist[1])
                            if dist <= 140:
                                wx, wy = int(wrist[0]), int(wrist[1])
                                # Draw a violet line from hand to tool
                                cv2.line(frame, (wx, wy), (tx, ty), (238, 130, 238), 2, cv2.LINE_AA)
                                cv2.putText(
                                    frame,
                                    "carrying",
                                    (int((wx + tx) / 2), int((wy + ty) / 2) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.4,
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
