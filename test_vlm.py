import logging
import unittest
from unittest.mock import patch

import yaml

from core.models import Detection, FrameResult, HazardAssessment, HazardState, Severity
from integration.vlm_hook import VLMHook


logging.basicConfig(level=logging.INFO)


class VLMHookOfflineTests(unittest.TestCase):
    def setUp(self) -> None:
        with open("config.yaml") as f:
            config = yaml.safe_load(f)
        self.vlm = VLMHook(config["vlm"])
        self.detections = [
            Detection(bbox=(100, 100, 200, 200), class_name="person", confidence=0.9, track_id=1),
            Detection(bbox=(300, 300, 400, 400), class_name="car", confidence=0.8, track_id=2),
            Detection(bbox=(150, 150, 250, 250), class_name="truck", confidence=0.85, track_id=3),
            Detection(bbox=(50, 50, 80, 80), class_name="hardhat", confidence=0.95),
        ]

    def tearDown(self) -> None:
        self.vlm.stop()

    def test_ask_question_uses_structured_context(self) -> None:
        with patch.object(self.vlm, "_run_chat_loop", return_value="Two vehicles are visible.") as mock_run:
            answer = self.vlm.ask_question("How many vehicles do you see?", self.detections)
        self.assertEqual(answer, "Two vehicles are visible.")
        self.assertTrue(mock_run.called)

    def test_escalate_to_reasoning_accepts_current_hazard_schema(self) -> None:
        hazard = HazardAssessment(
            hazard_id="hz-test-1",
            hazard_type="vehicle_proximity",
            severity=Severity.DANGER,
            description="Worker is close to a truck.",
            worker_track_id=1,
            hazard_bbox=(100, 100, 220, 220),
            state=HazardState.ESCALATED,
            is_escalated=True,
        )
        frame_result = FrameResult(frame_number=1, detections=self.detections)
        with patch.object(self.vlm, "_run_chat_loop", return_value="Vehicle proximity confirmed.") as mock_run:
            answer = self.vlm.escalate_to_reasoning(
                frame=None,
                detections=self.detections,
                hazard_context={"hazard": hazard},
                frame_result=frame_result,
            )
        self.assertEqual(answer, "Vehicle proximity confirmed.")
        self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()
