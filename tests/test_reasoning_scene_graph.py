import time
import unittest

from core.models import Detection, FrameResult, TrackedObject
from reasoning.scene_graph import SceneGraphBuilder


class SceneGraphBuilderTests(unittest.TestCase):
    def test_worker_tool_relation_is_serialized(self) -> None:
        now = time.time()
        worker = TrackedObject(
            track_id=1,
            class_name="person",
            bbox=(100, 100, 200, 260),
            first_seen=now,
            last_seen=now,
        )
        worker.position_history.extend(
            [
                (120.0, 160.0, now - 0.2),
                (140.0, 170.0, now),
            ]
        )
        result = FrameResult(
            frame_number=1,
            timestamp=now,
            detections=[
                Detection(class_name="person", confidence=0.9, bbox=(100, 100, 200, 260), track_id=1),
                Detection(class_name="drill", confidence=0.8, bbox=(170, 150, 220, 220), track_id=5),
            ],
            tracked_objects={1: worker},
            poses=[],
            worker_ppe_states={},
            hazards=[],
            alerts=[],
            active_zones=[],
            fps=15.0,
        )

        builder = SceneGraphBuilder({"tool_labels": ["drill"]})
        usages = builder.infer_tool_usage(result, {1: "worker-1"})
        graph = builder.build(result, {1: "worker-1"}, usages)
        payload = builder.serialize(graph)

        relations = {(edge["source"], edge["target"], edge["relation"]) for edge in payload["edges"]}
        self.assertIn(("person:1", "drill:5", "using"), relations)


if __name__ == "__main__":
    unittest.main()
