import numpy as np
import pytest
from core.depth_estimator import DepthEstimator

def test_depth_estimator_distance_calculation():
    # Mock configuration
    config = {
        "enabled": True,
        "path": "mock-path",
        "scale_factor": 15.0,
        "run_every_n_frames": 3
    }
    
    # Instantiate DepthEstimator without loading weights
    estimator = DepthEstimator.__new__(DepthEstimator)
    estimator.config = config
    estimator.enabled = True
    estimator.scale_factor = 15.0
    
    # Create a mock depth map (inverse depth, e.g. values 2.0 to 5.0)
    # Shape 480x640
    depth_map = np.ones((480, 640), dtype=np.float32) * 3.0
    
    # Mock a region with depth 5.0
    depth_map[100:200, 100:200] = 5.0
    
    # Bbox completely inside depth 5.0 region
    # get_distance returns scale_factor / median_depth
    # 15.0 / 5.0 = 3.0 meters
    dist = estimator.get_distance((105, 105, 195, 195), depth_map)
    assert dist == 3.0
    
    # Bbox completely inside depth 3.0 region
    # 15.0 / 3.0 = 5.0 meters
    dist_far = estimator.get_distance((10, 10, 50, 50), depth_map)
    assert dist_far == 5.0
    
    # Out of bounds bbox should clip gracefully
    dist_oob = estimator.get_distance((-50, -50, 1000, 1000), depth_map)
    assert dist_oob is not None
