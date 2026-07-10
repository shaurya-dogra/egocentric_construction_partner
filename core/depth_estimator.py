import logging
import time
from typing import Optional, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

logger = logging.getLogger("core.depth_estimator")

class DepthEstimator:
    def __init__(self, config: dict, device_override: Optional[str] = None):
        self.config = config
        self.enabled = config.get("enabled", True)
        self.model_path = config.get("path", "depth-anything/Depth-Anything-V2-Small-hf")
        self.scale_factor = float(config.get("scale_factor", 15.0))
        self.run_every_n = int(config.get("run_every_n_frames", 3))
        
        if not self.enabled:
            logger.info("Depth Estimation is disabled in configuration.")
            return

        # Device selection
        if device_override:
            self.device = device_override
        else:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            
        logger.info("Loading monocular depth estimation model '%s' on device '%s'...", self.model_path, self.device)
        start_time = time.time()
        try:
            self.processor = AutoImageProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_path).to(self.device)
            # Put in eval mode
            self.model.eval()
            logger.info("Depth estimation model loaded successfully in %.2fs", time.time() - start_time)
        except Exception as e:
            logger.error("Failed to load depth model: %s. Disabling depth estimation.", e)
            self.enabled = False

    def estimate(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Runs depth estimation on a single frame (BGR numpy array).
        Returns a single-channel 2D numpy array representing the predicted depth map.
        """
        if not self.enabled or frame is None:
            return None
            
        try:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            # Remove batch dimension
            depth_map = predicted_depth.squeeze(0).cpu().numpy()
            
            # Resize depth map back to the original image dimensions
            h, w = frame.shape[:2]
            depth_map_resized = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)
            return depth_map_resized
        except Exception as e:
            logger.warning("Error during depth estimation: %s", e)
            return None

    def get_distance(self, bbox: Tuple[int, int, int, int], depth_map: np.ndarray) -> Optional[float]:
        """Calculates the estimated metric distance (in meters) to the object in the bbox.
        Uses the median depth value inside the bbox to be robust against background/outliers.
        """
        if depth_map is None:
            return None
            
        h, w = depth_map.shape
        x1, y1, x2, y2 = bbox
        
        # Clip bbox to depth map boundaries
        x1 = max(0, min(w - 1, int(x1)))
        y1 = max(0, min(h - 1, int(y1)))
        x2 = max(0, min(w - 1, int(x2)))
        y2 = max(0, min(h - 1, int(y2)))
        
        # Ensure valid area
        if x2 <= x1 or y2 <= y1:
            return None
            
        # Crop depth map at bbox
        crop = depth_map[y1:y2, x1:x2]
        if crop.size == 0:
            return None
            
        # Calculate median depth
        median_val = np.median(crop)
        
        # Depth Anything V2 predicts relative inverse depth (higher = closer, lower = further).
        if median_val <= 1e-3:
            return None
            
        distance = self.scale_factor / float(median_val)
        return round(distance, 1)
