"""Compute device selection with macOS 26 Tahoe workaround.

On macOS 26 (Tahoe), PyTorch's MPS backend has a known initialization
regression. This module runs a smoke test on startup and falls back
to CPU automatically if MPS fails.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device(preferred: str = "auto") -> str:
    """Select the best available compute device.

    Args:
        preferred: "auto" (try MPS then CPU), "mps", or "cpu".

    Returns:
        Device string compatible with Ultralytics' ``device`` parameter.
    """
    import torch

    if preferred not in ("auto", "mps", "cpu"):
        logger.warning("Unknown device '%s', falling back to auto", preferred)
        preferred = "auto"

    if preferred == "cpu":
        logger.info("Device forced to CPU by configuration")
        return "cpu"

    # Attempt MPS (Apple Silicon Metal Performance Shaders)
    if preferred in ("auto", "mps"):
        mps_available = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if mps_available:
            try:
                # Smoke test — catches macOS 26 Tahoe regression where
                # MPS reports as available but operations fail.
                t = torch.zeros(2, 3, device="mps")
                _ = (t * 2 + 1).sum().item()  # Force real computation
                del t
                logger.info(
                    "✅ MPS device passed smoke test — using Apple Silicon GPU"
                )
                return "mps"
            except Exception as e:
                logger.warning(
                    "MPS available but smoke test failed "
                    "(likely macOS 26 Tahoe regression): %s — falling back to CPU",
                    e,
                )
        else:
            if preferred == "mps":
                logger.warning("MPS explicitly requested but not available")
            else:
                logger.info("MPS not available on this system")

    logger.info("Using CPU for inference")
    return "cpu"


def log_device_info() -> None:
    """Log detailed device/platform information for debugging."""
    import platform
    import torch

    logger.info("Platform: %s %s", platform.system(), platform.release())
    logger.info("Python: %s", platform.python_version())
    logger.info("PyTorch: %s", torch.__version__)
    logger.info(
        "MPS available: %s",
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available(),
    )
    try:
        import ultralytics
        logger.info("Ultralytics: %s", ultralytics.__version__)
    except ImportError:
        logger.warning("Ultralytics not installed")
