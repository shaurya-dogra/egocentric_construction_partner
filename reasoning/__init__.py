"""Tier 2 reasoning package."""

__all__ = ["ReasoningCoordinator"]


def __getattr__(name: str):
    if name == "ReasoningCoordinator":
        from reasoning.pipeline import ReasoningCoordinator

        return ReasoningCoordinator
    raise AttributeError(name)
