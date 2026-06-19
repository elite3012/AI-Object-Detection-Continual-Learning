"""Adaptive few-shot image classification components."""

from .adaptive_service import AdaptiveVisionService
from .prototype_memory import PrototypeMemory

__all__ = ["AdaptiveVisionService", "PrototypeMemory"]
