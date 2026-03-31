"""Voice — Phase 2 interpretation layer.

Takes structured pipeline output from Layers 1-3 and produces
natural language descriptions, opinions, and tags via LLM.
"""

from earworm.voice.interpret import interpret, interpret_from_file

__all__ = ["interpret", "interpret_from_file"]
