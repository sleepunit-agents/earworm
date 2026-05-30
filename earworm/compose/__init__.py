"""Earworm compose — structural response pipeline.

Takes a Layer 2 analysis and generates a responding WAV composition using
pytheory. The composition responds to the source track's structural DNA
(energy arc, phrase structure, section count) without imitating it.
"""

from earworm.compose.composer import ComposeManifest, compose, compose_generative
from earworm.compose.strudel import to_strudel, to_strudel_generative

__all__ = [
    "compose",
    "compose_generative",
    "ComposeManifest",
    "to_strudel",
    "to_strudel_generative",
]
