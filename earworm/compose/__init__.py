"""Earworm compose — structural response pipeline.

Takes a Layer 2 analysis and generates a responding WAV composition using
pytheory. The composition responds to the source track's structural DNA
(energy arc, phrase structure, section count) without imitating it.
"""

from earworm.compose.composer import ComposeManifest, compose

__all__ = ["compose", "ComposeManifest"]
