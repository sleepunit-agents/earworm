"""Voice interpretation orchestrator.

Runs the analysis pipeline (or accepts pre-computed results),
builds the LLM prompt, calls the provider, and parses the response
into a VoiceResult.
"""

from __future__ import annotations

import json
from pathlib import Path

from earworm.models import Layer1Features, Layer2Features, Layer3Features, VoiceResult
from earworm.voice.prompt import build_prompt
from earworm.voice.provider import LLMProvider, get_provider


def interpret(
    layer1: Layer1Features,
    layer2: Layer2Features | None = None,
    layer3: Layer3Features | None = None,
    mode: str = "quick",
    provider: LLMProvider | None = None,
) -> VoiceResult:
    """Interpret pipeline results into natural language.

    Args:
        layer1: Layer 1 signal features (required).
        layer2: Layer 2 structural features (optional but recommended).
        layer3: Layer 3 quality features (optional but recommended).
        mode: "quick" for 2-3 sentence take, "deep" for full walkthrough.
        provider: LLM provider to use. Defaults to env-configured provider.

    Returns:
        VoiceResult with natural language interpretation.
    """
    if provider is None:
        provider = get_provider()

    system, prompt = build_prompt(layer1, layer2, layer3, mode=mode)
    raw = provider.generate(system, prompt)

    return parse_voice_response(raw, file_path=layer1.file_path, mode=mode)


def interpret_from_file(
    path: str | Path,
    mode: str = "quick",
    provider: LLMProvider | None = None,
) -> VoiceResult:
    """Run the full pipeline on an audio file and interpret the results.

    Convenience function that runs Layers 1-3 then Voice interpretation.
    """
    from earworm.pipeline import analyze_layer1, analyze_layer2, analyze_layer3

    path = Path(path)
    layer1 = analyze_layer1(path)
    layer2 = analyze_layer2(path, layer1=layer1)
    layer3 = analyze_layer3(path, layer1=layer1, layer2=layer2)

    return interpret(layer1, layer2, layer3, mode=mode, provider=provider)


def parse_voice_response(raw: str, file_path: str, mode: str) -> VoiceResult:
    """Parse the LLM's JSON response into a VoiceResult.

    Handles common LLM output quirks (markdown fences, trailing text).
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = text.index("\n")
        text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith("```"):
            text = text[:-3].rstrip()

    data = json.loads(text)

    return VoiceResult(
        file_path=file_path,
        mode=mode,
        description=data["description"],
        opinion=data["opinion"],
        tags=data.get("tags", []),
        comparisons=data.get("comparisons", []),
        highlights=data.get("highlights", []),
        concerns=data.get("concerns", []),
        section_notes=data.get("section_notes"),
    )
