"""Bridge between earworm perception and samplebank's sample library."""

from earworm.bridge.samplebank import SamplebankBridge
from earworm.bridge.models import SampleMatch
from earworm.bridge.enrich import enrich_voice_with_samples

__all__ = ["SamplebankBridge", "SampleMatch", "enrich_voice_with_samples"]
