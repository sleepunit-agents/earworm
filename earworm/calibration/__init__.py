"""Phase 3: Calibration — feedback loop for developing perception and taste.

Collects tracks with known human descriptions, runs them through the pipeline,
compares what the pipeline captured against what humans noticed, and tracks
where perception diverges from consensus.
"""

from earworm.calibration.corpus import Corpus
from earworm.calibration.runner import CalibrationRunner
from earworm.calibration.alignment import AlignmentChecker

__all__ = ["Corpus", "CalibrationRunner", "AlignmentChecker"]
