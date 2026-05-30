"""Sample verification — use L1 audio analysis to validate sample category metadata.

Given a sample file and its labeled category (e.g. "kick", "hihat", "bass"), runs
Layer 1 feature extraction and checks the extracted features against expected profiles
for that category. Returns a VerifyResult with per-check details and an overall score.

Usage:
    from earworm.pipeline import analyze_layer1
    from earworm.verify import verify, verify_all_categories

    features = analyze_layer1("samples/kick_001.wav")
    result = verify(features, "kick")
    print(result.summary)  # "Consistent with kick profile (score: 0.85)"

    # Or identify an unknown sample:
    scores = verify_all_categories(features)
    print(scores[0].category)  # Best-matching category
"""

from earworm.verify.verify import verify, verify_all_categories

__all__ = ["verify", "verify_all_categories"]
