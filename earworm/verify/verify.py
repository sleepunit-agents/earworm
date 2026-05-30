"""Core sample verification logic.

Runs Layer 1 features against category profiles and produces a VerifyResult.
Optionally suggests the best-matching category when the labeled one doesn't fit.
"""

from __future__ import annotations

from earworm.models import (
    CategoryScore,
    CheckResult,
    Layer1Features,
    VerifyResult,
)
from earworm.verify.profiles import get_profile, known_categories, normalize_category


# Verdict thresholds
MATCH_THRESHOLD = 0.65
MISMATCH_THRESHOLD = 0.40


def _run_checks(
    features: Layer1Features, category: str
) -> tuple[list[CheckResult], float]:
    """Run all checks for a category and return (results, weighted_score)."""
    profile = get_profile(category)
    if profile is None:
        return [], 0.0

    results: list[CheckResult] = []
    weighted_sum = 0.0
    weight_total = 0.0

    for check in profile:
        passed, detail = check.test(features)
        results.append(
            CheckResult(
                name=check.name,
                passed=passed,
                weight=check.weight,
                detail=detail,
            )
        )
        weighted_sum += check.weight * (1.0 if passed else 0.0)
        weight_total += check.weight

    score = weighted_sum / weight_total if weight_total > 0 else 0.0
    return results, score


def _suggest_best(features: Layer1Features, exclude: str) -> CategoryScore | None:
    """Score all categories and return the best-fit (excluding the labeled one)."""
    best: CategoryScore | None = None

    for cat in known_categories():
        if cat == exclude:
            continue
        profile = get_profile(cat)
        if profile is None:
            continue

        checks, score = _run_checks(features, cat)
        passed = sum(1 for c in checks if c.passed)

        candidate = CategoryScore(
            category=cat,
            score=round(score, 3),
            checks_passed=passed,
            checks_total=len(checks),
        )
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def verify(
    features: Layer1Features,
    category: str,
    *,
    suggest: bool = True,
) -> VerifyResult:
    """Verify a sample's features against a category profile.

    Args:
        features: Layer 1 analysis of the sample.
        category: Labeled category (e.g. "kick", "hihat", "bass").
        suggest: If True and verdict is "mismatch", suggest best-fit category.

    Returns:
        VerifyResult with score, verdict, per-check details, and optional suggestion.
    """
    canonical = normalize_category(category)
    profile = get_profile(canonical)

    if profile is None:
        return VerifyResult(
            file_path=features.file_path,
            labeled_category=category,
            canonical_category=canonical,
            score=0.0,
            verdict="unknown_category",
            checks=[],
            summary=f"Unknown category '{category}' (canonical: '{canonical}'). "
            f"Known categories: {', '.join(known_categories())}",
        )

    checks, score = _run_checks(features, canonical)
    score = round(score, 3)

    if score >= MATCH_THRESHOLD:
        verdict = "match"
    elif score <= MISMATCH_THRESHOLD:
        verdict = "mismatch"
    else:
        verdict = "uncertain"

    passed = sum(1 for c in checks if c.passed)
    total = len(checks)

    summary = (
        f"{'Consistent with' if verdict == 'match' else 'Inconsistent with' if verdict == 'mismatch' else 'Uncertain match for'}"
        f" {canonical} profile (score: {score:.2f}, {passed}/{total} checks passed)"
    )

    suggestion = None
    if suggest and verdict == "mismatch":
        suggestion = _suggest_best(features, exclude=canonical)
        if suggestion and suggestion.score >= MATCH_THRESHOLD:
            summary += f" — may actually be: {suggestion.category} (score: {suggestion.score:.2f})"

    return VerifyResult(
        file_path=features.file_path,
        labeled_category=category,
        canonical_category=canonical,
        score=score,
        verdict=verdict,
        checks=checks,
        summary=summary,
        suggestion=suggestion,
    )


def verify_all_categories(
    features: Layer1Features,
) -> list[CategoryScore]:
    """Score a sample against every known category.

    Returns a list sorted by score (highest first). Useful for suggesting
    what a completely unlabeled sample might be.
    """
    scores: list[CategoryScore] = []
    for cat in known_categories():
        checks, score = _run_checks(features, cat)
        passed = sum(1 for c in checks if c.passed)
        scores.append(
            CategoryScore(
                category=cat,
                score=round(score, 3),
                checks_passed=passed,
                checks_total=len(checks),
            )
        )
    scores.sort(key=lambda s: s.score, reverse=True)
    return scores
