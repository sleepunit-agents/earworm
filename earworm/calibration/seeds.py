"""Seed data for the calibration corpus.

These are the initial calibration tracks with human descriptions
sourced from music criticism, reviews, and community discussion.
"""

from __future__ import annotations

from earworm.calibration.corpus import Corpus
from earworm.calibration.models import HumanDescription


def seed_corpus(corpus: Corpus) -> None:
    """Populate the corpus with initial calibration tracks and human descriptions."""

    _seed_born_under_punches(corpus)
    _seed_xpander(corpus)
    corpus.save()


def _seed_born_under_punches(corpus: Corpus) -> None:
    """Born Under Punches (The Heat Goes On) — Talking Heads.

    From Remain in Light (1980). The definitive polyrhythmic art-rock track.
    Dense, layered, African-influenced, anxious. A benchmark for what the
    pipeline should detect in complex rhythmic and structural music.
    """
    entry = corpus.add_track(
        artist="Talking Heads",
        title="Born Under Punches (The Heat Goes On)",
        album="Remain in Light",
        year=1980,
    )

    entry.human_descriptions = [
        HumanDescription(
            source="critical-consensus",
            text=(
                "Born Under Punches opens Remain in Light with a dense, polyrhythmic "
                "assault. Multiple interlocking guitar parts, Tina Weymouth's insistent "
                "bass line, and Chris Frantz's Afrobeat-influenced drums create a "
                "hypnotic groove that David Byrne's anxious, fractured vocals ride over. "
                "The production by Brian Eno layers textures until the track feels like "
                "a living organism — parts entering and exiting, building tension without "
                "conventional verse-chorus resolution. The track is simultaneously "
                "danceable and deeply unsettling."
            ),
            tags=[
                "art-rock", "post-punk", "new wave", "afrobeat",
                "polyrhythmic", "anxious", "hypnotic", "dense",
                "experimental", "funk",
            ],
            key_observations=[
                "Polyrhythmic structure with interlocking guitar parts",
                "Afrobeat-influenced drumming — not standard rock rhythm",
                "No conventional verse-chorus structure — evolving form",
                "Brian Eno production — dense layering, textures phase in/out",
                "Tension builds throughout without conventional release",
                "Anxious, paranoid vocal delivery over danceable groove",
                "Bass line is hypnotic and repetitive — the anchor",
            ],
        ),
        HumanDescription(
            source="allmusic",
            text=(
                "The album's opening track epitomizes the Remain in Light approach: "
                "African-inspired polyrhythms, layered guitars that function as "
                "percussion, and Byrne's vocals treated as another rhythmic element "
                "rather than a melodic lead. The production is spacious yet dense — "
                "every frequency occupied but nothing cluttered."
            ),
            tags=[
                "art-rock", "new wave", "afrobeat", "experimental",
                "funky", "paranoid", "tense",
            ],
            key_observations=[
                "Guitars function as percussion, not melody",
                "Vocals treated as rhythmic element",
                "Production is spacious yet dense — fullness without mud",
                "African-inspired polyrhythms are the foundation",
            ],
        ),
        HumanDescription(
            source="rym-community",
            text=(
                "This track is the sound of organized chaos. The groove is relentless "
                "but the layers keep shifting — you can listen ten times and hear "
                "something new each time. Byrne sounds like he's having a panic attack "
                "over a funk groove. The syncopation is wild, nothing lands where you "
                "expect it. It's funky and terrifying at the same time."
            ),
            tags=[
                "art-rock", "funk", "post-punk", "frantic",
                "syncopated", "layered", "dark",
            ],
            key_observations=[
                "Organized chaos — groove is relentless but layers shift",
                "High replay value from layered complexity",
                "Syncopation is extreme — nothing lands on expected beats",
                "Simultaneously funky and terrifying",
            ],
        ),
    ]


def _seed_xpander(corpus: Corpus) -> None:
    """Xpander — Sasha.

    From the Xpander EP (1999). The definitive progressive trance/UK techno track.
    Layered architecture, four distinct section types, euphoric build. A benchmark
    for what the pipeline should detect in progressive electronic structure.
    """
    entry = corpus.add_track(
        artist="Sasha",
        title="Xpander",
        album="Xpander EP",
        year=1999,
    )

    entry.human_descriptions = [
        HumanDescription(
            source="critical-consensus",
            text=(
                "Xpander is Sasha's defining statement — a near seven-minute arc of "
                "progressive trance architecture that transformed club culture in 1999. "
                "A four-on-the-floor foundation anchors the track as distinct layers "
                "enter and exit in four identifiable structural phases. The melody is "
                "deceptively simple but emotionally devastating when the full arrangement "
                "locks in around the midpoint. The production is dense but transparent — "
                "low end is controlled, the mid-range carries emotional weight, and the "
                "high-frequency shimmer gives the track its euphoric quality. Every layer "
                "changes the meaning of all the others."
            ),
            tags=[
                "progressive trance", "uk techno", "electronic",
                "euphoric", "building", "layered", "emotional",
                "dark", "four-on-the-floor", "transcendent",
            ],
            key_observations=[
                "Four distinct structural phases — not just A/B, genuine evolution",
                "Progressive architecture — each layer recontextualizes what came before",
                "Melody enters mid-track and builds to euphoric peak",
                "Dense but transparent production — all elements have space",
                "Low end is controlled and punchy — not overwhelming",
                "High-frequency shimmer creates euphoric quality",
                "~6:46 duration — compact but complete arc",
            ],
        ),
        HumanDescription(
            source="electronic-community",
            text=(
                "This is what progressive trance is supposed to be. Sasha builds the "
                "track in actual phases — the second half doesn't just repeat the first "
                "half louder. Each new element earns its place and changes the texture of "
                "everything underneath it. The kick drives relentlessly but the space "
                "around it keeps shifting. The euphoria at the peak doesn't feel forced — "
                "it's the logical conclusion of everything that came before. Sasha understood "
                "that transcendence requires architecture."
            ),
            tags=[
                "trance", "progressive", "electronic", "uk techno",
                "euphoric", "structured", "building", "emotional",
            ],
            key_observations=[
                "Four identifiable section types (not just verse/chorus)",
                "Progressive — second half genuinely different from first",
                "Each new layer earns its place structurally",
                "Euphoric peak feels earned, not arbitrary",
                "Transcendence through architecture, not just volume",
            ],
        ),
        HumanDescription(
            source="production-analysis",
            text=(
                "The production on Xpander is a study in frequency discipline. The kick "
                "and sub-bass are clearly separated, the main bassline occupies the "
                "low-mids without masking, and the melodic content lives in the upper-mids "
                "and highs. The stereo field expands gradually — tighter in the opening "
                "sections, wide and immersive at the peak. The mastering is notably hot "
                "for 1999 — some limiter activity is audible on heavy transients — but the "
                "dynamic range is preserved well enough to serve the track's emotional arc. "
                "The mix has a heavy low-end bias that suits the dancefloor context."
            ),
            tags=[
                "trance", "electronic", "dense", "hot-mastered",
                "low-heavy", "stereo-expanding",
            ],
            key_observations=[
                "Kick and sub-bass clearly separated in frequency",
                "Stereo field expands from narrow (intro) to wide (peak)",
                "Hot mastering for the era — limiter artifacts on transients",
                "Heavy low-end bias — spectral balance tilted toward bass",
                "Dynamic range mostly preserved despite hot master",
                "Mix density increases through four structural phases",
            ],
        ),
    ]
