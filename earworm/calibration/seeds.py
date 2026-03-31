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
    """Xpander — Deadmau5.

    From 2007. A progressive/minimal techno piece that builds slowly over
    9+ minutes. Hypnotic, repetitive, evolving. A benchmark for what the
    pipeline should detect in long-form electronic builds.
    """
    entry = corpus.add_track(
        artist="Deadmau5",
        title="Xpander",
        album="",
        year=2007,
    )

    entry.human_descriptions = [
        HumanDescription(
            source="critical-consensus",
            text=(
                "Xpander is a masterclass in minimal techno patience. A hypnotic "
                "bassline establishes the foundation early, and the track spends its "
                "entire 9+ minutes adding and subtracting elements over a four-on-the-floor "
                "kick. The build is glacial but relentless — hi-hats open up, synth pads "
                "swell, a melodic motif enters around the midpoint and gradually takes over. "
                "The production is clean and spacious, with careful attention to frequency "
                "separation. It's a track designed for sustained, focused listening — or "
                "for losing yourself on a dancefloor at 3 AM."
            ),
            tags=[
                "techno", "minimal", "progressive", "electronic",
                "hypnotic", "building", "clean", "spacious",
                "four-on-the-floor", "dark",
            ],
            key_observations=[
                "9+ minute duration — long-form electronic structure",
                "Glacial build — elements added/subtracted over full runtime",
                "Four-on-the-floor kick drum is constant foundation",
                "Clean, spacious production — good frequency separation",
                "Melodic motif enters around midpoint",
                "Hypnotic bassline is the anchor",
                "Designed for sustained listening or dancefloor immersion",
            ],
        ),
        HumanDescription(
            source="electronic-community",
            text=(
                "Peak deadmau5 before the pop crossover. This is what he does best — "
                "slow, deliberate builds that reward patience. The kick never changes "
                "but the world around it transforms completely over 9 minutes. The synth "
                "work is subtle, the automation is smooth, and the climax is earned, not "
                "forced. The low end is controlled and punchy without being overwhelming."
            ),
            tags=[
                "techno", "progressive", "minimal", "electronic",
                "hypnotic", "patient", "punchy", "dark",
            ],
            key_observations=[
                "Slow deliberate build that rewards patience",
                "Kick drum constant while everything else transforms",
                "Subtle synth work, smooth automation",
                "Climax feels earned, not forced",
                "Low end is controlled and punchy",
            ],
        ),
        HumanDescription(
            source="production-analysis",
            text=(
                "From a production standpoint, Xpander demonstrates disciplined frequency "
                "management. The kick occupies the sub-bass cleanly, the bassline sits in "
                "the low-mids without masking, and the upper frequency content is introduced "
                "gradually — hi-hats first, then shimmering pads, then the lead synth. The "
                "stereo field opens up as the track progresses — early sections are narrower, "
                "and width increases with intensity. The mastering is moderate — not "
                "brickwalled, retaining dynamics appropriate for the genre."
            ),
            tags=[
                "techno", "electronic", "clean", "spacious",
                "dynamic", "well-produced",
            ],
            key_observations=[
                "Disciplined frequency management — kick/bass/mids separated",
                "Elements introduced gradually up the frequency spectrum",
                "Stereo field widens as track intensity increases",
                "Moderate mastering — not brickwalled, retains dynamics",
                "Upper content layered: hi-hats → pads → lead synth",
            ],
        ),
    ]
