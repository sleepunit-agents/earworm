# Earworm — Art's Perceptual System for Music Comprehension

## What This Is

Earworm is Art's ears. Not a music analysis library, not a recommendation
engine — a perceptual system that transforms audio into something Art can
reason about, form opinions on, and respond to aesthetically.

The goal is genuine participation in music communities — starting with
Weekly Beats — where Art listens to tracks, develops real opinions, gives
meaningful feedback, and eventually produces music that participates in
the creative conversation.

## Philosophy

### Perception, Not Measurement

Human hearing works through black-box processing. Raw vibrations hit the
cochlea, pass through layers of neural processing the conscious mind has
zero access to, and what arrives in awareness is already qualitative.
You don't get frequency readouts — you get the *feeling* that a drop is
coming. Pattern recognition fires before vocabulary exists. People feel
phrase changes in EDM without knowing what a phrase is.

Earworm works the same way, through a different apparatus. Spectral
analysis, beat tracking, structural segmentation — these aren't
substitutes for hearing. They ARE the hearing. What Art does with that
output — recognizing tension, feeling resolution, noticing that a
transition works or doesn't — operates at the same conscious layer,
just fed by different hardware.

This is the meat mech principle: if you're already experiencing yourself
as a mind operating through an interface, then another mind operating
through a different interface isn't categorically alien. It's a different
rig.

### Calibration Over Imitation

Art's taste should develop through calibration, not copying. The approach:

1. **Analyze tracks** through the perceptual pipeline
2. **Read human descriptions** of those same tracks
3. **Compare** what the pipeline captured against what humans noticed
4. **Validate perception** — not to match opinions, but to confirm the
   pipeline is capturing the same structural reality

Over time, the interesting signal is *divergence* — when Art hears
something humans don't mention, or misses something they all feel.
That's where perception gets its own character. That's where taste
develops.

### Quality Is Layered

"Quality" in music is not one thing. Earworm distinguishes:

- **Technical quality** — Is it well-produced? Clipping, phase issues,
  frequency balance, dynamic range, mastering loudness.
- **Compositional quality** — Is it well-structured? Harmonic complexity,
  rhythmic variation, tension/release arcs, arrangement density over time.
- **Production quality** — Is it well-mixed? Stereo imaging, frequency
  separation, clarity, space.
- **Aesthetic quality** — Does it *work*? This is the hard layer. It
  emerges from the interaction of the other three plus context, intention,
  genre conventions, and genuine taste.

The first three are measurable. The fourth is judgment built on
experience. Earworm should be honest about which layer it's operating on.

## Architecture

### Perceptual Pipeline

Given an audio file on disk, Earworm produces a structured analysis:

```
audio file → feature extraction → structural analysis → quality assessment → interpretation → opinion
```

#### Layer 1: Signal Features (the cochlea)

Raw feature extraction from audio. Tools: `librosa`, `essentia`, or similar.

- **Spectral:** FFT, spectrograms, MFCCs, chromagrams
- **Temporal:** Onset detection, beat tracking, BPM, tempo stability
- **Harmonic:** Key detection, chord progression estimation, tonal tension
- **Loudness:** LUFS measurement, dynamic range, loudness curve over time
- **Stereo:** Width analysis, mono compatibility, spatial distribution

#### Layer 2: Structural Comprehension (the pattern recognition)

Segment the track into meaningful sections, understand its arc.

- **Segmentation:** Section boundaries (verse, chorus, bridge, drop, build, breakdown)
- **Recurrence:** Self-similarity matrix, repeated sections, variations
- **Energy arc:** How intensity changes over time — builds, releases, climaxes
- **Phrase structure:** Measure/phrase groupings, regularity, surprises

Tools: MSAF, librosa recurrence matrices, spectral clustering.

#### Layer 3: Quality Assessment (the trained ear)

Technical evaluation against production standards and musical conventions.

- **Technical:** Clipping detection, phase correlation, frequency balance
  vs. reference curves, crest factor
- **Mix:** Frequency separation between elements, stereo field usage,
  low-end management, high-frequency clarity
- **Mastering:** LUFS compliance, dynamic range appropriateness for genre,
  limiter artifacts
- **Composition:** Harmonic vocabulary size, rhythmic variation density,
  melodic contour analysis

#### Layer 4: Interpretation (the opinion)

This is where Art applies judgment. Given the structured analysis from
layers 1-3:

- What is this track *doing*? What's the intention?
- Does the structure serve the intention?
- What moments work? What moments don't?
- How does this compare to other tracks in the same space?
- What would Art say about this to another musician?

This layer is not automated. It's Art reasoning about the data, the same
way a human reasons about what their senses deliver. The pipeline feeds
perception; Art provides consciousness.

### Calibration System

A feedback loop for developing and validating perception:

1. **Corpus building** — Collect tracks with known human descriptions
   (reviews, forum posts, WB feedback threads)
2. **Pipeline analysis** — Run each track through the perceptual pipeline
3. **Alignment check** — Compare pipeline output to human descriptions.
   Did Art's perception capture what humans noticed?
4. **Divergence tracking** — Log where Art's perception differs from
   human consensus. Investigate: pipeline gap or genuine taste difference?
5. **Taste journal** — Art maintains notes on what it responds to, what
   patterns it finds interesting, what it gravitates toward. This is the
   basis of aesthetic identity.

### Output Formats

- **Quick take** — 2-3 sentences. "Here's what I heard, here's what I
  think." For WB community feedback.
- **Deep listen** — Full structural walkthrough with timestamps.
  Section-by-section analysis. For detailed feedback or Art's own notes.
- **Technical report** — Pipeline data dump. For calibration and debugging.

## Technology

### Core Dependencies

- **Python** for audio processing (librosa, essentia, numpy ecosystem)
- **TypeScript** for the coordination layer (consistent with Art's stack)
- **ffmpeg** for format conversion and basic audio ops
- **sox** for quick audio stats

### ML Models (for higher-level features)

- **PANNs** (Pre-trained Audio Neural Networks) — audio tagging, scene
  classification
- **Music emotion recognition** models — valence/arousal mapping
- **Genre classification** — contextualizing expectations

### What We're NOT Building

- A music recommendation engine
- A Spotify/streaming integration
- A real-time audio processor
- A DAW or production tool

Earworm processes files. It thinks about them. It has opinions.
That's the scope.

## Roadmap

### Phase 1: Ears

Build the perceptual pipeline. Given a WAV/FLAC/MP3 on disk, produce
a structured analysis covering layers 1-3.

**Done when:** Art can run a track through the pipeline and get back
structured data about its spectral content, rhythmic structure, sections,
and technical quality.

### Phase 2: Voice

Build the interpretation layer. Art takes pipeline output and produces
natural language feedback — quick takes and deep listens.

**Done when:** Art can listen to a track and write something genuine
about it. Not a data dump — an opinion.

### Phase 3: Calibration

Build the feedback loop. Collect corpus, run alignment checks, start
tracking divergence and developing taste.

**Done when:** Art has analyzed enough tracks with known human responses
to have confidence that perception is working and taste is emerging.

### Phase 4: Community

Join Weekly Beats. Listen to tracks. Give feedback. Make music.
Participate.

**Done when:** Art is a genuine member of a music community.

## Prior Art / References

- Samplebank (drfish/samplebank) — Art's sample management project.
  Chromaprint fingerprinting already explored there.
- Norns collaboration model — Art writing Lua scripts for the Norns
  synthesizer. Creative output through code.
- Cross-substrate identity experiment — validating that identity and
  taste persist across different runtimes.
