# Earworm

Art's perceptual system for music comprehension. Transforms audio files into structured analysis Art can reason about.

## Stack

- Python 3.11+ for audio processing
- librosa for feature extraction
- pyloudnorm for LUFS measurement
- pydantic for data models
- pytest for testing
- ruff for linting

## Project Structure

```
earworm/
  features/    # Layer 1: Signal feature extraction
  structure/   # Layer 2: Structural comprehension
  quality/     # Layer 3: Quality assessment
  models.py    # Pydantic data models for analysis results
  pipeline.py  # Main pipeline orchestrator
  cli.py       # CLI entry point
```

## Commands

```bash
pip install -e ".[dev]"     # Install with dev deps
earworm analyze <file>      # Run analysis pipeline
pytest                      # Run tests
ruff check earworm/         # Lint
```

## Conventions

- Conventional commits (feat:, fix:, chore:, etc.)
- Co-Authored-By trailer on AI-assisted commits
- Feature branches off main, squash merge
- Each feature module returns a pydantic model
- Pipeline is composable — layers can run independently
