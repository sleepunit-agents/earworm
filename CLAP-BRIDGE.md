# CLAP Integration Design: Earworm-Samplebank Semantic Bridge

*Design spec for earworm issue #9*

## What This Is

A bridge that connects earworm's perceptual output to samplebank's sample
library through CLAP embeddings. When Art analyzes a track, the bridge
answers: "What samples in my library relate to what I just heard?"

Two search paths:

1. **Text-to-audio** — Voice output (descriptions, tags) queries samplebank
   via CLAP text encoding. "Find samples that match this description."
2. **Audio-to-audio** — The analyzed track's audio queries samplebank via
   CLAP audio encoding. "Find samples that sound like this track."

## Current State

### What Already Exists

**Samplebank CLAP pipeline (fully deployed):**
- Model: `laion/larger_clap_music` (512-dim shared embedding space)
- Service: FastAPI on port 8100, GPU-accelerated (CUDA 12.4)
- Endpoints: `POST /embed/audio`, `POST /embed/text`, `GET /health`
- Qdrant collection: `samplebank_laion_clap`, cosine distance, 512-dim
- Background indexing worker with priority queue (batch size 10)
- Semantic search API: `GET /api/samples/semantic?q=<text>&limit=<n>`
- All 50k+ samples indexed (or indexing) via embedding worker

**Earworm Voice layer (shipped):**
- `VoiceResult` model with: `description`, `opinion`, `tags`,
  `comparisons`, `highlights`, `concerns`, `section_notes`
- Quick and deep interpretation modes
- LLM-powered (Anthropic or Ollama)

**Shared infrastructure:**
- Same Qdrant instance on the homelab
- Both projects deployed on the same host

### What Doesn't Exist Yet

- Any code that connects earworm output to samplebank search
- Audio-to-audio search in samplebank (only text-to-audio exists)
- An earworm-side Qdrant collection for cross-track similarity

## CLAP Model Selection

**Decision: `laion/larger_clap_music`** (already deployed, no change needed)

This is the right model. Rationale:
- Music-focused training data (vs general-purpose `laion/clap-htsat-unfused`)
- 512-dim embeddings — good balance of expressiveness and storage
- Shared text-audio embedding space enables both search paths
- Already proven in production with samplebank's 50k+ sample library

Alternatives considered and rejected:
- `microsoft/clap` — smaller model, less music-specific training
- PANNs — audio-only, no text embedding (was in earworm SPEC.md as a
  possibility, but CLAP's shared space is strictly more useful)
- Fine-tuned models — premature. Validate the bridge works with the
  base model first, tune later if semantic gaps emerge

## Deployment Plan

**No new deployment required for the primary path.**

Samplebank's CLAP service is already running on GPU. Earworm just needs
HTTP access to two things:

1. **Samplebank API** (`samplebank:8000`) — for text-to-audio search
2. **CLAP service** (`clap:8100`) — for audio-to-audio embedding

Both are on the homelab network. Earworm runs there too.

### Configuration

New earworm config (env vars or config file):

```
EARWORM_SAMPLEBANK_URL=http://localhost:8000
EARWORM_CLAP_URL=http://localhost:8100
```

No GPU needed on earworm's side — it delegates embedding to the CLAP
service. Earworm stays CPU-only (librosa, essentia).

## Qdrant Collection Schema

### Samplebank Collection (existing, no changes)

```
Collection: samplebank_laion_clap
Vectors:    512-dim, cosine distance
Payload:    { sample_id: int, filename: str, path: str }
```

This is the target collection for both search paths. Earworm searches
it; samplebank indexes it. Clean separation.

### Earworm Track Collection (future, not in this phase)

A potential `earworm_tracks` collection could store CLAP audio embeddings
of full tracks earworm has analyzed, enabling cross-track similarity
("what other tracks sound like this one?"). This is out of scope for the
bridge — file as a separate issue if we want it.

## API Surface Between Earworm and Samplebank

### Path 1: Text-to-Audio (Voice → Samples)

Earworm calls samplebank's existing API. No samplebank changes needed.

```
GET {SAMPLEBANK_URL}/api/samples/semantic?q={query}&limit={n}

Response: {
  "results": [
    { "id": int, "filename": str, "path": str, "semantic_score": float, ... }
  ],
  "total": int,
  "query": str
}
```

The bridge constructs queries from Voice output. Strategy:

**Primary query** — the `description` field as-is:
```
"A driving, hypnotic electronic track built on a four-on-the-floor
kick pattern with layered synth textures and a relentless minor-key
bassline."
```

**Tag queries** — individual or grouped `tags` for focused results:
```
"techno kick drum"
"dark pad texture"
"hypnotic bassline"
```

**Highlight queries** — specific sonic characteristics from `highlights`:
```
"clean low end with good stereo separation"
```

The bridge should run multiple queries and merge results, giving the
caller a richer picture than a single search. Weight strategy:

- `description` query: weight 1.0 (broadest semantic match)
- Per-tag queries: weight 0.6 (focused, may surface different samples)
- Per-highlight queries: weight 0.4 (specific sonic characteristics)

Dedup by sample ID, keep the highest score per sample across queries.

### Path 2: Audio-to-Audio (Track → Samples)

Earworm calls the CLAP service directly to embed the audio, then
searches samplebank's Qdrant collection.

```
# Step 1: Embed the input audio
POST {CLAP_URL}/embed/audio
Body: multipart/form-data with audio file
Response: { "embedding": [512 floats], "dim": 512 }

# Step 2: Search samplebank collection
POST {QDRANT_URL}/collections/samplebank_laion_clap/points/search
Body: { "vector": [512 floats], "limit": n, "with_payload": true }
Response: { "result": [{ "id": int, "score": float, "payload": {...} }] }
```

Note: This bypasses samplebank's application layer and queries Qdrant
directly. This is acceptable because:
- Earworm and samplebank share the same Qdrant instance
- The collection schema is stable and documented
- Read-only access — earworm never writes to samplebank's collection
- Adding an audio-search endpoint to samplebank is possible but
  unnecessary coupling for a read-only operation

Alternative: Add `POST /api/samples/semantic/audio` to samplebank that
accepts an audio file and returns matching samples. Worth doing later
if other consumers need it, but earworm can go direct for now.

### Bridge Module API

New module: `earworm.bridge.samplebank`

```python
from earworm.models import VoiceResult

class SamplebankBridge:
    """Connect earworm perception to samplebank's sample library."""

    def __init__(
        self,
        samplebank_url: str = "http://localhost:8000",
        clap_url: str = "http://localhost:8100",
        qdrant_url: str = "http://localhost:6333",
    ): ...

    def search_by_voice(
        self,
        voice: VoiceResult,
        limit: int = 20,
    ) -> list[SampleMatch]:
        """Text-to-audio: find samples matching Voice interpretation.

        Runs multiple queries (description, tags, highlights) and
        merges results by highest score.
        """

    def search_by_audio(
        self,
        audio_path: str,
        limit: int = 20,
    ) -> list[SampleMatch]:
        """Audio-to-audio: find samples that sound like this track.

        Embeds the audio via CLAP, searches samplebank's Qdrant
        collection directly.
        """

    def search_combined(
        self,
        voice: VoiceResult,
        audio_path: str,
        limit: int = 20,
        text_weight: float = 0.6,
        audio_weight: float = 0.4,
    ) -> list[SampleMatch]:
        """Both paths combined — weighted merge of text and audio results."""


class SampleMatch:
    sample_id: int
    filename: str
    path: str
    score: float          # 0.0-1.0, cosine similarity
    match_source: str     # "text", "audio", or "combined"
```

## Integration with Earworm Voice Output Format

The bridge consumes `VoiceResult` fields as follows:

| VoiceResult field | Bridge usage | Query weight |
|---|---|---|
| `description` | Primary semantic query — broad match | 1.0 |
| `tags` | Per-tag focused queries | 0.6 |
| `highlights` | Specific sonic characteristic queries | 0.4 |
| `opinion` | Not used for search (subjective judgment) | — |
| `comparisons` | Not used (artist names don't map to sample content) | — |
| `concerns` | Not used (negative traits don't help find samples) | — |
| `section_notes` | Future: per-section queries for deep mode | — |

### Query Construction

The bridge doesn't send Voice output verbatim. It constructs search
queries optimized for CLAP's text encoder:

**Description query**: Used as-is. CLAP was trained on music descriptions,
so natural language about sound works well.

**Tag queries**: Tags are short and precise ("techno", "dark", "hypnotic").
Group related tags for better semantic signal:
- Genre + energy: `"driving techno"` rather than `"driving"` + `"techno"` separately
- Texture + mood: `"dark hypnotic pad"` rather than three separate queries

**Highlight queries**: These are already natural language about specific
sonic qualities ("clean low end with good stereo separation") — pass
through directly.

### Example Flow

```
1. Art runs: earworm analyze --deep track.wav
2. Voice returns:
     description: "A driving techno track with layered percussion..."
     tags: ["techno", "driving", "dark", "minimal"]
     highlights: ["Clean low end", "Effective energy arc"]

3. Bridge constructs queries:
     q1: "A driving techno track with layered percussion..." (description)
     q2: "driving dark techno" (grouped tags)
     q3: "minimal techno percussion" (grouped tags)
     q4: "Clean low end" (highlight)
     q5: "Effective energy arc" (highlight)

4. Each query hits samplebank semantic search
5. Bridge merges results: dedup by sample_id, keep highest weighted score
6. Returns: top 20 samples with scores and match sources
```

## Indexing Pipeline for 50k+ Samples

**Already handled.** Samplebank's background embedding worker processes
the entire library:

- Worker pool: configurable thread count (default 1)
- Batch size: 10 samples per cycle
- Priority queue: new imports at priority 10, bulk reindex at priority 0
- Backpressure: queue depth cap (default 10,000)
- Staleness detection: tracks file mtime + size, re-embeds on change
- Reindex endpoint: `POST /api/samples/semantic/reindex?provider=laion_clap`

Earworm doesn't need to manage indexing. The bridge just searches
whatever is indexed.

**Monitoring**: Earworm can check indexing progress via
`GET /api/samples/semantic/status` before searching, and warn if
coverage is low (e.g., < 80% of samples indexed).

## Implementation Plan

### Phase 1: Text-to-Audio Bridge

1. Add `earworm/bridge/__init__.py` and `earworm/bridge/samplebank.py`
2. Implement `SamplebankBridge.search_by_voice()`:
   - Query construction from VoiceResult fields
   - Multi-query execution against samplebank API
   - Result merging with weighted dedup
3. Add config for samplebank URL
4. Tests with mocked samplebank responses
5. CLI integration: `earworm bridge <audio_file>` — analyze + search

### Phase 2: Audio-to-Audio Bridge

1. Implement `SamplebankBridge.search_by_audio()`:
   - Send audio file to CLAP service /embed/audio
   - Search samplebank's Qdrant collection directly
   - Return SampleMatch results
2. Add config for CLAP service and Qdrant URLs
3. Tests with mocked CLAP/Qdrant responses
4. CLI: `earworm bridge --audio-only <audio_file>`

### Phase 3: Combined Search

1. Implement `SamplebankBridge.search_combined()`:
   - Run both paths
   - Weighted merge with configurable text/audio balance
   - Score normalization across different query types
2. CLI: `earworm bridge --combined <audio_file>` (default mode)

### Phase 4: Integration with Art's Workflow

1. Loom MCP tool: `earworm_find_samples` — Art can search samplebank
   from any session after analyzing a track
2. Voice integration: option to append "related samples" section to
   VoiceResult output
3. Taste journal entries that reference specific samples

## Open Questions

1. **Tag grouping heuristic** — How to group tags into queries? Simple
   adjacency? Genre+modifier clustering? Start simple (2-3 tag groups),
   tune based on result quality.

2. **Score normalization** — Cosine similarity from different query types
   (long description vs single tag) will have different score
   distributions. May need per-query-type normalization. Measure first.

3. **CLAP service shared or separate?** — Currently earworm would use
   samplebank's CLAP service. If earworm needs heavy embedding workloads
   (batch analysis), consider a second CLAP instance. Not needed yet.

4. **Audio excerpt selection** — CLAP caps at 30s. For a 6-minute track,
   which 30s to embed? Options: first 30s, loudest 30s, random 30s,
   multiple excerpts averaged. Start with first 30s, revisit if results
   are poor.

## Non-Goals

- Earworm does not write to samplebank's database or Qdrant collection
- No new ML models or fine-tuning
- No real-time search (batch/on-demand only)
- No samplebank UI changes
- No cross-track similarity (earworm's own Qdrant collection) — separate issue
