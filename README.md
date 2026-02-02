# Qwen3-TTS Streaming

Real-time streaming audio generation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS).

## Features

From [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming):
- `stream_generate_voice_clone()` - streaming with voice cloning
- `stream_generate_pcm()` - real-time PCM audio streaming
- `torch.compile` + CUDA graphs optimization

Added in this fork:
- **Two-phase streaming** - faster first-chunk latency (2.75x improvement)
- **Audio quality fixes** - click-free chunk blending with Hann crossfade

## Two-Phase Streaming

Standard streaming with Qwen's TTS library waits for `emit_every_frames` (e.g., 12) before emitting the first audio. Two-phase uses aggressive settings for the first chunk to improve latency, then switches to stable settings.

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1 (First N frames)      │  PHASE 2 (Rest of audio)      │
│  - emit_every = 5 (fast)       │  - emit_every = 12 (stable)   │
│  - decode_window = 48          │  - decode_window = 80         │
│  → FAST first chunk            │  → QUALITY for rest           │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmarks

| Test | Method | emit | 1st Chunk | 1st Spdup | Total | Tot Spdup | RTF |
|------|--------|------|-----------|-----------|-------|-----------|-----|
| 2 | Baseline (no opt) | 12 | 570ms | 1.00x | 3.16s | 1.00x | 0.56 |
| 3 | Optimized | 12 | 389ms | 1.47x | 2.37s | 1.34x | 0.37 |
| 4 | Optimized_2 (stable) | 12 | 382ms | 1.49x | 2.27s | 1.39x | 0.36 |
| 5 | **Two-phase (5→12)** | 5→12 | **208ms** | **2.75x** | 2.58s | 1.23x | 0.39 |

User hears audio **362ms earlier** vs baseline, **174ms earlier** vs only optimized.

**First-chunk latency improvement:**
- vs Baseline: **2.75x faster** (570ms → 208ms, saves 362ms)
- vs Optimized: **1.87x faster** (389ms → 208ms, saves 181ms)
- vs Optimized_2: **1.84x faster** (382ms → 208ms, saves 174ms)

## Audio Quality Fixes

Streaming TTS can produce clicks, pops, and artifacts at chunk boundaries. This fork implements several fixes:

### Crossfade Blending

Chunks are blended using a Hann window crossfade to eliminate boundary discontinuities:

```python
MIN_BLEND_SAMPLES = 512  # ~21ms at 24kHz

# blend_samples is at least MIN_BLEND_SAMPLES, even if overlap_samples is smaller
blend_samples = max(overlap_samples, MIN_BLEND_SAMPLES)

# Hann crossfade
fade_out = 0.5 * (1 + np.cos(np.pi * t))
fade_in = 0.5 * (1 - np.cos(np.pi * t))
blended = prev_tail * fade_out + curr_head * fade_in
```

### Overlap Trimming

Each chunk is processed in this order to prevent audio duplication (echo artifacts):

1. Crossfade current chunk's HEAD with previous chunk's saved TAIL
2. Apply fade-in (first chunk only)
3. Save FULL processed chunk for next iteration's crossfade
4. Trim END of chunk before emission (this region will be replaced by next chunk's crossfade)
5. Yield trimmed chunk

### First/Last Chunk Fades

- **First chunk**: Hann fade-in prevents pop at audio start
- **Final chunk**: Hann fade-out prevents pop at audio end

## Optimization API

### enable_streaming_optimizations()

Call after loading the model to enable torch.compile and CUDA graphs:

```python
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended)
model.enable_streaming_optimizations(
    decode_window_frames=80,         # Must match streaming parameter
    use_compile=True,                # torch.compile the decoder
    compile_mode="reduce-overhead",  # Includes CUDA graphs automatically
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decode_window_frames` | 80 | Window size (must match streaming call) |
| `use_compile` | True | Apply torch.compile to decoder |
| `use_cuda_graphs` | True | Capture CUDA graphs for fixed window |
| `compile_mode` | "reduce-overhead" | torch.compile mode |
| `use_fast_codebook` | False | Use fast codebook generation (experimental) |
| `compile_codebook_predictor` | True | Apply torch.compile to codebook predictor |

## Usage

```python
import torch
import sounddevice as sd
from qwen_tts import Qwen3TTSModel

# Load model
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-Base",
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

# Enable optimizations (recommended for streaming)
model.enable_streaming_optimizations(
    decode_window_frames=80,
    use_compile=True,
    compile_mode="reduce-overhead",
)

# Create voice clone prompt from reference audio
prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Transcript of the reference audio.",
)

# Stream audio with two-phase settings
for chunk, sr in model.stream_generate_voice_clone(
    text="Hello, this is a streaming TTS demo!",
    language="en",
    voice_clone_prompt=prompt,
    # Phase 2 settings (stable)
    emit_every_frames=12,
    decode_window_frames=80,
    # Phase 1 settings (fast first chunk)
    first_chunk_emit_every=5,
    first_chunk_decode_window=48,
    first_chunk_frames=48,
):
    sd.play(chunk, sr)
    sd.wait()
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks |
| `max_frames` | 10000 | Maximum codec frames to generate |
| `first_chunk_emit_every` | 0 | Phase 1 emit interval (0 = disabled) |
| `first_chunk_decode_window` | 48 | Phase 1 decode window |
| `first_chunk_frames` | 48 | Switch to phase 2 after N frames |
| `use_optimized_decode` | True | Use torch.compile/CUDA graph optimized decode |

## Installation

```bash
sudo apt install sox
pip install torch torchaudio flash-attn
pip install -e .
```

---

Based on:
- [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- [dffdeeq/Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming)
