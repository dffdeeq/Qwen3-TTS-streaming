# Qwen3-TTS Streaming

Streaming inference implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) that the official repo doesn't provide.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper and marketing, but the actual streaming code was never released — they point users to vLLM-Omni, which still doesn't support online serving.

This fork adds real streaming generation directly to the `qwen-tts` package.

## What's Added

- `stream_generate_pcm()` — real-time PCM audio streaming
- `stream_generate_voice_clone()` — streaming with voice cloning

## Benchmark (RTX 5090)

| Metric | Value |
|--------|-------|
| Standard generation | 12.58s |
| **First chunk latency** | **0.86s** |
| Streaming total | 12.11s |
| Latency improvement | 11.72s faster to first audio |

15 chunks, ~0.8s between emissions.

## Usage

```python
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Create voice clone prompt (do once, reuse)
voice_clone_prompt = model.create_voice_clone_prompt(
    ref_audio="reference.wav",
    ref_text="Your reference text here",
)

# Streaming generation
chunks = []
for chunk, sr in model.stream_generate_voice_clone(
    text="Your text to synthesize",
    language="Russian",  # or any supported language
    voice_clone_prompt=voice_clone_prompt,
    emit_every_frames=8,
    decode_window_frames=80,
    overlap_samples=512,
):
    chunks.append(chunk)
    # Here you can send chunk to audio player in real-time

# Or save complete audio
final_audio = np.concatenate(chunks)
sf.write("output.wav", final_audio, sr)
```

See [`examples/test_streaming.py`](examples/test_streaming.py) for a complete benchmark script.

## Installation

```bash
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git
cd Qwen3-TTS-streaming
pip install -e .
pip install flash-attn --no-build-isolation  # recommended
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 8 | Emit audio every N frames (~0.64s at 12Hz) |
| `decode_window_frames` | 80 | Decoder context window |
| `overlap_samples` | 512 | Crossfade overlap between chunks |

## Why This Exists

From official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork provides streaming now, without waiting for vLLM-Omni updates.

---

Based on [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
