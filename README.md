# Qwen3-TTS Streaming

Streaming inference implementation for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) that the official repo doesn't provide.

The official team mentions "Extreme Low-Latency Streaming Generation" in their paper and marketing, but the actual streaming code was never released - they point users to vLLM-Omni, which still doesn't support online serving.

This fork adds real streaming generation directly to the `qwen-tts` package.

## What's Added

- `stream_generate_pcm()` - real-time PCM audio streaming
- `stream_generate_voice_clone()` - streaming with voice cloning

## Benchmark (RTX 5090)

### Non-streaming (full inference)

<img width="575" height="136" alt="image" src="https://github.com/user-attachments/assets/e0744b19-d31b-44df-9f61-5be56f29056b" />


### Streaming

<img width="738" height="130" alt="image" src="https://github.com/user-attachments/assets/5694aa47-f2db-4b97-ae41-4d834838f8d8" />


## Usage

See examples/
- [test_streaming_optimized.py](https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_streaming_optimized.py)
- [test_optimized_no_streaming.py](https://github.com/dffdeeq/Qwen3-TTS-streaming/blob/main/examples/test_optimized_no_streaming.py)

## Installation (python 3.12)

> Note: torch versions differ between Linux/Windows due to available flash_attn prebuilt wheels.

### 1. Install SOX

**Linux:**
```bash
sudo apt install sox libsox-fmt-all
```

**Windows:** 
```bash
# Download from https://sourceforge.net/projects/sox/ and add to PATH !!
```

### 2. Create environment
```bash
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
```

### 3. Install dependencies

**Linux:**
```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu130torch2.9-cp312-cp312-linux_x86_64.whl
```

**Windows:**
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.8.3%2Bcu130torch2.10-cp312-cp312-win_amd64.whl
pip install -U "triton-windows<3.7"
```

### 4. Install package
```bash
git clone https://github.com/dffdeeq/Qwen3-TTS-streaming.git
cd Qwen3-TTS-streaming
pip install -e .
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `emit_every_frames` | 4 | Emit audio every N frames (~0.33s at 12Hz) |
| `decode_window_frames` | 80 | Decoder context window |

## Why This Exists

From official Qwen3-TTS README:
> Now only offline inference is supported. Online serving will be supported later.

This fork provides streaming now, without waiting for vLLM-Omni updates.

---

Based on [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
