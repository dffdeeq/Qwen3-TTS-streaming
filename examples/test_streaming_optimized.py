"""
Test streaming TTS with torch.compile and CUDA graphs optimizations.

This script compares:
1. Standard (non-streaming) generation
2. Streaming without optimizations
3. Streaming with torch.compile + CUDA graphs (decoder, talker, codebook predictor)

Usage:
    cd Qwen3-TTS
    python examples/test_streaming_optimized.py
"""

import time
import numpy as np
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# Enable TensorFloat32 for better performance on Ampere+ GPUs
torch.set_float32_matmul_precision('high')


def log_time(start, operation):
    elapsed = time.time() - start
    print(f"[{elapsed:.2f}s] {operation}")
    return time.time()


def run_streaming_test(
    model,
    text: str,
    language: str,
    voice_clone_prompt,
    emit_every_frames: int = 8,
    decode_window_frames: int = 80,
    label: str = "streaming",
):
    """Run streaming generation and return timing stats."""
    start = time.time()
    chunks = []
    chunk_sizes = []
    first_chunk_time = None
    chunk_count = 0
    sample_rate = 24000

    for chunk, chunk_sr in model.stream_generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=emit_every_frames,
        decode_window_frames=decode_window_frames,
        overlap_samples=0,
    ):
        chunk_count += 1
        chunks.append(chunk)
        chunk_sizes.append(len(chunk))
        sample_rate = chunk_sr
        if first_chunk_time is None:
            first_chunk_time = time.time() - start

    total_time = time.time() - start
    final_audio = np.concatenate(chunks) if chunks else np.array([])

    # Calculate audio duration and chunk stats
    audio_duration = len(final_audio) / sample_rate if sample_rate > 0 else 0
    avg_chunk_samples = np.mean(chunk_sizes) if chunk_sizes else 0
    avg_chunk_duration = avg_chunk_samples / sample_rate if sample_rate > 0 else 0

    return {
        "label": label,
        "first_chunk_time": first_chunk_time,
        "total_time": total_time,
        "chunk_count": chunk_count,
        "audio": final_audio,
        "sample_rate": sample_rate,
        "audio_duration": audio_duration,
        "avg_chunk_samples": avg_chunk_samples,
        "avg_chunk_duration": avg_chunk_duration,
    }


def main():
    total_start = time.time()

    # Streaming parameters - KEEP THESE CONSISTENT!
    EMIT_EVERY = 4  # Reduced from 8 for lower latency
    DECODE_WINDOW = 80

    print("=" * 60)
    print("Loading model...")
    print("=" * 60)

    start = time.time()
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    log_time(start, "Model loaded")

    # Reference audio setup
    ref_audio_path = "kuklina-1.wav"
    ref_text = (
        "Это брат Кэти, моей одноклассницы. А что у тебя с рукой? И почему ты голая? У него ведь куча наград по "
        "боевым искусствам. Кэти рассказывала, правда, Лео? Понимаешь кого ты побила, Лая? "
        "Только потрогай эти мышцы... Не знала, что у тебя такой классный котик. Рожденная луной. "
        "Лай всегда откопает что-нибудь этакое. Да, жаль только, что занимает почти всё её время. "
        "Не понимаю, почему эта рухлядь не может подождать, пока ты проведешь время с сестрой."
    )

    start = time.time()
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )
    log_time(start, "Voice clone prompt created")

    # Test text
    test_text = "Всем привет! Это тестовый текст для озвучки! Теперь стриминг звучит хорошо сразу."

    results = []

    # ============== Test 1: Standard generation ==============
    print("\n" + "=" * 60)
    print("Test 1: Standard (non-streaming) generation")
    print("=" * 60)

    start = time.time()
    wavs, sr = model.generate_voice_clone(
        text=test_text,
        language="Russian",
        voice_clone_prompt=voice_clone_prompt,
    )
    standard_time = time.time() - start
    standard_audio_duration = len(wavs[0]) / sr
    standard_rtf = standard_time / standard_audio_duration
    print(f"[{standard_time:.2f}s] Standard generation complete")
    print(f"Audio duration: {standard_audio_duration:.2f}s, RTF: {standard_rtf:.2f}")
    sf.write("output_standard.wav", wavs[0], sr)
    results.append({
        "label": "standard",
        "total_time": standard_time,
        "audio_duration": standard_audio_duration,
    })

    # ============== Test 2: Streaming WITHOUT optimizations ==============
    print("\n" + "=" * 60)
    print("Test 2: Streaming WITHOUT optimizations")
    print("=" * 60)

    result = run_streaming_test(
        model, test_text, "Russian", voice_clone_prompt,
        emit_every_frames=EMIT_EVERY,
        decode_window_frames=DECODE_WINDOW,
        label="streaming_baseline",
    )
    results.append(result)
    sf.write("output_streaming_baseline-ref-fixed.wav", result["audio"], result["sample_rate"])
    rtf = result['total_time'] / result['audio_duration'] if result['audio_duration'] > 0 else 0
    print(f"First chunk: {result['first_chunk_time']:.2f}s, Total: {result['total_time']:.2f}s, Chunks: {result['chunk_count']}")
    print(f"Audio duration: {result['audio_duration']:.2f}s, Chunk duration: {result['avg_chunk_duration']*1000:.0f}ms, RTF: {rtf:.2f}")

    # ============== Test 3: Streaming WITH optimizations ==============
    print("\n" + "=" * 60)
    print("Test 3: Streaming WITH torch.compile (decoder + talker + codebook)")
    print("=" * 60)

    # Enable optimizations - this is the key step!
    # - Decoder torch.compile with reduce-overhead mode (includes CUDA graphs)
    # - Talker model compilation for faster main generation loop
    # - Codebook predictor compilation
    print("\nEnabling streaming optimizations...")
    model.enable_streaming_optimizations(
        decode_window_frames=DECODE_WINDOW,
        use_compile=True,
        use_cuda_graphs=False,  # Not needed with reduce-overhead mode
        compile_mode="reduce-overhead",
        use_fast_codebook=True,
        compile_codebook_predictor=True,
        compile_talker=True,
    )

    # Warmup runs (first runs after compile are slower due to compilation)
    warmup_texts = [
        "Тест один два три четыре пять.",
        "Привет, как дела? Это второй прогрев системы.",
        "Третий тестовый запуск для полного прогрева всех компонентов модели.",
    ]

    print("\nWarmup runs (compilation happens here)...")
    for i, warmup_text in enumerate(warmup_texts, 1):
        warmup_result = run_streaming_test(
            model, warmup_text, "Russian", voice_clone_prompt,
            emit_every_frames=EMIT_EVERY,
            decode_window_frames=DECODE_WINDOW,
            label=f"warmup_{i}",
        )
        warmup_rtf = warmup_result['total_time'] / warmup_result['audio_duration'] if warmup_result['audio_duration'] > 0 else 0
        print(f"  Warmup {i}: {warmup_result['total_time']:.2f}s, Audio: {warmup_result['audio_duration']:.2f}s, RTF: {warmup_rtf:.2f}")

    # Multiple optimized test runs
    test_texts = [
        ("Всем привет! Это тестовый текст для озвучки! Теперь стриминг звучит хорошо сразу.", "short"),
        ("Это более длинный текст для тестирования производительности стриминга. Мы хотим убедиться, что система работает стабильно на разных длинах текста.", "medium"),
        ("Короткий тест.", "tiny"),
        ("Всем привет! Это тестовый текст для озвучки! Теперь стриминг звучит хорошо сразу.", "short_repeat"),
    ]


    print("\nOptimized test runs...")
    for i, (text, text_label) in enumerate(test_texts, 1):
        result = run_streaming_test(
            model, text, "Russian", voice_clone_prompt,
            emit_every_frames=EMIT_EVERY,
            decode_window_frames=DECODE_WINDOW,
            label=f"optimized_{text_label}",
        )
        results.append(result)
        rtf = result['total_time'] / result['audio_duration'] if result['audio_duration'] > 0 else 0
        print(f"  Run {i} ({text_label}): First chunk: {result['first_chunk_time']:.2f}s, Total: {result['total_time']:.2f}s, Audio: {result['audio_duration']:.2f}s, RTF: {rtf:.2f}")

        if i == 1:
            sf.write("output_streaming_optimized.wav", result["audio"], result["sample_rate"])

    # ============== Summary ==============
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    baseline_result = results[1]  # streaming_baseline
    baseline_audio_dur = baseline_result.get("audio_duration", 0)
    baseline_rtf = baseline_result["total_time"] / baseline_audio_dur if baseline_audio_dur > 0 else 0

    print(f"\n{'Method':<25} {'1st Chunk':>10} {'Total':>8} {'Audio':>8} {'RTF':>6} {'Chunks':>7} {'RTF Speedup':>12}")
    print("-" * 80)

    # Standard generation
    std = results[0]
    std_rtf = std['total_time'] / std['audio_duration'] if std.get('audio_duration', 0) > 0 else 0
    print(f"{'Standard (no streaming)':<25} {'N/A':>10} {std['total_time']:>7.2f}s {std.get('audio_duration', 0):>7.2f}s {std_rtf:>6.2f} {'N/A':>7} {'N/A':>12}")

    for r in results[1:]:
        first = r.get("first_chunk_time", 0)
        total = r["total_time"]
        audio_dur = r.get("audio_duration", 0)
        rtf = total / audio_dur if audio_dur > 0 else 0
        chunks = r.get("chunk_count", 0)
        speedup_rtf = baseline_rtf / rtf if rtf > 0 else 0
        print(f"{r['label']:<25} {first:>9.2f}s {total:>7.2f}s {audio_dur:>7.2f}s {rtf:>6.2f} {chunks:>7} {speedup_rtf:>11.2f}x")

    # Statistics for optimized runs (excluding baseline)
    optimized_results = [r for r in results[2:] if r.get('audio_duration', 0) > 0]
    if optimized_results:
        avg_rtf = np.mean([r['total_time'] / r['audio_duration'] for r in optimized_results])
        avg_first_chunk = np.mean([r['first_chunk_time'] for r in optimized_results])
        print(f"\nOptimized runs statistics ({len(optimized_results)} runs):")
        print(f"  Average RTF: {avg_rtf:.2f}")
        print(f"  Average first chunk latency: {avg_first_chunk:.2f}s")

    # Chunk duration info
    if baseline_result.get("avg_chunk_duration", 0) > 0:
        print(f"\nChunk duration: ~{baseline_result['avg_chunk_duration']*1000:.0f}ms ({baseline_result['avg_chunk_samples']:.0f} samples @ {baseline_result['sample_rate']}Hz)")

    print(f"\n[{time.time() - total_start:.2f}s] TOTAL SCRIPT TIME")

    # Tips
    print("\n" + "=" * 60)
    print("TIPS FOR BEST PERFORMANCE")
    print("=" * 60)
    print("""
1. Call enable_streaming_optimizations() ONCE after model loading
2. Use compile_mode="reduce-overhead" (default) - it includes CUDA graphs automatically
3. First run after compile is slow (compilation), subsequent runs are fast
4. For lowest latency: use smaller emit_every_frames (e.g., 4)
5. For best quality: use larger decode_window_frames (e.g., 80-100)
6. compile_talker=True compiles the main talker model (uses "default" mode internally)
7. compile_codebook_predictor=True compiles the codebook predictor
8. You can also try compile_mode="max-autotune" for potentially better performance
   (but longer initial compilation time)
""")


if __name__ == "__main__":
    main()
