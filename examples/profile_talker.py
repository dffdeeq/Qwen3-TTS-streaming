"""
Profile Talker forward pass to identify bottlenecks.
"""

import time
import torch
from qwen_tts import Qwen3TTSModel

torch.set_float32_matmul_precision('high')


def profile_generate():
    print("Loading model...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    # Get internal model for profiling
    tts_model = model.model
    talker = tts_model.talker
    code_predictor = talker.code_predictor

    print(f"\nModel structure:")
    print(f"  Talker: {talker.__class__.__name__}")
    print(f"  CodePredictor: {code_predictor.__class__.__name__}")
    print(f"  CodePredictor.model: {code_predictor.model.__class__.__name__}")
    print(f"  Num codebook groups: {talker.config.num_code_groups}")

    # Check attention implementation
    print(f"\nAttention implementation:")
    print(f"  Talker: {talker.config._attn_implementation}")
    print(f"  CodePredictor: {code_predictor.config._attn_implementation}")

    # Create test inputs
    ref_audio_path = "../neurona-10sec.wav"
    ref_text = "Тестовый текст для профилирования."

    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text,
    )

    test_text = "Привет, это тест профилирования генерации речи."

    # Warmup
    print("\nWarmup run...")
    for chunk, sr in model.stream_generate_voice_clone(
        text="Раз два три.",
        language="Russian",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
    ):
        pass

    # Profile with torch profiler
    print("\nProfiling with torch.profiler...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        chunk_count = 0
        for chunk, sr in model.stream_generate_voice_clone(
            text=test_text,
            language="Russian",
            voice_clone_prompt=voice_clone_prompt,
            emit_every_frames=4,
        ):
            chunk_count += 1
            if chunk_count >= 5:  # Profile first 5 chunks only
                break

    # Print profiler results
    print("\n" + "=" * 80)
    print("TOP 20 CUDA OPERATIONS BY TIME:")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print("\n" + "=" * 80)
    print("TOP 20 CPU OPERATIONS BY TIME:")
    print("=" * 80)
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Manual timing of code_predictor.generate vs code_predictor.model.forward
    print("\n" + "=" * 80)
    print("MANUAL TIMING: code_predictor.generate() breakdown")
    print("=" * 80)

    # Create dummy inputs for code_predictor
    batch_size = 1
    hidden_size = talker.config.hidden_size
    device = talker.device
    dtype = next(talker.parameters()).dtype

    # Simulate past_hidden and last_id_hidden
    past_hidden = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
    last_id_hidden = torch.randn(batch_size, 1, hidden_size, device=device, dtype=dtype)
    inputs_embeds = torch.cat((past_hidden, last_id_hidden), dim=1)

    # Time code_predictor.generate()
    torch.cuda.synchronize()

    times_generate = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            result = code_predictor.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=talker.config.num_code_groups - 1,
                do_sample=True,
                top_k=50,
                temperature=1.0,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        torch.cuda.synchronize()
        times_generate.append(time.perf_counter() - start)

    print(f"code_predictor.generate() (10 runs):")
    print(f"  Mean: {sum(times_generate)/len(times_generate)*1000:.1f}ms")
    print(f"  Min:  {min(times_generate)*1000:.1f}ms")
    print(f"  Max:  {max(times_generate)*1000:.1f}ms")

    # Time individual forward calls
    projected = code_predictor.small_to_mtp_projection(inputs_embeds)

    times_forward = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            out = code_predictor.model(
                inputs_embeds=projected,
                use_cache=True,
            )
        torch.cuda.synchronize()
        times_forward.append(time.perf_counter() - start)

    print(f"\ncode_predictor.model.forward() single call (10 runs):")
    print(f"  Mean: {sum(times_forward)/len(times_forward)*1000:.1f}ms")
    print(f"  Min:  {min(times_forward)*1000:.1f}ms")
    print(f"  Max:  {max(times_forward)*1000:.1f}ms")

    # Time 7 sequential forward calls (what generate does internally)
    times_7_forwards = []
    for _ in range(10):
        start = time.perf_counter()
        with torch.no_grad():
            # Prefill
            out = code_predictor.model(inputs_embeds=projected, use_cache=True)
            past_kv = out.past_key_values

            # 6 more forward calls
            dummy_embed = torch.randn(1, 1, code_predictor.config.hidden_size,
                                      device=device, dtype=dtype)
            for _ in range(6):
                out = code_predictor.model(
                    inputs_embeds=dummy_embed,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out.past_key_values
        torch.cuda.synchronize()
        times_7_forwards.append(time.perf_counter() - start)

    print(f"\n7x code_predictor.model.forward() sequential (10 runs):")
    print(f"  Mean: {sum(times_7_forwards)/len(times_7_forwards)*1000:.1f}ms")
    print(f"  Min:  {min(times_7_forwards)*1000:.1f}ms")
    print(f"  Max:  {max(times_7_forwards)*1000:.1f}ms")

    overhead = (sum(times_generate)/len(times_generate) - sum(times_7_forwards)/len(times_7_forwards)) * 1000
    print(f"\nHF generate() overhead: ~{overhead:.1f}ms per call")

    # Time main talker forward
    print("\n" + "=" * 80)
    print("MANUAL TIMING: Talker.forward() breakdown")
    print("=" * 80)

    # Need to prepare proper inputs for talker
    # This is complex, so let's just time the streaming loop

    times_per_step = []
    step_count = 0

    import time as _time

    # Patch to measure step time
    original_forward = talker.forward.__wrapped__ if hasattr(talker.forward, '__wrapped__') else talker.forward

    print("\nMeasuring actual streaming step times...")

    for chunk, sr in model.stream_generate_voice_clone(
        text=test_text,
        language="Russian",
        voice_clone_prompt=voice_clone_prompt,
        emit_every_frames=4,
    ):
        pass  # Just run to completion

    print("\nDone profiling!")


if __name__ == "__main__":
    profile_generate()
