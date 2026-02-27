import tensorflow as tf
import numpy as np
import time

# ── Configuration ────────────────────────────────────────────
MODEL_PATH = "hello_world_int8.tflite"

ARENA_SIZES_KB = [8, 16, 32, 48, 64, 96, 128]
# ─────────────────────────────────────────────────────────────

def test_model_at_arena_size(model_path, arena_kb, iterations=5):
    """Load a TFLite model, run multiple times, and measure average elapsed time.

    Runs `iterations` times and returns the average elapsed time.
    Returns a tuple (success, result_or_error, avg_elapsed_secs).
    """
    try:
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=1
        )
        interpreter.allocate_tensors()

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        # ensure the test tensor uses the expected dtype (e.g. INT8 for quantized models)
        input_dtype = input_details[0]['dtype']
        input_data  = np.ones(input_shape, dtype=input_dtype)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # warmup (discard first run to account for initialization)
        interpreter.invoke()

        # measure actual runs and average
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            interpreter.invoke()
            end = time.perf_counter()
            times.append(end - start)

        output = interpreter.get_tensor(output_details[0]['index'])
        avg_time = sum(times) / len(times)
        return True, output, avg_time

    except Exception as e:
        print(f"    ERROR at {arena_kb}KB: {e}")
        return False, str(e), None



def profile_model(model_path, improvement_threshold=0.05):
    """Return detailed timing and pass/fail information for each arena size.

    Returns:
        results: list of dicts {'kb','success','time','result'}
        minimum_ram: first successful kb or None
        optimal_ram: computed by find_optimal_arena or None
    """
    results = []
    minimum_ram = None
    arena_times = []

    for kb in ARENA_SIZES_KB:
        success, result, elapsed = test_model_at_arena_size(model_path, kb)
        results.append({'kb': kb, 'success': success, 'time': elapsed, 'result': result})
        if success:
            arena_times.append((kb, elapsed))
            if minimum_ram is None:
                minimum_ram = kb

    optimal_ram = find_optimal_arena(arena_times, improvement_threshold)
    return results, minimum_ram, optimal_ram


def run_profiler(model_path, improvement_threshold=0.05):
    print(f"\n{'='*50}")
    print(f"  TinyML RAM Profiler")
    print(f"  Model: {model_path}")
    print(f"{'='*50}\n")

    results, minimum_ram, optimal_ram = profile_model(model_path, improvement_threshold)

    for entry in results:
        status = "✅ PASS" if entry['success'] else "❌ FAIL"
        time_str = f" ({entry['time']*1000:.1f} ms)" if entry['time'] is not None else ""
        print(f"  Arena {entry['kb']:>4} KB  →  {status}{time_str}")

    print(f"\n{'='*50}")
    if minimum_ram:
        print(f"  Minimum RAM required : {minimum_ram} KB")
        recommend_mcu(minimum_ram)
        if optimal_ram is not None and optimal_ram != minimum_ram:
            print(f"  Optimal RAM (≈fastest): {optimal_ram} KB")
    else:
        print("  Model failed at all tested arena sizes.")
    print(f"{'='*50}\n")



def find_optimal_arena(arena_times, threshold=0.05):
    """Pick the smallest arena size whose execution time is within
    `threshold` (fraction) of the fastest observed time.

    arena_times: list of (kb, elapsed_secs) for successful runs.
    """
    if not arena_times:
        return None
    fastest = min(t for _, t in arena_times)
    cutoff = fastest * (1 + threshold)
    candidates = [kb for kb, t in arena_times if t <= cutoff]
    return min(candidates) if candidates else None


def recommend_mcu(min_ram_kb):
    print(f"\n  MCU Recommendation based on {min_ram_kb} KB RAM:\n")
    if min_ram_kb <= 16:
        print("  → Ultra Low-End  : STM32L0, ATtiny (16–32 KB RAM)")
    elif min_ram_kb <= 32:
        print("  → Low-End        : STM32L1, nRF51822 (32–64 KB RAM)")
    elif min_ram_kb <= 64:
        print("  → Mid-Range      : STM32L4, nRF52832 (64–128 KB RAM)")
    elif min_ram_kb <= 128:
        print("  → Upper Mid      : STM32F4, ESP32 (128–256 KB RAM)")
    else:
        print("  → High-End MCU   : STM32H7, ESP32-S3 (256 KB+ RAM)")


# ── Run ──────────────────────────────────────────────────────
run_profiler(MODEL_PATH)