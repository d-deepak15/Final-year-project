import tensorflow as tf
import numpy as np

# ── Configuration ────────────────────────────────────────────
MODEL_PATH = "hello_world_int8.tflite"

ARENA_SIZES_KB = [8, 16, 32, 48, 64, 96, 128]
# ─────────────────────────────────────────────────────────────

def test_model_at_arena_size(model_path, arena_kb):
    try:
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=1
        )
        interpreter.allocate_tensors()

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_data  = np.ones(input_shape, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details[0]['index'])
        return True, output

    except Exception as e:
        print(f"    ERROR at {arena_kb}KB: {e}")  # ← add this line
        return False, str(e)


def run_profiler(model_path):
    print(f"\n{'='*50}")
    print(f"  TinyML RAM Profiler")
    print(f"  Model: {model_path}")
    print(f"{'='*50}\n")

    minimum_ram = None

    for kb in ARENA_SIZES_KB:
        success, result = test_model_at_arena_size(model_path, kb)
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  Arena {kb:>4} KB  →  {status}")

        if success and minimum_ram is None:
            minimum_ram = kb

    print(f"\n{'='*50}")
    if minimum_ram:
        print(f"  Minimum RAM required : {minimum_ram} KB")
        recommend_mcu(minimum_ram)
    else:
        print("  Model failed at all tested arena sizes.")
    print(f"{'='*50}\n")


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