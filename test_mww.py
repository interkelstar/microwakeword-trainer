#!/usr/bin/env python3
"""
Evaluate a trained microWakeWord TFLite model against WAV files.

Simulates the exact pymicro_wakeword inference loop used by linux-voice-assistant:
  - Extracts 40-bin mel spectrogram frames at 10 ms steps (pymicro_features)
  - Accumulates 'stride' frames, runs inference, clears the frame buffer
  - Maintains a sliding window of the last sliding_window_size probabilities
  - Triggers when mean(window) > probability_cutoff

The model and config paths are set at the top of this file.  Edit MODEL_PATH
and CONFIG_PATH if you use a different model name.

Usage:
  .venv/bin/python3 test_mww.py [WAV ...]
  .venv/bin/python3 test_mww.py record/sample*.wav

If no WAV files are passed, all *.wav files in record/ are tested.
Output shows per-inference-window probabilities and a TRIGGERED / no trigger result.
"""

import sys
import json
import struct
import pathlib
import collections
import numpy as np
import scipy.io.wavfile as wavfile

MODEL_PATH  = pathlib.Path("ru_jarvis_mww.tflite")
CONFIG_PATH = pathlib.Path("ru_jarvis_mww.json")
CHUNK_SAMPLES = 160   # 10ms @ 16kHz — pymicro_features fixed


def load_model():
    try:
        import tensorflow as tf
        interp = tf.lite.Interpreter(str(MODEL_PATH))
    except Exception:
        import tflite_runtime.interpreter as tflite
        interp = tflite.Interpreter(str(MODEL_PATH))
    interp.allocate_tensors()
    return interp


def run_inference(interp, window_frames: np.ndarray) -> float:
    """Feed a (stride, 40) float32 window → return probability 0-1."""
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    scale, zero_point = inp_detail["quantization"]
    dtype = inp_detail["dtype"]

    if dtype == np.int8:
        quant = np.clip(
            np.round(window_frames / scale + zero_point), -128, 127
        ).astype(np.int8)
    else:
        quant = np.clip(
            np.round(window_frames / scale + zero_point), 0, 255
        ).astype(np.uint8)

    # Shape: model expects (1, stride, 40) — no channel dim for streaming models
    inp_shape = inp_detail["shape"]
    if len(inp_shape) == 3:
        interp.set_tensor(inp_detail["index"], quant[np.newaxis])
    else:
        interp.set_tensor(inp_detail["index"], quant[np.newaxis, :, :, np.newaxis])
    interp.invoke()

    out_raw = interp.get_tensor(out_detail["index"])
    out_scale, out_zp = out_detail["quantization"]
    prob = float((out_raw.flat[0] - out_zp) * out_scale)
    return max(0.0, min(1.0, prob))


def detect(wav_path: str, interp, probability_cutoff: float,
           sliding_window_size: int, verbose: bool = True) -> dict:
    """Simulate pymicro_wakeword on a single WAV file."""
    from pymicro_features import MicroFrontend

    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        import scipy.signal as sig
        audio = sig.resample(audio, int(len(audio) * 16000 / sr)).astype(np.int16)
    audio = audio.astype(np.int16)

    inp_detail = interp.get_input_details()[0]
    FEATURE_FRAMES = inp_detail["shape"][1]   # dimension 1 = stride

    frontend = MicroFrontend()
    frame_buffer = []           # accumulates up to FEATURE_FRAMES frames
    prob_window  = collections.deque(maxlen=sliding_window_size)
    triggered    = False
    trigger_time = None
    all_probs    = []
    all_times    = []

    frame_idx = 0   # counts 10ms frames processed

    for i in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES].tobytes()
        result = frontend.process_samples(chunk)
        if result is None or len(result.features) == 0:
            continue

        frame_buffer.append(list(result.features))
        frame_idx += 1

        if len(frame_buffer) == FEATURE_FRAMES:
            window = np.array(frame_buffer, dtype=np.float32)  # (16, 40)
            prob   = run_inference(interp, window)
            t_ms   = (frame_idx * 10)

            prob_window.append(prob)
            all_probs.append(prob)
            all_times.append(t_ms / 1000)

            avg = sum(prob_window) / len(prob_window)

            if verbose:
                bar = "█" * int(prob * 20)
                mark = " ◄ TRIGGER" if (avg > probability_cutoff and not triggered) else ""
                print(f"  t={t_ms/1000:5.2f}s  prob={prob:.3f} [{bar:<20}]  avg={avg:.3f}{mark}")

            if avg > probability_cutoff and not triggered:
                triggered    = True
                trigger_time = t_ms / 1000

            frame_buffer = []   # non-overlapping: clear after each inference

    return {
        "triggered": triggered,
        "trigger_time": trigger_time,
        "max_prob": max(all_probs) if all_probs else 0.0,
        "max_avg":  max(
            sum(list(prob_window)[-sliding_window_size:]) / min(len(all_probs), sliding_window_size)
            for prob_window in [
                collections.deque(all_probs[:k+1], maxlen=sliding_window_size)
                for k in range(len(all_probs))
            ]
        ) if all_probs else 0.0,
        "probs": all_probs,
        "times": all_times,
    }


def main():
    wav_files = sys.argv[1:] if len(sys.argv) > 1 else sorted(pathlib.Path("record").glob("*.wav"))
    if not wav_files:
        print("Usage: python3 test_mww.py [WAV ...]")
        sys.exit(1)

    # Load config
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    micro = config.get("micro", {})
    probability_cutoff  = micro.get("probability_cutoff", 0.5)
    sliding_window_size = micro.get("sliding_window_size", 5)

    print(f"Model:   {MODEL_PATH}  ({MODEL_PATH.stat().st_size/1024:.1f} KB)")
    print(f"Config:  probability_cutoff={probability_cutoff}  sliding_window_size={sliding_window_size}")

    # Load model once
    interp = load_model()
    inp = interp.get_input_details()[0]
    print(f"Input:   shape={inp['shape']}  dtype={inp['dtype'].__name__}  quant={inp['quantization']}\n")

    results = []
    for wav in wav_files:
        # Reset interpreter state (ring buffers) between files
        interp.allocate_tensors()
        print(f"{'='*60}")
        print(f"File: {wav}")
        r = detect(str(wav), interp, probability_cutoff, sliding_window_size, verbose=True)
        status = "TRIGGERED" if r["triggered"] else "no trigger"
        print(f"→ {status}  max_prob={r['max_prob']:.3f}  max_avg={r['max_avg']:.3f}"
              + (f"  at t={r['trigger_time']:.2f}s" if r['trigger_time'] else ""))
        results.append((str(wav), r))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    triggered = sum(1 for _, r in results if r["triggered"])
    print(f"  {triggered}/{len(results)} samples triggered")
    for path, r in results:
        mark = "✓" if r["triggered"] else "✗"
        print(f"  {mark}  {pathlib.Path(path).name}  max_prob={r['max_prob']:.3f}  max_avg={r['max_avg']:.3f}")


if __name__ == "__main__":
    main()
