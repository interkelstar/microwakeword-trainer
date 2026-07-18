#!/usr/bin/env python3
"""
Batch-test a set of false-positive WAVs through a trained TFLite model at the
production cutoff (read from the model's JSON manifest). Prints only a summary:
which clips still trigger, sorted by max_avg. Non-overlapping inference loop
mirrors test_mww.py / pymicro_wakeword exactly.

Usage:
  python3 batch_test_fps.py --model MODEL.tflite DIR [DIR ...]
"""
import argparse
import json
import pathlib
import importlib.util
import sys

import numpy as np

_here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("test_mww", _here / "test_mww.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="TFLite model path")
    parser.add_argument("--config", default=None,
                        help="JSON manifest (default: model path with .json suffix)")
    parser.add_argument("dirs", nargs="+", help="directories of FP WAV clips")
    args = parser.parse_args()

    model_path = pathlib.Path(args.model)
    config_path = pathlib.Path(args.config) if args.config else model_path.with_suffix(".json")

    wavs = []
    for d in args.dirs:
        wavs += sorted(pathlib.Path(d).glob("*.wav"))
    if not wavs:
        print("no wavs found"); sys.exit(1)

    micro = json.load(open(config_path)).get("micro", {})
    cutoff = micro.get("probability_cutoff", 0.5)
    win = micro.get("sliding_window_size", 5)

    tm.MODEL_PATH = model_path
    interp = tm.load_model()
    print(f"Model: {model_path} ({model_path.stat().st_size/1024:.1f} KB)")
    print(f"Testing {len(wavs)} clips @ cutoff={cutoff} sliding_window={win}")
    print("=" * 70)

    rows = []
    for w in wavs:
        interp.allocate_tensors()
        r = tm.detect(str(w), interp, cutoff, win, verbose=False)
        rows.append((w.name, r["triggered"], r["trigger_time"],
                     r["max_prob"], r["max_avg"]))

    rows.sort(key=lambda r: r[4], reverse=True)
    n_trig = sum(1 for r in rows if r[1])
    print(f"{'clip':<40} {'trig':>5} {'maxP':>6} {'maxAvg':>7}  t_trig")
    print("-" * 70)
    for name, trig, tt, mp, ma in rows:
        flag = "YES" if trig else " - "
        print(f"{name:<40} {flag:>5} {mp:6.3f} {ma:7.3f}  {tt:.2f}s" if tt else
              f"{name:<40} {flag:>5} {mp:6.3f} {ma:7.3f}")
    print("-" * 70)
    print(f"STILL TRIGGERING at cutoff={cutoff}: {n_trig}/{len(rows)}")
    mas = np.array([r[4] for r in rows])
    for thr in (0.99, 0.97, 0.95, 0.90, 0.80):
        print(f"  clips with max_avg >= {thr}: {int((mas >= thr).sum())}")


if __name__ == "__main__":
    main()
