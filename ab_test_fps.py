#!/usr/bin/env python3
"""
A/B test two TFLite models on the same false-positive clips.
Compares an 'old' (baseline) and a 'new' model by swapping which TFLite
test_mww.py loads (via shim MODEL_PATH) — so we don't touch production files.

Usage:
  python3 ab_test_fps.py OLD.tflite NEW.tflite DIR [DIR ...] [--config MODEL.json]

The config (probability_cutoff, sliding_window_size) defaults to the JSON
manifest next to NEW.tflite.
"""
import argparse
import json
import pathlib
import importlib.util
import sys

_here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("test_mww", _here / "test_mww.py")
tm = importlib.util.module_from_spec(spec); spec.loader.exec_module(tm)
import numpy as np


def run_model(model_path, wavs, cutoff, win):
    # Patch MODEL_PATH used inside test_mww.load_model()
    tm.MODEL_PATH = pathlib.Path(model_path)
    interp = tm.load_model()
    rows = []
    for w in wavs:
        interp.allocate_tensors()
        r = tm.detect(str(w), interp, cutoff, win, verbose=False)
        rows.append((w.name, r["triggered"], r["max_prob"], r["max_avg"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", help="baseline .tflite model")
    parser.add_argument("new", help="candidate .tflite model")
    parser.add_argument("dirs", nargs="+", help="directories of FP WAV clips")
    parser.add_argument("--config", default=None,
                        help="JSON manifest (default: NEW model path with .json suffix)")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config) if args.config \
        else pathlib.Path(args.new).with_suffix(".json")
    micro = json.load(open(cfg_path)).get("micro", {})
    cutoff = micro.get("probability_cutoff", 0.5)
    win = micro.get("sliding_window_size", 5)

    wavs = []
    for d in args.dirs:
        wavs += sorted(pathlib.Path(d).glob("*.wav"))
    if not wavs:
        print("no wavs found"); sys.exit(1)

    print(f"A/B test: {len(wavs)} clips @ cutoff={cutoff} win={win}")
    print(f"  OLD: {args.old}")
    print(f"  NEW: {args.new}")
    print("=" * 72)

    old = run_model(args.old, wavs, cutoff, win)
    new = run_model(args.new, wavs, cutoff, win)

    n_old_trig = sum(1 for r in old if r[1])
    n_new_trig = sum(1 for r in new if r[1])
    old_ma = np.array([r[3] for r in old])
    new_ma = np.array([r[3] for r in new])

    def thr_rows(arr):
        return "  " + "  ".join(f"≥{t}: {int((arr >= t).sum())}" for t in (0.99, 0.97, 0.95, 0.9, 0.8))

    print(f"\nOLD triggered: {n_old_trig}/{len(old)}")
    print(thr_rows(old_ma))
    print(f"\nNEW triggered: {n_new_trig}/{len(new)}")
    print(thr_rows(new_ma))

    print("\nChanges (clip, OLD_maxAvg → NEW_maxAvg):")
    worst = []
    for (n_, ot, omp, omv), (_, nt, nmp, nmv) in zip(old, new):
        d = nmv - omv
        worst.append((d, n_, omv, nmv, ot, nt))
    worst.sort()
    print(f"  improved (max_avg dropped most):")
    for d, n_, omv, nmv, ot, nt in worst[:8]:
        print(f"    {n_:<38} {omv:.3f}→{nmv:.3f} ({d:+.3f})  {'YES' if ot else ' - '}→{'YES' if nt else ' - '}")
    print(f"  regressed (max_avg rose most, or new trigger):")
    for d, n_, omv, nmv, ot, nt in worst[-8:][::-1]:
        print(f"    {n_:<38} {omv:.3f}→{nmv:.3f} ({d:+.3f})  {'YES' if ot else ' - '}→{'YES' if nt else ' - '}")

    print("\n" + "=" * 72)
    n_new_triggers_from_old_clean = sum(1 for (_, _, _, omv, ot, nt) in worst if nt and not ot)
    n_fixed = sum(1 for (_, _, _, omv, ot, nt) in worst if ot and not nt)
    print(f"FPs eliminated (old trigger → no trigger): {n_fixed}")
    print(f"New FPs introduced (no trigger → trigger): {n_new_triggers_from_old_clean}")
    print(f"Net: {n_old_trig - n_new_trig:+d} triggers  ({n_old_trig}→{n_new_trig})")


if __name__ == "__main__":
    main()
