"""Put held-out synthetic clips through a room and see what the model does.

The deployed ru_stop v2 scores 0.996 on held-out Piper «стоп» and 0.000 on a
person saying it into the speaker's microphone. Between those two conditions
sit a room, a loudspeaker and a microphone. This applies just that — real
measured room impulse responses and real background noise, no change of voice
or wording — and writes the result out for scoring.

If the score collapses here, the missing ingredient is the channel, and adding
it to training is the fix. If it survives, the channel is not the difference
and something else is, in which case do not start a multi-hour training run on
this theory.

    python tools/channel_probe.py --n 12 --out /tmp/channel_probe
"""
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train_mww import ChannelAugmenter, SAMPLE_RATE  # noqa: E402


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def write_wav(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(np.clip(samples, -32768, 32767).astype("<i2").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path,
                    default=Path("training/output/ru_stop/positive_test"))
    ap.add_argument("--rir-dir", type=Path, default=Path("training/mit_rirs/16khz"))
    ap.add_argument("--background-dir", type=Path, default=Path("training/background"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.random.seed(args.seed)
    import random
    random.seed(args.seed)

    channel = ChannelAugmenter(args.rir_dir, args.background_dir, {
        # Always apply both: this is a probe of the channel's effect, not a
        # sample of the training distribution.
        "rir_probability": 1.0,
        "background_probability": 1.0,
        "background_snr_db": [0, 15],
    })
    if not channel:
        print("no augmentation assets found — nothing to probe", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    clips = sorted(args.source.glob("*.wav"))[:args.n]
    if not clips:
        print(f"no clips in {args.source}", file=sys.stderr)
        return 2

    for clip in clips:
        original = read_wav(clip).astype(np.float32)
        through = channel.apply(original)
        write_wav(args.out / clip.name, through)

    print(f"{len(clips)} clips through the channel -> {args.out}")
    print("score them beside the originals; a collapse here is the diagnosis")
    return 0


if __name__ == "__main__":
    sys.exit(main())
