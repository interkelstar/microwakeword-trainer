"""Generate wake/stop-word positives from the speech engines Home Assistant already has.

Why this exists: ru_stop v2 was trained on four Piper voices, all `-medium`,
and it learned those voices rather than the word — 0.996 on held-out Piper,
0.000 on a person saying «стоп» into the device's own microphone. Piper's
Russian voices are a narrow and rather mechanical slice of how the word can
sound. Home Assistant is already configured with Google Cloud, ElevenLabs and
others, and Google Cloud alone exposes ten native Russian speakers plus speed
and pitch, all reachable over the REST API.

Crucially this makes no sound: `/api/tts_get_url` synthesises to a file and
returns its URL rather than playing it anywhere, so a few hundred clips can be
generated with someone in the room.

It also stays clear of the held-out set. These are synthetic voices, not the
user's, so the recordings of the actual person remain untouched for evaluation
— which is both the honest way to measure recall and the answer to "will it
just overfit to my voice".

    HA_TOKEN=... python tools/generate_ha_tts.py --out training/output/ru_stop/ha_tts_positive
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

SAMPLE_RATE = 16_000

# Native Russian speakers across Google's three quality tiers. Standard and
# Wavenet are markedly different from each other, which is the point: the
# failure being fixed is a model that knows one synthesiser.
GOOGLE_VOICES = [
    "ru-RU-Standard-A", "ru-RU-Standard-B", "ru-RU-Standard-C",
    "ru-RU-Standard-D", "ru-RU-Standard-E",
    "ru-RU-Wavenet-A", "ru-RU-Wavenet-B", "ru-RU-Wavenet-C",
    "ru-RU-Wavenet-D", "ru-RU-Wavenet-E",
    "ru-RU-Chirp3-HD-Aoede", "ru-RU-Chirp3-HD-Charon",
    "ru-RU-Chirp3-HD-Kore", "ru-RU-Chirp3-HD-Puck",
]

# Punctuation changes the delivery — emphatic, questioning — which matters for
# a word that is normally said sharply, cutting in.
PHRASES = ["Стоп", "Стоп!", "Стоп?"]
SPEEDS = [0.85, 1.0, 1.15, 1.35]
PITCHES = [-4.0, 0.0, 4.0]


def tts_url(base: str, token: str, engine: str, message: str,
            language: str, options: dict) -> str | None:
    body = json.dumps({
        "engine_id": engine,
        "message": message,
        "language": language,
        "options": options,
    }).encode()
    req = urllib.request.Request(
        f"{base}/api/tts_get_url", data=body,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read())["url"]
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, json.JSONDecodeError):
        return None


def fetch_as_wav(url: str, dest: Path) -> bool:
    """Download and transcode to what the feature extractor expects."""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        return False
    if len(data) < 512:
        return False

    tmp = dest.with_suffix(".src")
    tmp.write_bytes(data)
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(tmp),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dest)],
        capture_output=True,
    )
    tmp.unlink(missing_ok=True)
    return proc.returncode == 0 and dest.exists() and dest.stat().st_size > 1024


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base", default=os.environ.get("HA_URL", "http://192.168.1.113:8123"))
    ap.add_argument("--engine", default="tts.google_cloud")
    ap.add_argument("--language", default="ru-RU")
    ap.add_argument("--limit", type=int, default=0, help="stop after N clips (0 = no limit)")
    ap.add_argument("--phrases-file", type=Path,
                    help="one phrase per line; blank lines and # comments ignored. "
                         "Used for negatives — the things the device hears that are "
                         "not the wake word")
    ap.add_argument("--holdout", type=Path,
                    help="second directory; voices are split between --out and this "
                         "one so no voice appears in both. Without a split by voice, "
                         "a model is tested on speakers it trained on and the number "
                         "means nothing")
    ap.add_argument("--speeds", default=None,
                    help="comma-separated; default 0.85,1.0,1.15,1.35. For negatives "
                         "breadth of phrases and voices matters more than of tempo, "
                         "and the grid multiplies fast")
    ap.add_argument("--pitches", default=None, help="comma-separated; default -4,0,4")
    ap.add_argument("--holdout-voices", type=int, default=4,
                    help="how many voices go to the holdout side")
    args = ap.parse_args()

    speeds = [float(v) for v in args.speeds.split(",")] if args.speeds else SPEEDS
    pitches = [float(v) for v in args.pitches.split(",")] if args.pitches else PITCHES

    phrases = PHRASES
    if args.phrases_file:
        phrases = [ln.strip() for ln in args.phrases_file.read_text().splitlines()
                   if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"{len(phrases)} phrases from {args.phrases_file}")

    token = os.environ.get("HA_TOKEN")
    if not token:
        print("HA_TOKEN not set", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    if args.holdout:
        args.holdout.mkdir(parents=True, exist_ok=True)
    # Split by *voice*, not by clip: a held-out clip from a voice that is also in
    # training measures memorisation of that speaker, not generalisation.
    holdout_voices = set(GOOGLE_VOICES[-args.holdout_voices:]) if args.holdout else set()
    if holdout_voices:
        print(f"holdout voices: {', '.join(sorted(holdout_voices))}")
    made = skipped = failed = 0

    for voice in GOOGLE_VOICES:
        target = args.holdout if voice in holdout_voices else args.out
        # A voice name the backend does not know fails identically for all 36
        # of its combinations, so one failure with nothing yet to its name is
        # enough to drop it.
        voice_ok = False
        voice_failed = 0

        for phrase in phrases:
            for speed in speeds:
                for pitch in pitches:
                    if not voice_ok and voice_failed >= 2:
                        break
                    if args.limit and made >= args.limit:
                        print(f"\nlimit reached: {made} clips")
                        return 0

                    options = {"voice": voice, "speed": speed, "pitch": pitch}
                    key = f"{args.engine}|{voice}|{phrase}|{speed}|{pitch}"
                    name = hashlib.sha1(key.encode()).hexdigest()[:16] + ".wav"
                    dest = target / name
                    if dest.exists():
                        skipped += 1
                        voice_ok = True
                        continue

                    url = tts_url(args.base, token, args.engine, phrase,
                                  args.language, options)
                    if url is not None and fetch_as_wav(url, dest):
                        made += 1
                        voice_ok = True
                        if made % 25 == 0:
                            print(f"  {made} clips", flush=True)
                    else:
                        failed += 1
                        voice_failed += 1

        if not voice_ok:
            print(f"  skipped voice {voice} (backend rejected it)")

    print(f"\n{made} new, {skipped} already present, {failed} failed -> {args.out}")
    print(f"total in directory: {len(list(args.out.glob('*.wav')))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
