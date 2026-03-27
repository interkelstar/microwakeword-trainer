#!/usr/bin/env python3
"""
Generate high-quality Russian wake word clips using ElevenLabs multilingual v2.

Outputs WAV files (16kHz mono int16) into training/output/<model_name>/elevenlabs_positive/
These get mixed into positive training data during feature extraction.

Usage:
    .venv/bin/python generate_elevenlabs.py --api-key <KEY> --config ru_jarvis.yaml
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import requests
import yaml

API_BASE = "https://api.elevenlabs.io/v1"

# Voices to use — diverse set of male/female voices
# All support Russian via eleven_multilingual_v2
VOICES = [
    ("CwhRBWXzGAHq8TQ4Fs17", "Roger"),      # male, laid-back
    ("EXAVITQu4vr4xnSDxMaL", "Sarah"),      # female, mature
    ("JBFqnCBsd6RMkjVDRZzb", "George"),      # male, warm
    ("IKne3meq5aSn9XLyUdCD", "Charlie"),     # male, deep
    ("Xb7hH8MSUJpSbSDYk0k2", "Alice"),      # female, clear
    ("onwK4e9ZLuTAKqWW03F9", "Daniel"),      # male, steady
    ("pFZP5JQG7iQjIQuC4Bku", "Lily"),       # female, velvety
    ("nPczCjzI2devNBz1zQrb", "Brian"),      # male, deep resonant
]

MODEL_ID = "eleven_multilingual_v2"


def check_credits(api_key: str) -> int:
    r = requests.get(f"{API_BASE}/user/subscription",
                     headers={"xi-api-key": api_key})
    r.raise_for_status()
    d = r.json()
    remaining = d["character_limit"] - d["character_count"]
    print(f"Credits: {d['character_count']}/{d['character_limit']} used, {remaining} remaining")
    return remaining


def generate_clip(api_key: str, voice_id: str, text: str,
                  stability: float, similarity_boost: float,
                  output_path: Path) -> bool:
    """Generate one clip via ElevenLabs API, convert to 16kHz WAV."""
    mp3_path = output_path.with_suffix(".mp3")

    r = requests.post(
        f"{API_BASE}/text-to-speech/{voice_id}",
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        },
        json={
            "text": text,
            "model_id": MODEL_ID,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        },
        timeout=30,
    )

    if r.status_code == 429:
        # Rate limited — wait and retry
        print("  Rate limited, waiting 60s...")
        time.sleep(60)
        return False

    if r.status_code != 200:
        print(f"  Error {r.status_code}: {r.text[:200]}")
        return False

    mp3_path.write_bytes(r.content)

    # Convert to 16kHz mono WAV
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_path), "-ar", "16000", "-ac", "1",
         str(output_path)],
        capture_output=True, timeout=10,
    )
    mp3_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  ffmpeg failed: {result.stderr[:200]}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--config", default="ru_jarvis.yaml")
    parser.add_argument("--max-clips", type=int, default=1200,
                        help="Maximum clips to generate (default: 1200)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model_name"]
    phrases = cfg["target_phrases"]

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("training/output") / model_name / "elevenlabs_positive"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check existing clips
    existing = len(list(out_dir.glob("*.wav")))
    if existing > 0:
        print(f"Found {existing} existing clips in {out_dir}")

    # Check credits
    remaining = check_credits(args.api_key)
    avg_chars = sum(len(p) for p in phrases) / len(phrases)
    max_affordable = int(remaining / avg_chars)
    max_clips = min(args.max_clips, max_affordable)
    estimated_chars = int(max_clips * avg_chars)

    print(f"Phrases: {phrases}")
    print(f"Avg chars/clip: {avg_chars:.1f}")
    print(f"Will generate: {max_clips} clips (~{estimated_chars} chars)")
    print(f"Voices: {len(VOICES)} ({', '.join(n for _, n in VOICES)})")
    print()

    # Build generation plan: distribute clips across voices and phrases
    # Vary stability and similarity_boost for diversity
    plan = []
    clips_per_voice = max_clips // len(VOICES)

    for voice_id, voice_name in VOICES:
        for i in range(clips_per_voice):
            phrase = phrases[i % len(phrases)]
            # Vary voice settings for diversity
            stability = random.uniform(0.25, 0.85)
            similarity = random.uniform(0.5, 0.95)
            plan.append((voice_id, voice_name, phrase, stability, similarity))

    # Fill remainder
    while len(plan) < max_clips:
        voice_id, voice_name = random.choice(VOICES)
        phrase = random.choice(phrases)
        plan.append((voice_id, voice_name, phrase,
                     random.uniform(0.25, 0.85), random.uniform(0.5, 0.95)))

    random.shuffle(plan)
    print(f"Generation plan: {len(plan)} clips")

    generated = 0
    failed = 0
    chars_used = 0

    for idx, (voice_id, voice_name, phrase, stab, sim) in enumerate(plan):
        clip_name = f"el_{voice_name}_{idx:04d}.wav"
        clip_path = out_dir / clip_name

        if clip_path.exists():
            generated += 1
            continue

        ok = generate_clip(args.api_key, voice_id, phrase, stab, sim, clip_path)
        if ok:
            generated += 1
            chars_used += len(phrase)
        else:
            failed += 1
            # Retry once on failure
            time.sleep(1)
            ok = generate_clip(args.api_key, voice_id, phrase, stab, sim, clip_path)
            if ok:
                generated += 1
                chars_used += len(phrase)
                failed -= 1

        if (idx + 1) % 50 == 0:
            print(f"  Progress: {generated}/{idx+1} generated, {failed} failed, "
                  f"~{chars_used} chars used")

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    print(f"\nDone! Generated: {generated}, Failed: {failed}")
    print(f"Characters used: ~{chars_used}")
    print(f"Output: {out_dir}")

    # Final credit check
    check_credits(args.api_key)


if __name__ == "__main__":
    main()
