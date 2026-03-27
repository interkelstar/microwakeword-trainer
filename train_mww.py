#!/usr/bin/env python3
"""
Standalone microWakeWord training pipeline.

Trains a streaming wake word model using the official microWakeWord package
(MixedNet/Inception architecture with Stream layers) and exports a uint8-
quantised TFLite model compatible with pymicro_wakeword and ESPHome.

Quick start:
    chmod +x setup_mww.sh && ./setup_mww.sh
    cp example_ru_jarvis.yaml my_wakeword.yaml
    # edit my_wakeword.yaml for your target phrase
    ./run_mww.sh --config my_wakeword.yaml

Individual phases:
    ./run_mww.sh --config my_wakeword.yaml --phase voices
    ./run_mww.sh --config my_wakeword.yaml --phase train
    ./run_mww.sh --config my_wakeword.yaml --phase export
    ./run_mww.sh --config my_wakeword.yaml --phase test

Phases (run in order for a fresh training):
    setup      Verify microwakeword package, TF, and Piper binary are available
    voices     Download Piper ONNX voice models from HuggingFace
    generate   Synthesise positive + adversarial negative TTS clips via piper
    features   Extract pymicro_features spectrograms + download HF negatives
    train      Build and train the MixedNet/Inception streaming model
    export     Convert to streaming TFLite (uint8 quantised) + write JSON manifest
    test       Evaluate the TFLite model on test WAV clips (not run in 'all')

Requires: setup_mww.sh (installs microwakeword, TensorFlow, pymicro-features, etc.)
Output:   <model_name>_mww.tflite + <model_name>_mww.json at the project root

Feature extraction uses pymicro_features.MicroFrontend (not librosa).
This is critical: inference and training must use the same frontend to avoid
a feature-scale mismatch that would prevent the model from ever firing.
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import types
import uuid
import zipfile
from pathlib import Path

import numpy as np
import yaml

# ===========================================================================
# Constants
# ===========================================================================

SAMPLE_RATE = 16_000
N_MEL_BINS = 40
CHUNK_SAMPLES = 160        # 10 ms at 16 kHz — pymicro_features fixed step
STEP_MS = 10               # feature extraction step matching pymicro_features

WORK_DIR = Path("training")
MODELS_DIR = WORK_DIR / "piper_models"
OUTPUT_DIR = WORK_DIR / "output"
PIPER_BIN_DIR = WORK_DIR / "piper_binary" / "piper"
PIPER_BIN = PIPER_BIN_DIR / "piper"
PIPER_ESPEAK = PIPER_BIN_DIR / "espeak-ng-data"

# ===========================================================================
# Logging
# ===========================================================================

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Absolute path to the Python interpreter running this script.
PYTHON = sys.executable


# ===========================================================================
# Config loading
# ===========================================================================

def load_config(config_path: str) -> dict:
    """Load YAML config and extract all MWW-relevant parameters."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw:
        raise ValueError(f"Config file is empty: {config_path}")
    if "model_name" not in raw:
        raise ValueError("Config missing required field: model_name")
    if "target_phrases" not in raw or not raw["target_phrases"]:
        raise ValueError("Config missing required field: target_phrases")

    voices = raw.get("voices", {})
    primary = voices.get("primary", {})
    secondary = voices.get("secondary", {})
    mww = raw.get("microwakeword", {})
    training = raw.get("training", {})

    return {
        "model_name": raw["model_name"],
        "wake_word": raw.get("wake_word", raw["model_name"]),
        "author": raw.get("author", ""),
        "trained_languages": raw.get("trained_languages", []),
        "version": raw.get("version", 1),
        "target_phrases": raw["target_phrases"],
        "negative_phrases": raw.get("negative_phrases", []),

        # Voice configuration
        "primary_voices_base_url": primary.get("base_url", ""),
        "primary_voices": primary.get("models", {}),
        "secondary_voices_base_url": secondary.get("base_url", ""),
        "secondary_voices": secondary.get("models", {}),

        # Training data counts
        "n_samples": training.get("n_samples", 50_000),
        "n_samples_val": training.get("n_samples_val", 10_000),

        # MWW architecture
        "architecture": mww.get("architecture", "mixednet"),
        "pointwise_filters": str(mww.get("pointwise_filters", "64,64,64,64")),
        "repeat_in_block": str(mww.get("repeat_in_block", "1,1,1,1")),
        "mixconv_kernel_sizes": str(
            mww.get("mixconv_kernel_sizes", "[5], [7,11], [9,15], [23]")
        ),
        "first_conv_filters": int(mww.get("first_conv_filters", 32)),
        "first_conv_kernel_size": int(mww.get("first_conv_kernel_size", 5)),
        "stride": int(mww.get("stride", 3)),
        "spectrogram_length": int(mww.get("spectrogram_length", 49)),

        # MWW training
        "training_steps": int(mww.get("training_steps", 10_000)),
        "batch_size": int(mww.get("batch_size", 128)),
        "learning_rate": float(mww.get("learning_rate", 0.001)),
        "positive_class_weight": float(mww.get("positive_class_weight", 1)),
        "negative_class_weight": float(mww.get("negative_class_weight", 20)),
        "clip_duration_ms": int(mww.get("clip_duration_ms", 1500)),

        # MWW SpecAugment
        "time_mask_max_size": int(mww.get("time_mask_max_size", 5)),
        "time_mask_count": int(mww.get("time_mask_count", 2)),
        "freq_mask_max_size": int(mww.get("freq_mask_max_size", 5)),
        "freq_mask_count": int(mww.get("freq_mask_count", 2)),

        # MWW output / inference
        "probability_cutoff": float(mww.get("probability_cutoff", 0.5)),
        "sliding_window_size": int(mww.get("sliding_window_size", 5)),
    }


# ===========================================================================
# Utilities
# ===========================================================================

def run(cmd: str, **kw):
    log.info("$ %s", cmd)
    subprocess.run(cmd, shell=True, check=True, **kw)


def wget(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 4096:
        log.info("  skip (exists): %s", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(f'wget -q --show-progress -O "{dest}" "{url}"')


def count_wav(d: Path) -> int:
    return len(list(d.glob("*.wav")))


def _is_latin(text: str) -> bool:
    latin = sum(1 for c in text if c.isalpha() and ord(c) < 256)
    total = sum(1 for c in text if c.isalpha())
    return total > 0 and latin / total > 0.5


def synthesize_clip(phrase: str, voice_onnx: str, out_path: Path,
                    length_scale: float = 1.0, noise_scale: float = 0.667,
                    noise_w: float = 0.8) -> bool:
    """Call Piper binary to synthesize one WAV clip. Returns True on success."""
    import scipy.io.wavfile as wavfile
    import scipy.signal

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = (
        str(PIPER_BIN_DIR.resolve()) + ":" + env.get("LD_LIBRARY_PATH", "")
    )
    env["ESPEAK_DATA_PATH"] = str(PIPER_ESPEAK.resolve())

    result = subprocess.run(
        [str(PIPER_BIN.resolve()), "--model", str(voice_onnx),
         "--output_file", str(out_path),
         "--length_scale", str(length_scale),
         "--noise_scale", str(noise_scale),
         "--noise_w", str(noise_w)],
        input=phrase, text=True, capture_output=True, env=env,
    )
    if result.returncode != 0 or not out_path.exists() or out_path.stat().st_size <= 44:
        return False

    # Resample to 16 kHz if the voice model outputs a different rate
    sr, data = wavfile.read(str(out_path))
    if sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / sr)
        data = scipy.signal.resample(data, n).astype(data.dtype)
        wavfile.write(str(out_path), SAMPLE_RATE, data)
    return True


# ===========================================================================
# Feature extraction  (pymicro_features — matches inference exactly)
# ===========================================================================

def extract_all_frames(audio_int16: np.ndarray) -> np.ndarray:
    """Run pymicro_features MicroFrontend on audio. Returns (N, 40) float32."""
    from pymicro_features import MicroFrontend

    frontend = MicroFrontend()
    frames = []
    for i in range(0, len(audio_int16) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        chunk = audio_int16[i:i + CHUNK_SAMPLES].astype(np.int16).tobytes()
        result = frontend.process_samples(chunk)
        if result is not None and len(result.features) > 0:
            frames.append(list(result.features))
    if not frames:
        return np.zeros((1, N_MEL_BINS), dtype=np.float32)
    return np.array(frames, dtype=np.float32)


def _get_window(frames: np.ndarray, start: int, spec_length: int) -> np.ndarray:
    """Extract a spec_length-sized window, zero-pad if short."""
    window = frames[start:start + spec_length]
    if len(window) < spec_length:
        window = np.pad(window, ((0, spec_length - len(window)), (0, 0)))
    return window.astype(np.float32)


def _augment_audio(audio: np.ndarray, n_augments: int = 3) -> list:
    """Create augmented copies of audio with noise and speed perturbation.

    Returns list of int16 arrays (original + augmented versions).
    """
    import scipy.signal

    results = [audio]  # always include original

    for _ in range(n_augments):
        aug = audio.astype(np.float32)

        # Random speed perturbation (±10%)
        if random.random() < 0.5:
            speed = random.uniform(0.9, 1.1)
            n_out = int(len(aug) / speed)
            if n_out > 100:
                aug = scipy.signal.resample(aug, n_out).astype(np.float32)

        # Add Gaussian noise at random SNR (10-30 dB)
        if random.random() < 0.7:
            signal_power = np.mean(aug ** 2) + 1e-10
            snr_db = random.uniform(10, 30)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), len(aug))
            aug = aug + noise

        # Random volume change (±6 dB)
        if random.random() < 0.5:
            gain_db = random.uniform(-6, 6)
            aug = aug * (10 ** (gain_db / 20))

        results.append(np.clip(aug, -32768, 32767).astype(np.int16))

    return results


def extract_features_from_dir(
    wav_dir: Path,
    desc: str,
    spec_length: int,
    max_samples: int = None,
    augment_stride: int = None,
    noise_augments: int = 0,
    min_energy: float = 0.5,
) -> np.ndarray:
    """Extract features from all WAV files in a directory.

    Args:
        spec_length: number of time frames per training sample
        augment_stride: if set, slide a window with this step for temporal augmentation
        noise_augments: number of noise-augmented copies per clip (0 = no noise aug)
        min_energy: skip augmented windows below this mean value (avoids silence-as-positive)

    Returns:
        (N, spec_length, 40) float32 array
    """
    import scipy.io.wavfile as wavfile
    from tqdm import tqdm

    wav_files = sorted(wav_dir.glob("*.wav"))
    if max_samples:
        wav_files = wav_files[:max_samples]
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {wav_dir}")

    log.info("  Extracting: %d files (%s, augment=%s, noise_aug=%d)",
             len(wav_files), desc, augment_stride is not None, noise_augments)
    features, failed = [], 0

    for wav_path in tqdm(wav_files, desc=f"[{desc}]", unit="clip"):
        try:
            sr, audio = wavfile.read(str(wav_path))
            if audio.ndim > 1:
                audio = audio[:, 0]
            if sr != SAMPLE_RATE:
                import scipy.signal
                n = int(len(audio) * SAMPLE_RATE / sr)
                audio = scipy.signal.resample(audio, n).astype(np.int16)

            # Create audio variants (original + noise-augmented)
            if noise_augments > 0:
                audio_variants = _augment_audio(audio.astype(np.int16), noise_augments)
            else:
                audio_variants = [audio.astype(np.int16)]

            for audio_var in audio_variants:
                frames = extract_all_frames(audio_var)
                n = len(frames)

                if augment_stride and augment_stride > 0:
                    stop = max(1, n - spec_length + 1)
                    for start in range(0, stop, augment_stride):
                        w = _get_window(frames, start, spec_length)
                        if w.mean() >= min_energy:
                            features.append(w)
                    centre = max(0, (n - spec_length) // 2)
                    features.append(_get_window(frames, centre, spec_length))
                else:
                    centre = max(0, (n - spec_length) // 2)
                    features.append(_get_window(frames, centre, spec_length))
        except Exception as exc:
            failed += 1
            log.debug("  Skip %s: %s", wav_path.name, exc)

    if failed:
        log.warning("  %d / %d files failed", failed, len(wav_files))

    return np.array(features, dtype=np.float32)


# ===========================================================================
# Phase: setup
# ===========================================================================

def phase_setup(cfg: dict):
    log.info("=" * 60)
    log.info("Phase: setup — verify environment")
    log.info("=" * 60)

    # Check Piper binary
    if not PIPER_BIN.exists():
        raise FileNotFoundError(
            f"Piper binary not found: {PIPER_BIN}\n"
            "Run setup_mww.sh first."
        )
    log.info("  Piper binary: %s", PIPER_BIN)

    # Check microwakeword
    try:
        import microwakeword  # noqa: F401
        log.info("  microwakeword: installed")
    except ImportError:
        raise ImportError(
            "microwakeword package not installed.\n"
            "Run setup_mww.sh first, or:\n"
            "  pip install git+https://github.com/kahrendt/microWakeWord.git"
        )

    # Check TensorFlow
    try:
        import tensorflow as tf
        log.info("  TensorFlow: %s", tf.__version__)
    except ImportError:
        raise ImportError("TensorFlow not found. Run setup_mww.sh first.")

    # Check pymicro_features
    try:
        from pymicro_features import MicroFrontend  # noqa: F401
        log.info("  pymicro_features: available")
    except ImportError:
        raise ImportError(
            "pymicro_features not installed.\n"
            "Run: pip install pymicro-features"
        )

    log.info("  Setup OK.")


# ===========================================================================
# Phase: voices
# ===========================================================================

def phase_voices(cfg: dict) -> list[str]:
    log.info("=" * 60)
    log.info("Phase: voices — downloading Piper voices")
    log.info("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    available = []

    for name, suffix in cfg["primary_voices"].items():
        onnx = MODELS_DIR / f"{name}.onnx"
        json_f = MODELS_DIR / f"{name}.onnx.json"
        base = f"{cfg['primary_voices_base_url']}/{suffix}/{name}"
        try:
            wget(f"{base}.onnx", onnx)
            wget(f"{base}.onnx.json", json_f)
            available.append(str(onnx.resolve()))
            log.info("  ✓ %s", name)
        except subprocess.CalledProcessError as exc:
            log.warning("  Failed: %s (%s)", name, exc)

    if not available:
        raise RuntimeError("No primary voices downloaded. Check internet connection.")
    log.info("  %d primary voice(s) available", len(available))

    # Secondary voices (cross-language negatives)
    if cfg["secondary_voices"] and cfg["secondary_voices_base_url"]:
        for name, suffix in cfg["secondary_voices"].items():
            onnx = MODELS_DIR / f"{name}.onnx"
            json_f = MODELS_DIR / f"{name}.onnx.json"
            base = f"{cfg['secondary_voices_base_url']}/{suffix}/{name}"
            try:
                wget(f"{base}.onnx", onnx)
                wget(f"{base}.onnx.json", json_f)
                log.info("  ✓ %s (secondary)", name)
            except subprocess.CalledProcessError:
                log.warning("  Failed: %s", name)

    return available


# ===========================================================================
# Phase: generate  (inline Piper TTS — no OWW dependency)
# ===========================================================================

def _generate_clips(phrases: list, voice_files: list, n_total: int,
                    out_dir: Path, desc: str):
    """Generate n_total TTS clips from phrases × voices into out_dir."""
    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = count_wav(out_dir)
    if existing >= n_total:
        log.info("  %s: %d clips exist (target %d) — skip", desc, existing, n_total)
        return
    remaining = n_total - existing

    length_scales = [0.75, 1.0, 1.1, 1.25]
    noise_scales = [0.667, 0.8, 0.98]
    noise_ws = [0.8, 0.9, 0.98]

    combos = [
        (p, str(v), ls, ns, nw)
        for p in phrases
        for v in voice_files
        for ls in length_scales
        for ns in noise_scales
        for nw in noise_ws
    ]
    random.shuffle(combos)

    generated, idx = 0, 0
    pbar = tqdm(total=remaining, desc=desc, unit="clip")
    while generated < remaining:
        phrase, voice, ls, ns, nw = combos[idx % len(combos)]
        idx += 1
        out_path = out_dir / f"{uuid.uuid4().hex}.wav"
        if synthesize_clip(phrase, voice, out_path, ls, ns, nw):
            generated += 1
            pbar.update(1)
    pbar.close()
    log.info("  %s: %d new clips", desc, generated)


def phase_generate(cfg: dict):
    log.info("=" * 60)
    log.info("Phase: generate — TTS clip generation")
    log.info("=" * 60)

    model_name = cfg["model_name"]
    base = OUTPUT_DIR / model_name

    # Collect voice files
    primary_voices = [
        MODELS_DIR / f"{name}.onnx"
        for name in cfg["primary_voices"]
        if (MODELS_DIR / f"{name}.onnx").exists()
    ]
    secondary_voices = [
        MODELS_DIR / f"{name}.onnx"
        for name in cfg.get("secondary_voices", {})
        if (MODELS_DIR / f"{name}.onnx").exists()
    ]
    if not primary_voices:
        raise FileNotFoundError(
            "No primary voice models found. Run --phase voices first."
        )

    n_train = cfg["n_samples"]
    n_val = cfg["n_samples_val"]

    # --- Positive clips ---
    _generate_clips(
        cfg["target_phrases"], primary_voices, n_train,
        base / "positive_train", "pos_train",
    )
    _generate_clips(
        cfg["target_phrases"], primary_voices, n_val,
        base / "positive_test", "pos_test",
    )

    # --- Adversarial negative clips ---
    neg_phrases = cfg["negative_phrases"]
    if not neg_phrases:
        log.info("  No negative_phrases configured — skipping adversarial negatives")
        return

    cyrillic = [p for p in neg_phrases if not _is_latin(p)]
    latin = [p for p in neg_phrases if _is_latin(p)]

    n_neg_train = n_train // 2
    n_neg_val = n_val // 2

    if cyrillic and primary_voices:
        _generate_clips(
            cyrillic, primary_voices, n_neg_train,
            base / "negative_train", "neg_train_cyr",
        )
        _generate_clips(
            cyrillic, primary_voices, n_neg_val,
            base / "negative_test", "neg_test_cyr",
        )
    if latin and secondary_voices:
        _generate_clips(
            latin, secondary_voices, max(1, n_neg_train // 4),
            base / "negative_train", "neg_train_lat",
        )
        _generate_clips(
            latin, secondary_voices, max(1, n_neg_val // 4),
            base / "negative_test", "neg_test_lat",
        )

    log.info("  Generation complete.")


# ===========================================================================
# Phase: features
# ===========================================================================

def _download_hf_negatives(feat_dir: Path, spec_length: int,
                           max_windows: int = 200_000) -> np.ndarray | None:
    """Load pre-computed spectrograms from kahrendt/microwakeword HF dataset.

    The HF dataset contains RaggedMmap directories (not WAV files) inside ZIPs.
    Values are uint16 (out_scale=1 in official micro frontend) — we scale them
    to match pymicro_features float32 range [0, ~26] by dividing by ~24.7.
    """
    log.info("  Loading HF negative datasets (kahrendt/microwakeword)...")

    # Scale factor: official uint16 (out_scale=1) → pymicro_features float32
    UINT16_TO_FLOAT_SCALE = 24.7

    try:
        from huggingface_hub import hf_hub_download
        from mmap_ninja.ragged import RaggedMmap
        from tqdm import tqdm

        hf_audio_dir = feat_dir / "hf_audio"
        hf_audio_dir.mkdir(parents=True, exist_ok=True)
        all_features = []

        for zip_name in ["speech.zip", "no_speech.zip", "dinner_party.zip"]:
            try:
                log.info("    Processing %s ...", zip_name)
                zip_path = hf_hub_download(
                    "kahrendt/microwakeword", zip_name,
                    repo_type="dataset",
                    local_dir=str(feat_dir / "hf_cache"),
                )

                extract_dir = hf_audio_dir / zip_name.replace(".zip", "")
                if not extract_dir.exists() or not list(extract_dir.rglob("*.ninja")):
                    extract_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(extract_dir)

                # Find all RaggedMmap directories (contain data.ninja)
                mmap_dirs = sorted(
                    p.parent for p in extract_dir.rglob("data.ninja")
                    if "training" in str(p) and "_mmap" in str(p.parent)
                )
                log.info("    %s: %d mmap directories", zip_name, len(mmap_dirs))

                for mmap_dir in mmap_dirs:
                    try:
                        rm = RaggedMmap(str(mmap_dir))
                        n_items = len(rm)
                        # Sample if too many
                        indices = list(range(n_items))
                        if n_items > max_windows // (3 * max(1, len(mmap_dirs))):
                            random.shuffle(indices)
                            indices = indices[:max_windows // (3 * max(1, len(mmap_dirs)))]

                        for idx in tqdm(indices, desc=mmap_dir.name, unit="spec"):
                            spec = rm[idx].astype(np.float32) / UINT16_TO_FLOAT_SCALE
                            # spec is (500, 40) — extract windows of spec_length
                            n_frames = spec.shape[0]
                            step = max(1, spec_length)
                            for start in range(0, max(1, n_frames - spec_length + 1), step):
                                w = spec[start:start + spec_length]
                                if len(w) == spec_length:
                                    all_features.append(w)
                            if len(all_features) >= max_windows:
                                break
                    except Exception as exc:
                        log.warning("    Failed mmap %s: %s", mmap_dir.name, exc)

                    if len(all_features) >= max_windows:
                        break

            except Exception as exc:
                log.warning("    Failed to process %s: %s", zip_name, exc)

            if len(all_features) >= max_windows:
                break

        if all_features:
            result = np.array(all_features[:max_windows], dtype=np.float32)
            log.info("  HF negatives: %d windows (spec_length=%d)", len(result), spec_length)
            return result

        log.warning("  No HF negative features extracted")
        return None

    except ImportError as exc:
        log.warning("  Required package not available (%s) — skipping HF negatives", exc)
        return None


def phase_features(cfg: dict):
    log.info("=" * 60)
    log.info("Phase: features — spectrogram extraction")
    log.info("=" * 60)

    model_name = cfg["model_name"]
    base = OUTPUT_DIR / model_name
    feat_dir = base / "mww_features"
    feat_dir.mkdir(parents=True, exist_ok=True)

    spec_length = cfg["spectrogram_length"]
    augment_stride = max(1, spec_length // 12)

    # --- Positive features ---
    pos_train_npy = feat_dir / "positive_train.npy"
    if pos_train_npy.exists():
        log.info("  positive_train.npy exists — skipping")
    else:
        pos_arrays = []

        # 1. Piper TTS clips (bulk) — light noise augmentation
        d = base / "positive_train"
        if not d.exists() or not list(d.glob("*.wav")):
            raise FileNotFoundError(
                f"No positive clips in {d}. Run --phase generate first."
            )
        piper_feats = extract_features_from_dir(
            d, "piper_positive", spec_length,
            augment_stride=augment_stride, noise_augments=2,
        )
        pos_arrays.append(piper_feats)

        # 2. ElevenLabs clips (high-quality TTS, if available) — light noise aug
        el_dir = base / "elevenlabs_positive"
        if el_dir.exists() and list(el_dir.glob("*.wav")):
            el_feats = extract_features_from_dir(
                el_dir, "elevenlabs_pos", spec_length,
                augment_stride=augment_stride, noise_augments=3,
            )
            pos_arrays.append(el_feats)
            log.info("  ElevenLabs positives: %s", el_feats.shape)

        # 3. Real recordings (if available in record/ directory)
        #    Only included if train_with_recordings: true in YAML.
        #    WARNING: recordings with ambient silence around the word will
        #    poison positives with room-specific mic noise — use for test only.
        rec_dir = Path("record")
        if cfg.get("train_with_recordings", True) and rec_dir.exists() and list(rec_dir.glob("*.wav")):
            rec_feats = extract_features_from_dir(
                rec_dir, "real_recordings", spec_length,
                augment_stride=augment_stride, noise_augments=10,
            )
            pos_arrays.append(rec_feats)
            log.info("  Real recordings: %s", rec_feats.shape)
        elif not cfg.get("train_with_recordings", True):
            log.info("  Real recordings: skipped (train_with_recordings=false)")

        pos_train = np.concatenate(pos_arrays, axis=0)
        np.random.shuffle(pos_train)
        np.save(str(pos_train_npy), pos_train)
        log.info("  positive_train.npy: %s (combined)", pos_train.shape)

    pos_val_npy = feat_dir / "positive_val.npy"
    if pos_val_npy.exists():
        log.info("  positive_val.npy exists — skipping")
    else:
        d = base / "positive_test"
        if d.exists() and list(d.glob("*.wav")):
            pos_val = extract_features_from_dir(d, "positive_val", spec_length)
            np.save(str(pos_val_npy), pos_val)
            log.info("  positive_val.npy: %s", pos_val.shape)
        else:
            log.warning("  No positive_test dir — splitting positive_train 90/10")
            pt = np.load(str(pos_train_npy))
            split = int(len(pt) * 0.9)
            np.save(str(pos_train_npy), pt[:split])
            np.save(str(pos_val_npy), pt[split:])

    # --- Negative features ---
    neg_train_npy = feat_dir / "negative_train.npy"
    neg_val_npy = feat_dir / "negative_val.npy"

    if neg_train_npy.exists() and neg_val_npy.exists():
        log.info("  negative_*.npy exist — skipping")
        return

    neg_arrays = []

    # 1. Adversarial negatives from generated clips
    neg_train_dir = base / "negative_train"
    if neg_train_dir.exists() and list(neg_train_dir.glob("*.wav")):
        adv = extract_features_from_dir(
            neg_train_dir, "adversarial_neg", spec_length,
            noise_augments=2,
        )
        neg_arrays.append(adv)
        log.info("  Adversarial negatives: %s", adv.shape)

    # 2. User-supplied real-environment negatives (densely windowed)
    user_neg_dir = base / "user_negatives"
    if user_neg_dir.exists() and list(user_neg_dir.glob("*.wav")):
        user_neg = extract_features_from_dir(
            user_neg_dir, "user_negatives", spec_length,
            augment_stride=1,  # dense: ~1 window per 10ms → maximize coverage
        )
        neg_arrays.append(user_neg)
        log.info("  User negatives: %s", user_neg.shape)

    # 3. HF negatives (speech, no_speech, dinner_party)
    hf_neg = _download_hf_negatives(feat_dir, spec_length, max_windows=500_000)
    if hf_neg is not None:
        neg_arrays.append(hf_neg)

    if not neg_arrays:
        raise RuntimeError(
            "No negative features available.\n"
            "Generate adversarial clips (--phase generate) or ensure HF download works."
        )

    all_neg = np.concatenate(neg_arrays, axis=0)
    np.random.shuffle(all_neg)

    split = int(len(all_neg) * 0.9)
    np.save(str(neg_train_npy), all_neg[:split])
    np.save(str(neg_val_npy), all_neg[split:])
    log.info("  negative_train: %d, negative_val: %d", split, len(all_neg) - split)
    log.info("  Feature extraction complete.")


# ===========================================================================
# Phase: train
# ===========================================================================

def _build_model(cfg: dict):
    """Build a MixedNet or Inception model using official microwakeword builders.

    The model includes Stream layers that enable streaming inference conversion.
    Flags are passed as strings — the builder parses them internally
    (kws_streaming convention).
    """
    spec_length = cfg["spectrogram_length"]
    batch_size = cfg["batch_size"]
    shape = (spec_length, N_MEL_BINS)
    arch = cfg["architecture"]

    if arch == "mixednet":
        flags = types.SimpleNamespace(
            pointwise_filters=cfg["pointwise_filters"],
            repeat_in_block=cfg["repeat_in_block"],
            mixconv_kernel_sizes=cfg["mixconv_kernel_sizes"],
            residual_connection="True,True,True,True",
            first_conv_filters=cfg["first_conv_filters"],
            first_conv_kernel_size=cfg["first_conv_kernel_size"],
            spatial_attention=False,
            pooled=True,
            stride=cfg["stride"],
            max_pool=False,
        )
        from microwakeword.mixednet import model as build_fn
    elif arch == "inception":
        flags = types.SimpleNamespace(
            cnn1_filters=cfg.get("cnn1_filters", 64),
            cnn1_kernel_sizes=str(cfg.get("cnn1_kernel_sizes", "[3,5]")),
            cnn2_filters1=cfg.get("cnn2_filters1", 128),
            cnn2_filters2=cfg.get("cnn2_filters2", 128),
            cnn2_kernel_sizes=str(cfg.get("cnn2_kernel_sizes", "[3,5]")),
            cnn2_subspectral_groups=cfg.get("cnn2_subspectral_groups", 2),
            cnn2_dilation=str(cfg.get("cnn2_dilation", "1,1")),
            dropout=cfg.get("dropout", 0.1),
        )
        from microwakeword.inception import model as build_fn
    else:
        raise ValueError(f"Unknown architecture: {arch!r} (must be mixednet or inception)")

    log.info("  Building %s model (shape=%s, batch=%d)", arch, shape, batch_size)
    return build_fn(flags, shape, batch_size)


def _make_spec_augment_fn(spec_length: int, cfg: dict):
    """Return a tf.data-compatible SpecAugment map function."""
    import tensorflow as tf

    tm_max = cfg["time_mask_max_size"]
    tm_cnt = cfg["time_mask_count"]
    fm_max = cfg["freq_mask_max_size"]
    fm_cnt = cfg["freq_mask_count"]

    def spec_augment(features, label):
        feat = features  # (spec_length, 40)
        for _ in range(tm_cnt):
            t = tf.random.uniform([], 1, tm_max + 1, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, tf.maximum(1, spec_length - t), dtype=tf.int32)
            mask = tf.cast(
                tf.logical_or(
                    tf.range(spec_length) < t0,
                    tf.range(spec_length) >= t0 + t,
                ),
                tf.float32,
            )
            feat = feat * tf.reshape(mask, [spec_length, 1])
        for _ in range(fm_cnt):
            f = tf.random.uniform([], 1, fm_max + 1, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, tf.maximum(1, N_MEL_BINS - f), dtype=tf.int32)
            mask = tf.cast(
                tf.logical_or(
                    tf.range(N_MEL_BINS) < f0,
                    tf.range(N_MEL_BINS) >= f0 + f,
                ),
                tf.float32,
            )
            feat = feat * tf.reshape(mask, [1, N_MEL_BINS])
        return feat, label

    return spec_augment


def phase_train(cfg: dict):
    log.info("=" * 60)
    log.info("Phase: train — model training")
    log.info("=" * 60)

    import tensorflow as tf

    model_name = cfg["model_name"]
    base = OUTPUT_DIR / model_name
    feat_dir = base / "mww_features"
    spec_length = cfg["spectrogram_length"]

    # --- Load features ---
    for fname in ("positive_train.npy", "positive_val.npy",
                  "negative_train.npy", "negative_val.npy"):
        p = feat_dir / fname
        if not p.exists():
            raise FileNotFoundError(
                f"Feature file missing: {p}\nRun --phase features first."
            )

    # Use mmap to avoid loading 14+ GB into RAM all at once
    pos_train = np.load(str(feat_dir / "positive_train.npy"), mmap_mode='r')
    neg_train = np.load(str(feat_dir / "negative_train.npy"), mmap_mode='r')
    # Val arrays are small (~700 MB total) — load fully
    pos_val = np.load(str(feat_dir / "positive_val.npy"))
    neg_val = np.load(str(feat_dir / "negative_val.npy"))

    log.info("  pos_train: %s  pos_val: %s", pos_train.shape, pos_val.shape)
    log.info("  neg_train: %s  neg_val: %s", neg_train.shape, neg_val.shape)

    # Val: small enough to concat in memory
    X_val = np.concatenate([pos_val, neg_val]).astype(np.float32)
    y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))]).astype(np.float32)

    n_pos = len(pos_train)
    n_neg = len(neg_train)
    log.info("  Train: %d pos + %d neg (generator)  Val: %d", n_pos, n_neg, len(X_val))

    # --- Build model ---
    model = _build_model(cfg)
    model.summary(print_fn=log.info)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    class_weight = {
        0: cfg["negative_class_weight"],
        1: cfg["positive_class_weight"],
    }
    log.info("  Class weights: %s", class_weight)

    # --- tf.data pipeline with SpecAugment ---
    batch_size = cfg["batch_size"]
    training_steps = cfg["training_steps"]
    steps_per_epoch = max(1, (n_pos + n_neg) // batch_size)
    epochs = max(1, training_steps // steps_per_epoch)

    spec_augment = _make_spec_augment_fn(spec_length, cfg)

    # Generator-based pipeline: reads from mmap arrays without loading all into RAM.
    # Alternates pos/neg with shuffled index arrays so the model sees balanced batches.
    def _train_gen():
        pos_idx = np.arange(n_pos)
        neg_idx = np.arange(n_neg)
        np.random.shuffle(pos_idx)
        np.random.shuffle(neg_idx)
        pi = ni = 0
        while True:
            if pi >= n_pos:
                np.random.shuffle(pos_idx)
                pi = 0
            if ni >= n_neg:
                np.random.shuffle(neg_idx)
                ni = 0
            # Alternate pos / neg
            yield pos_train[pos_idx[pi]].astype(np.float32), np.float32(1.0)
            pi += 1
            yield neg_train[neg_idx[ni]].astype(np.float32), np.float32(0.0)
            ni += 1

    train_ds = tf.data.Dataset.from_generator(
        _train_gen,
        output_signature=(
            tf.TensorSpec(shape=(spec_length, 40), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float32),
        ),
    )
    train_ds = (
        train_ds
        .shuffle(10_000)
        .map(spec_augment, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=max(5, epochs // 5),
            restore_best_weights=True,
            mode="min",
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(3, epochs // 10),
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    log.info("  Training: %d steps (%d epochs × %d steps/epoch, batch=%d)",
             training_steps, epochs, steps_per_epoch, batch_size)

    model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=(X_val, y_val),
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    # --- Save model ---
    model_dir = base / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    keras_path = model_dir / f"{model_name}_mww.keras"
    weights_path = model_dir / f"{model_name}_mww.weights.h5"

    model.save(str(keras_path))
    log.info("  Model saved: %s", keras_path)
    try:
        model.save_weights(str(weights_path))
        log.info("  Weights saved: %s", weights_path)
    except Exception:
        pass

    # --- Final metrics ---
    val_metrics = model.evaluate(X_val, y_val, verbose=0)
    log.info("=" * 60)
    log.info("  TRAINING RESULTS (validation)")
    for name, val in zip(model.metrics_names, val_metrics):
        log.info("  %-20s %.4f", name, val)
    log.info("=" * 60)


# ===========================================================================
# Phase: export
# ===========================================================================

def phase_export(cfg: dict):
    log.info("=" * 60)
    log.info("Phase: export — streaming TFLite conversion")
    log.info("=" * 60)

    import tensorflow as tf

    model_name = cfg["model_name"]
    base = OUTPUT_DIR / model_name
    model_dir = base / "model"
    spec_length = cfg["spectrogram_length"]
    stride = cfg["stride"]

    # --- Load trained model ---
    keras_path = model_dir / f"{model_name}_mww.keras"
    weights_path = model_dir / f"{model_name}_mww.weights.h5"

    model = None
    for path in [keras_path]:
        if path.exists():
            try:
                model = tf.keras.models.load_model(str(path))
                log.info("  Loaded model: %s", path)
                break
            except Exception as exc:
                log.warning("  Failed to load %s: %s", path, exc)

    if model is None and weights_path.exists():
        log.info("  Rebuilding model and loading weights...")
        model = _build_model(cfg)
        model.load_weights(str(weights_path))
        log.info("  Loaded weights: %s", weights_path)

    if model is None:
        raise FileNotFoundError(
            f"No trained model found in {model_dir}. Run --phase train first."
        )

    # --- Convert to streaming ---
    streaming_model = None
    try:
        from microwakeword.utils import to_streaming_inference
        from microwakeword.layers.modes import Modes

        streaming_config = {
            "spectrogram_length": spec_length,
            "features_length": spec_length,
            "stride": stride,
        }
        streaming_model = to_streaming_inference(
            model, streaming_config,
            mode=Modes.STREAM_INTERNAL_STATE_INFERENCE,
        )
        log.info("  Streaming conversion OK")
        log.info("  Streaming input:  %s", streaming_model.input_shape)
        log.info("  Streaming output: %s", streaming_model.output_shape)
    except Exception as exc:
        log.warning("  Streaming conversion failed: %s", exc)
        log.warning("  Falling back to non-streaming model (may produce binary outputs)")

    export_model = streaming_model if streaming_model is not None else model

    # --- Representative dataset for quantisation ---
    feat_dir = base / "mww_features"
    pos_train = np.load(str(feat_dir / "positive_train.npy"))

    # Determine input time dimension for the export model
    input_shape = export_model.input_shape
    input_time = input_shape[1] if input_shape[1] is not None else spec_length

    def _rep_dataset():
        samples = pos_train[:500]
        # Ensure min/max of feature range are represented
        samples[0][0, 0] = 0.0
        samples[0][0, 1] = 26.0
        for s in samples:
            # For streaming model: yield stride-sized chunks
            for i in range(0, s.shape[0] - input_time, input_time):
                chunk = s[i : i + input_time].astype(np.float32)
                yield [chunk[np.newaxis]]

    # --- TFLite conversion (uint8 quantised) ---
    log.info("  Converting to TFLite (uint8 quantised)...")

    # Monkey-patch TF's is_tf_type to handle _DictWrapper __dict__ bug in TF 2.19/2.20
    from tensorflow.python.framework import tensor_util
    _orig_is_tf_type = tensor_util.is_tf_type
    def _safe_is_tf_type(x):
        try:
            return _orig_is_tf_type(x)
        except TypeError:
            return False
    tensor_util.is_tf_type = _safe_is_tf_type

    # Save as SavedModel — use ExportArchive with explicit input signature
    # (same approach as official microwakeword)
    saved_model_dir = str(model_dir / "saved_model_streaming")
    export_archive = tf.keras.export.ExportArchive()
    export_archive.track(export_model)
    export_archive.add_endpoint(
        name="serve",
        fn=export_model.call,
        input_signature=[tf.TensorSpec(shape=export_model.input.shape, dtype=tf.float32)],
    )
    export_archive.write_out(saved_model_dir)
    log.info("  SavedModel exported: %s", saved_model_dir)

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = {tf.lite.Optimize.DEFAULT}
    converter._experimental_variable_quantization = True
    converter.target_spec.supported_ops = {tf.lite.OpsSet.TFLITE_BUILTINS_INT8}
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = tf.lite.RepresentativeDataset(_rep_dataset)

    try:
        tflite_model = converter.convert()
    except Exception as exc:
        log.warning("  INT8 quantisation failed (%s), trying DEFAULT optimisation", exc)
        converter2 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter2.optimizations = [tf.lite.Optimize.DEFAULT]
        try:
            tflite_model = converter2.convert()
        except Exception as exc2:
            log.warning("  DEFAULT optimisation failed (%s), trying no optimisation", exc2)
            converter3 = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            tflite_model = converter3.convert()

    tflite_name = f"{model_name}_mww.tflite"
    tflite_path = base / tflite_name
    tflite_path.write_bytes(tflite_model)
    size_kb = tflite_path.stat().st_size / 1024
    log.info("  TFLite saved: %s (%.1f KB)", tflite_path, size_kb)

    # --- Copy to project root ---
    dest = Path(".") / tflite_name
    shutil.copy(str(tflite_path), str(dest))
    log.info("  Copied: %s", dest)

    # --- JSON manifest ---
    manifest = {
        "type": "micro",
        "wake_word": cfg.get("wake_word", model_name),
        "author": cfg.get("author", ""),
        "model": tflite_name,
        "trained_languages": cfg.get("trained_languages", []),
        "version": cfg.get("version", 1),
        "micro": {
            "probability_cutoff": cfg["probability_cutoff"],
            "sliding_window_size": cfg["sliding_window_size"],
            "feature_step_size": STEP_MS,
            "tensor_arena_size": 20000,
            "minimum_esphome_version": "2024.2.0",
        }
    }
    json_name = f"{model_name}_mww.json"
    json_path = Path(".") / json_name
    with open(json_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    log.info("  JSON: %s", json_path)

    log.info("=" * 60)
    log.info("  Export complete!")
    log.info("  Model: %s (%.1f KB)", dest, size_kb)
    log.info("  JSON:  %s", json_path)
    if streaming_model is not None:
        log.info("  Mode:  STREAMING (smooth probability curves)")
    else:
        log.info("  Mode:  NON-STREAMING (binary outputs — check logs above)")
    log.info("=" * 60)


# ===========================================================================
# Phase: test
# ===========================================================================

def phase_test(cfg: dict):
    log.info("=" * 60)
    log.info("Phase: test — model evaluation")
    log.info("=" * 60)

    model_name = cfg["model_name"]
    tflite_path = Path(f"{model_name}_mww.tflite")

    if not tflite_path.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {tflite_path}. Run --phase export first."
        )

    # Try official inference API
    try:
        from microwakeword.inference import Model as MWWModel
        mww = MWWModel(str(tflite_path))
        log.info("  Using microwakeword.inference.Model")
        _test_with_official(mww, cfg)
    except ImportError:
        log.info("  microwakeword.inference not available — using manual test")
        _test_manual(cfg)


def _test_with_official(mww, cfg: dict):
    """Evaluate using the official inference Model class."""
    import scipy.io.wavfile as wavfile

    model_name = cfg["model_name"]
    base = OUTPUT_DIR / model_name

    for label, subdir in [("positive", "positive_test"), ("negative", "negative_test")]:
        d = base / subdir
        if not d.exists():
            continue
        wav_files = sorted(d.glob("*.wav"))[:100]
        scores = []
        for f in wav_files:
            try:
                sr, audio = wavfile.read(str(f))
                if audio.ndim > 1:
                    audio = audio[:, 0]
                preds = mww.predict_clip(audio.astype(np.int16))
                if preds:
                    scores.append(max(preds))
            except Exception:
                pass
        if scores:
            log.info("  %s (%d clips): mean=%.3f  max=%.3f  min=%.3f",
                     label, len(scores), np.mean(scores), max(scores), min(scores))


def _test_manual(cfg: dict):
    """Fallback test using raw TFLite interpreter."""
    log.info("  Manual test: python3 test_mww.py record/sample*.wav")
    wav_dir = Path("record")
    if wav_dir.exists():
        n = count_wav(wav_dir)
        if n > 0:
            log.info("  Found %d WAV files in record/ — run test_mww.py manually", n)


# ===========================================================================
# Main
# ===========================================================================

ALL_PHASES = ["setup", "voices", "generate", "features", "train", "export", "test"]


def main():
    parser = argparse.ArgumentParser(
        description="Standalone microWakeWord training pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument(
        "--phase", choices=["all"] + ALL_PHASES, default="all",
        help="Which phase to run (default: all except test)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_all = args.phase == "all"

    log.info("=" * 60)
    log.info("microWakeWord Trainer — %s", cfg["model_name"])
    log.info("  Architecture:    %s", cfg["architecture"])
    log.info("  Spectrogram:     %d frames × %d bins (%.0f ms context)",
             cfg["spectrogram_length"], N_MEL_BINS,
             cfg["spectrogram_length"] * STEP_MS)
    log.info("  Stride:          %d frames (%.0f ms per output)",
             cfg["stride"], cfg["stride"] * STEP_MS)
    log.info("  Training:        %d steps, batch=%d, lr=%.4f",
             cfg["training_steps"], cfg["batch_size"], cfg["learning_rate"])
    log.info("  Class weights:   pos=%.0f, neg=%.0f",
             cfg["positive_class_weight"], cfg["negative_class_weight"])
    log.info("  Target phrases:  %s", cfg["target_phrases"])
    log.info("=" * 60)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if run_all or args.phase == "setup":
        phase_setup(cfg)
    if run_all or args.phase == "voices":
        phase_voices(cfg)
    if run_all or args.phase == "generate":
        phase_generate(cfg)
    if run_all or args.phase == "features":
        phase_features(cfg)
    if run_all or args.phase == "train":
        phase_train(cfg)
    if run_all or args.phase == "export":
        phase_export(cfg)
    # test is NOT included in 'all' — run explicitly with --phase test
    if args.phase == "test":
        phase_test(cfg)

    if run_all:
        tflite = Path(f"{cfg['model_name']}_mww.tflite")
        log.info("=" * 60)
        log.info("All phases complete!")
        if tflite.exists():
            log.info("  ✓ %s (%.1f KB)", tflite, tflite.stat().st_size / 1024)
        log.info("")
        log.info("Test:")
        log.info("  python3 test_mww.py record/sample*.wav")
        log.info("=" * 60)


if __name__ == "__main__":
    main()
