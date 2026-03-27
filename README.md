# microWakeWord Trainer

Config-driven pipeline for training custom wake word models using the
[microWakeWord](https://github.com/kahrendt/microWakeWord) framework.

microWakeWord trains a MixedNet/Inception streaming CNN that is exported as
a fully quantised TFLite model.  The final model is 50–300 KB and is designed
to run with [pymicro_wakeword](https://github.com/kahrendt/pymicro_wakeword)
(used by [linux-voice-assistant](https://github.com/your-lva-repo)) and
ESPHome on microcontrollers.

## Background

I built this because I wanted a custom Russian wake word — "Джарвис" (Jarvis)
— for my home voice assistant.  Russian isn't supported by any of the
off-the-shelf wake word solutions I found, and recording hundreds of your own
voice clips felt like the wrong approach.  I wanted a fully synthetic training
pipeline: pick your phrases, pick your voices, run the script, get a model.

I started with [openWakeWord](https://github.com/interkelstar/openwakeword-trainer)
and got something working, but the gap between training accuracy and real-world
recall was frustrating.  Switching to microWakeWord's streaming CNN architecture
gave much better results, and the model is a fraction of the size.

The pipeline is fully language-agnostic — the Russian example configs are just
that.  Any language with a [Piper](https://github.com/rhasspy/piper) voice
model should work the same way.

### Results on Russian "Джарвис"

Trained on 50k TTS clips across 4 Russian voices (Dmitri, Denis, Irina, Ruslan),
evaluated on 50 real recordings of the same person saying "Джарвис" naturally:

| Metric | Result |
|--------|--------|
| Recall — 50 real recordings | **50 / 50 (100%)** |
| Avg confidence on real speech | **0.996** |
| Silent room, 5 minutes of ambient noise | **No trigger** (max avg 0.087) |
| Production cutoff in daily use | **0.97** |
| False positives after FP-mining retraining | ~2–3 / day (English TV) |
| Model size | **82.5 KB** (streaming, uint8 quantised) |
| Val accuracy | **99.77%** |

The model was deployed and run continuously.  After one round of collecting
66 real false-positive clips from daily use and retraining, the ambient
false-positive score dropped from 0.385 to 0.087.  The remaining false
positives are English words with a /dʒ/ onset (just, defence, ideas) —
a second round of FP-mining should eliminate most of those.

---

## Requirements

- Linux (tested on Ubuntu 22.04, Debian 12, WSL2)
- Python 3.9+ with pip
- ~15 GB disk space for training data
- NVIDIA GPU strongly recommended (training takes 8–14 hours on CPU)
- Internet access for HuggingFace downloads

## Quick Start

```bash
# Clone the repo and set up the environment (one time only)
git clone https://github.com/interkelstar/microwakeword-trainer.git
cd microwakeword-trainer
chmod +x setup_mww.sh && ./setup_mww.sh

# Copy the example config and customise it for your wake word
cp example_ru_jarvis.yaml my_wakeword.yaml
# Edit my_wakeword.yaml: set model_name, target_phrases, voices, negatives

# Run the full training pipeline
./run_mww.sh --config my_wakeword.yaml
```

Output files created at the project root after training:

- `<model_name>_mww.tflite` — TFLite model ready for pymicro_wakeword
- `<model_name>_mww.json` — Deployment manifest with inference parameters

## Training Phases

The pipeline runs these phases in order:

| Phase      | What it does                                                        | Time (CPU)  |
|------------|---------------------------------------------------------------------|-------------|
| `setup`    | Verify microwakeword, TF, pymicro_features are installed            | seconds     |
| `voices`   | Download Piper ONNX voice models from HuggingFace                   | 1–5 min     |
| `generate` | Synthesise positive + adversarial negative TTS clips via piper      | 6–10 hours  |
| `features` | Extract pymicro_features spectrograms + download HF negatives       | 1–3 hours   |
| `train`    | Build and train the MixedNet model with SpecAugment                 | 0.5–3 hours |
| `export`   | Convert to streaming TFLite (uint8 quantised) + write JSON manifest | 5–15 min    |
| `test`     | Evaluate TFLite model on test clips (optional, not run in "all")    | minutes     |

Run individual phases with `--phase <name>`:

```bash
./run_mww.sh --config my_wakeword.yaml --phase train
./run_mww.sh --config my_wakeword.yaml --phase export
./run_mww.sh --config my_wakeword.yaml --phase test
```

## Config Reference

See `example_ru_jarvis.yaml` for a fully annotated example. Key sections:

```yaml
model_name: ru_jarvis        # Output filename prefix (ASCII, no spaces)

target_phrases:              # What the model should fire on
  - Джарвис
  - Джарвис!
  - Джарвис?

training:
  n_samples: 50000           # TTS clips for training
  n_samples_val: 10000       # TTS clips for validation

voices:
  primary:                   # For positive clips + same-language negatives
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU
    models:
      ru_RU-dmitri-medium: dmitri/medium
  secondary:                 # For cross-language negative phrases (optional)
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main
    models:
      en_US-amy-medium: en/en_US/amy/medium

negative_phrases:            # Acoustically similar words to reject
  - Джар
  - Алиса
  - Jarring                  # Latin script → synthesised with secondary voices

microwakeword:
  architecture: mixednet     # mixednet (default) or inception
  spectrogram_length: 204    # input frames per inference window
  stride: 3                  # frames consumed per streaming step (3 × 10ms = 30ms)
  training_steps: 10000      # gradient steps
  batch_size: 128
  learning_rate: 0.001
  positive_class_weight: 1.5 # raise if recall is too low
  negative_class_weight: 1   # raise if false positive rate is too high
  probability_cutoff: 0.9    # detection threshold (used at inference time)
  sliding_window_size: 10    # averaging window size (used at inference time)
```

### Understanding spectrogram_length and stride

microWakeWord uses pymicro_features to extract 40-bin mel spectrogram frames
at 10 ms intervals.  The model processes `stride` frames per inference call
and maintains a ring buffer of internal state.

The `spectrogram_length` determines how many frames the model can look back
when making each decision.  Larger values give more context but increase model
size and training memory.

`probability_cutoff` and `sliding_window_size` are used only at inference
time — re-run `--phase export` after changing them.

## Deployment

Copy both output files to your linux-voice-assistant config directory:

```bash
cp ru_jarvis_mww.tflite /path/to/linux-voice-assistant/models/
cp ru_jarvis_mww.json   /path/to/linux-voice-assistant/models/
```

The JSON manifest format:

```json
{
  "type": "micro",
  "wake_word": "Джарвис",
  "model": "ru_jarvis_mww.tflite",
  "micro": {
    "probability_cutoff": 0.9,
    "sliding_window_size": 10,
    "feature_step_size": 10
  }
}
```

The `type: micro` field is required — the linux-voice-assistant loader uses it
to select the pymicro_wakeword inference backend.

To change the detection threshold without retraining, edit `probability_cutoff`
in the JSON file directly, or change it in your YAML and re-run `--phase export`.

## Iterative Improvement — Reducing False Positives

The most effective way to reduce false positives is to collect negative examples
from your real environment and retrain.

**Workflow:**

1. Run the assistant for a day and note when it triggers incorrectly.
2. Record the audio that caused false positives as a WAV file (16 kHz, mono).
3. Copy the WAV to `training/output/<model_name>/user_negatives/`.
4. Re-run feature extraction and training:

```bash
./run_mww.sh --config my_wakeword.yaml --phase features
./run_mww.sh --config my_wakeword.yaml --phase train
./run_mww.sh --config my_wakeword.yaml --phase export
```

The `user_negatives/` directory is processed with dense windowing (1 window
per 10 ms) to maximise coverage of the triggering audio.

**Important:** Do NOT put actual wake word recordings in `user_negatives/`.
Only put audio that triggered incorrectly (TV speech, background noise, etc.).

## Testing a Model

Use `test_mww.py` to evaluate a trained model against WAV files.  The script
simulates the exact pymicro_wakeword inference loop including the sliding
window averaging:

```bash
# Test against all WAV files in record/
.venv/bin/python3 test_mww.py

# Test specific files
.venv/bin/python3 test_mww.py record/sample1.wav record/sample2.wav

# Output shows per-frame probabilities and a TRIGGERED / no trigger result
```

The model and config paths are hard-coded at the top of `test_mww.py` —
edit `MODEL_PATH` and `CONFIG_PATH` if you use a different filename.

## ElevenLabs High-Quality Positives (Optional)

`generate_elevenlabs.py` generates additional high-quality TTS clips using the
ElevenLabs API (requires an API key with credits).  These are automatically
picked up during feature extraction and can improve the naturalness of the
positive class:

```bash
.venv/bin/python generate_elevenlabs.py --api-key YOUR_KEY --config my_wakeword.yaml
```

Clips are saved to `training/output/<model_name>/elevenlabs_positive/`.

## Disk Space

After a full training run, the `training/` directory contains:

```
training/piper_binary/           # piper executable + shared libs (~150 MB)
training/piper_models/           # ONNX voice models (~280 MB for 4 voices)
training/output/<model_name>/
  positive_train/                # TTS positive clips (~4 GB for 50k)
  positive_test/                 # TTS validation clips (~800 MB for 10k)
  negative_train/                # Adversarial negative TTS clips
  negative_test/
  elevenlabs_positive/           # ElevenLabs clips (optional)
  user_negatives/                # Real environment negatives (optional)
  mww_features/                  # Extracted spectrograms + HF negatives
  model/                         # Saved Keras model weights
```

To clean up after training:

```bash
rm -rf .venv training/
```

## How It Works

microWakeWord uses the pymicro_features frontend (matching the on-device
inference pipeline on microcontrollers):

1. Audio is chunked into 10 ms frames of 160 samples at 16 kHz.
2. Each chunk is processed by `MicroFrontend` from the `pymicro-features`
   package, producing a 40-bin log-mel spectrogram frame.
3. The streaming model processes `stride` frames per call, maintaining internal
   convolutional state across calls.
4. A sliding window of `sliding_window_size` consecutive probabilities is
   averaged; when the mean exceeds `probability_cutoff`, the wake word fires.

**Feature scale:** pymicro_features outputs values in [0, ~26] (float32).
This range must match at training time and inference time.  The HuggingFace
negative dataset (`kahrendt/microwakeword`) uses a uint16 scale and is
automatically rescaled to match during feature extraction.

## License

MIT
