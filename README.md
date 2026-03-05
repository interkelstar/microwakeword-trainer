# openwakeword-trainer-ru

Train custom wake word models for [openWakeWord](https://github.com/dscripka/openWakeWord) using [Piper](https://github.com/rhasspy/piper) TTS — no microphone recordings needed.

The toolkit generates thousands of synthetic speech clips, augments them with real-world noise and room impulse responses, and trains a small DNN that runs in real time on a Raspberry Pi or any device supported by openWakeWord.

## Included examples

| Config | Wake word | Language |
|--------|-----------|----------|
| `ru_jarvis.yaml` | Джарвис | Russian |
| `ru_stop.yaml` | Стоп | Russian |

## Prerequisites

- **Linux x86_64** (tested on Ubuntu 22.04 / Debian 12, WSL2 works)
- **Python 3.9+**
- **~30 GB free disk** (virtual environment + training data + generated clips)
- **GPU optional** — CUDA speeds up training (1–3 h vs 8–12 h on CPU); TTS generation is CPU-bound either way

## Quick start

```bash
# 1. One-time setup: creates .venv/, downloads repos, installs dependencies
chmod +x setup.sh && ./setup.sh

# 2. Train a wake word (runs all phases sequentially)
./run_training.sh --config ru_jarvis.yaml

# 3. Output: ru_jarvis.onnx + ru_jarvis.tflite in the project root
```

## Usage

```bash
# Run all phases
./run_training.sh --config ru_jarvis.yaml

# Run a specific phase
./run_training.sh --config ru_jarvis.yaml --phase train

# Skip the large ACAV100M feature download (if already present)
./run_training.sh --config ru_jarvis.yaml --skip-features

# Preview TTS output (generate 5 clips and save to preview/)
./run_training.sh --config ru_jarvis.yaml --preview 5
```

### Phases

| Phase | What it does | Time estimate |
|-------|-------------|---------------|
| `setup` | Writes the Piper generate_samples wrapper | seconds |
| `voices` | Downloads Piper ONNX voice models from HuggingFace | 1–5 min |
| `features` | Downloads ACAV100M + validation feature files (~7 GB) | 10–30 min |
| `background` | Downloads MIT RIRs, AudioSet, FMA for noise augmentation | 10–30 min |
| `generate` | Generates positive + negative TTS clips | 6–10 h (CPU) |
| `augment` | Augments clips with noise, RIR, speed/pitch variation | 2–4 h |
| `train` | Trains the DNN classifier | 1–3 h (GPU) / 8–12 h (CPU) |
| `export` | Converts ONNX → TFLite | seconds |

## Creating a config for a new wake word

Copy one of the example configs and modify:

```yaml
model_name: my_wake_word        # output filename (ASCII, no spaces)

target_phrases:                 # what the model should activate on
  - My Wake Word
  - My Wake Word!
  - My Wake Word?

training:
  n_samples: 50000              # positive TTS clips (25k minimum)
  n_samples_val: 10000
  steps: 500000                 # gradient steps (increase if underfitting)
  max_neg_weight: 3000          # false-positive penalty (increase if too many FPs)

voices:
  primary:                      # must match target phrase language
    base_url: https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US
    models:
      en_US-amy-medium: amy/medium
      en_US-john-medium: john/medium

negative_phrases:               # acoustically similar words that should NOT trigger
  - My Cake Bird
  - Buy Lake Word
  # ... 50–100 phrases recommended
```

Browse available voices at [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices).

## How it works

1. **TTS generation** — Piper synthesizes the target phrases (and adversarial negatives) across multiple voices with varying intonation
2. **Augmentation** — clips are mixed with background audio (AudioSet, FMA), convolved with room impulse responses (MIT RIRs), and perturbed with speed/pitch changes
3. **Feature extraction** — audio is converted to mel-spectrogram features matching openWakeWord's input format
4. **Training** — a small fully-connected network learns to distinguish the wake word from background speech and similar-sounding words
5. **Export** — the trained model is exported to ONNX and converted to TFLite for on-device deployment

## Cleanup

```bash
rm -rf .venv training/    # removes ~33 GB of intermediate files
```

The trained `.onnx` and `.tflite` files in the project root are your final models — copy them before cleaning up.

## Credits

- [openWakeWord](https://github.com/dscripka/openWakeWord) by David Scripka — the wake word engine and training framework
- [Piper](https://github.com/rhasspy/piper) by rhasspy — fast, local neural TTS
- [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) — batch TTS clip generation

## License

[MIT](LICENSE)
