#!/usr/bin/env bash
# =============================================================================
# setup.sh — one-time environment setup for openWakeWord wake word training
#
# Run this once:
#   chmod +x setup.sh && ./setup.sh
#
# Everything is installed into .venv/ — delete it when done to clean up.
# After setup, use ./run_training.sh instead of python3 directly.
# =============================================================================
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[warn] ${NC} $*"; }
error() { echo -e "${RED}[error]${NC} $*"; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
TRAINING_DIR="$SCRIPT_DIR/training"

# ---------------------------------------------------------------------------
# 0. Verify python3 is available
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install with: sudo apt-get install python3 python3-venv"
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
info "Python $PY_VER detected"

if [[ $PY_MAJOR -lt 3 || ( $PY_MAJOR -eq 3 && $PY_MINOR -lt 9 ) ]]; then
    error "Python 3.9+ required (found $PY_VER)"
fi

# ---------------------------------------------------------------------------
# 1. System packages (apt — does NOT require venv, these are system tools)
# ---------------------------------------------------------------------------
info "Installing system packages..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    # Install python3-venv for whichever minor version is active
    sudo apt-get install -y --no-install-recommends \
        espeak-ng espeak-ng-data libespeak-ng-dev \
        ffmpeg wget git curl build-essential \
        python3-dev python3-venv \
        "python${PY_MAJOR}.${PY_MINOR}-venv" 2>/dev/null \
        || sudo apt-get install -y --no-install-recommends \
               espeak-ng espeak-ng-data libespeak-ng-dev \
               ffmpeg wget git curl build-essential \
               python3-dev python3-venv
else
    warn "apt-get not found — ensure espeak-ng, ffmpeg, git, python3-venv are installed"
fi

# ---------------------------------------------------------------------------
# 2. Clone repositories (system git, no venv needed)
# ---------------------------------------------------------------------------
mkdir -p "$TRAINING_DIR/repos"

info "Cloning piper-sample-generator..."
if [[ ! -d "$TRAINING_DIR/repos/piper-sample-generator" ]]; then
    git clone https://github.com/rhasspy/piper-sample-generator \
        "$TRAINING_DIR/repos/piper-sample-generator"
else
    info "  Already cloned"
fi

info "Cloning openWakeWord..."
if [[ ! -d "$TRAINING_DIR/repos/openwakeword" ]]; then
    git clone https://github.com/dscripka/openWakeWord \
        "$TRAINING_DIR/repos/openwakeword"
else
    info "  Already cloned"
fi

# ---------------------------------------------------------------------------
# 3. Create virtual environment
#    All Python packages go here — delete .venv/ to uninstall everything.
# ---------------------------------------------------------------------------
info "Creating virtual environment at .venv/ ..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    info "  Created: $VENV_DIR"
else
    info "  Already exists: $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

info "Upgrading pip inside venv..."
"$PIP" install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 4. PyTorch — choose CUDA wheel based on driver version
#    nvidia-smi reports the MAX CUDA version the driver supports.
#    CUDA is backward compatible: a driver that supports 13.x can run 12.x toolkits.
# ---------------------------------------------------------------------------
info "Installing PyTorch..."
if command -v nvidia-smi &>/dev/null; then
    # Extract major CUDA version only (e.g. "13.1" → "13")
    CUDA_MAJOR=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+' | head -1 || echo "0")
    info "  Driver max CUDA version: $CUDA_MAJOR.x"

    # Map to best available PyTorch wheel (cu126 is latest stable as of early 2026)
    if   [[ $CUDA_MAJOR -ge 12 ]]; then
        TORCH_TAG="cu126"
    elif [[ $CUDA_MAJOR -ge 11 ]]; then
        TORCH_TAG="cu118"
    else
        warn "  CUDA < 11 — using CPU PyTorch"
        TORCH_TAG="cpu"
    fi

    info "  Trying PyTorch $TORCH_TAG ..."
    "$PIP" install torch torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_TAG}" \
    || {
        warn "  $TORCH_TAG wheel unavailable — falling back to cu124"
        "$PIP" install torch torchaudio \
            --index-url "https://download.pytorch.org/whl/cu124" \
        || {
            warn "  cu124 also unavailable — falling back to CPU"
            "$PIP" install torch torchaudio \
                --index-url "https://download.pytorch.org/whl/cpu"
        }
    }
else
    warn "  No NVIDIA GPU detected — CPU-only PyTorch (training will be slow)"
    "$PIP" install torch torchaudio \
        --index-url "https://download.pytorch.org/whl/cpu"
fi

# ---------------------------------------------------------------------------
# 5. Piper standalone binary (replaces piper-tts + piper-phonemize entirely)
#    piper-phonemize has no Python 3.12 wheel and is difficult to build.
#    The binary bundles its own shared libs + espeak-ng-data and works on
#    any Python version — no pip packages needed for TTS generation.
# ---------------------------------------------------------------------------
PIPER_BIN_DIR="$TRAINING_DIR/piper_binary"
mkdir -p "$PIPER_BIN_DIR"

if [[ ! -f "$PIPER_BIN_DIR/piper/piper" ]]; then
    info "Downloading piper standalone binary (Linux x86_64)..."
    wget -q --show-progress \
        -O "$PIPER_BIN_DIR/piper_linux_x86_64.tar.gz" \
        "https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz"
    tar -xzf "$PIPER_BIN_DIR/piper_linux_x86_64.tar.gz" -C "$PIPER_BIN_DIR"
    rm -f "$PIPER_BIN_DIR/piper_linux_x86_64.tar.gz"
    chmod +x "$PIPER_BIN_DIR/piper/piper"
    info "  piper binary ready: $PIPER_BIN_DIR/piper/piper"
else
    info "  piper binary already present"
fi

# ---------------------------------------------------------------------------
# 6. openWakeWord training dependencies
# ---------------------------------------------------------------------------
info "Installing openWakeWord training dependencies..."
"$PIP" install \
    "mutagen==1.47.0" \
    "torchinfo>=1.8.0" \
    "torchmetrics>=1.0.0" \
    "webrtcvad==2.0.10" \
    "audiomentations>=0.33.0" \
    "torch-audiomentations>=0.11.0" \
    "acoustics>=0.2.6" \
    "pronouncing>=0.2.0" \
    "datasets>=2.14.0" \
    "huggingface_hub>=0.20.0" \
    "PyYAML>=6.0" \
    "scipy>=1.10.0" \
    "numpy>=1.24.0,<2.0" \
    "tqdm>=4.65.0" \
    "librosa>=0.10.0"

"$PIP" install speechbrain \
    || warn "speechbrain install failed — augmentation may be affected"

# ---------------------------------------------------------------------------
# 7. ONNX → TFLite conversion toolchain  (tensorflow-free)
#
# tensorflow-cpu is NOT installed here.  The full TF package has a layered
# incompatibility on Python 3.12 that has no clean pip resolution:
#   • onnx ≥ 1.17 requires protobuf ≥ 7.x
#   • onnx2tf 2.x requires onnx==1.20.1 AND ml-dtypes==0.5.1
#   • tensorflow-cpu 2.18 requires ml-dtypes  0.4.x  (→ conflict with onnx2tf)
#   • tensorflow-cpu 2.19 binary has a sobol_op undefined-symbol crash on WSL2
# Solution: install onnx2tf + tensorflow-cpu for ONNX → TFLite conversion.
# ---------------------------------------------------------------------------
info "Installing ONNX and TFLite conversion tools..."

# onnxscript: required by torch >= 2.10 for torch.onnx.export()
"$PIP" install onnxscript

# tensorflow-cpu + onnx2tf: ONNX → SavedModel → TFLite
"$PIP" install tensorflow-cpu onnx2tf

# torchcodec: torchaudio >= 2.10 uses it as default audio loading backend
"$PIP" install torchcodec

"$PIP" install --upgrade setuptools
PRONOUNCING_INIT="$VENV_DIR/lib/python$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/pronouncing/__init__.py"
if [[ -f "$PRONOUNCING_INIT" ]] && grep -q 'from pkg_resources import resource_stream' "$PRONOUNCING_INIT"; then
    info "  Patching pronouncing/__init__.py for Python 3.12 compatibility..."
    sed -i 's/from pkg_resources import resource_stream/try:\n    from pkg_resources import resource_stream\nexcept ImportError:\n    import importlib.resources as _ir\n    def resource_stream(package, resource):\n        return _ir.files(package).joinpath(resource).open("rb")/' "$PRONOUNCING_INIT"
fi

# ---------------------------------------------------------------------------
# 8. Install openWakeWord in editable mode (into the venv)
# ---------------------------------------------------------------------------
info "Installing openWakeWord (editable install into venv)..."
"$PIP" install -e "$TRAINING_DIR/repos/openwakeword"

# ---------------------------------------------------------------------------
# 9. Patch site-packages for Python 3.12 / latest scipy compatibility
#    These packages use deprecated APIs removed in modern Python/scipy.
# ---------------------------------------------------------------------------
SITE_PKG="$VENV_DIR/lib/python${PY_VER}/site-packages"

# webrtcvad: uses pkg_resources (removed from setuptools default namespace)
if [[ -f "$SITE_PKG/webrtcvad.py" ]] && grep -q '^import pkg_resources' "$SITE_PKG/webrtcvad.py"; then
    info "  Patching webrtcvad.py (pkg_resources → importlib.metadata)..."
    "$PYTHON" -c "
p = '$SITE_PKG/webrtcvad.py'
t = open(p).read()
t = t.replace(
    'import pkg_resources\n',
    'try:\\n    import pkg_resources\\n    __version__ = pkg_resources.get_distribution(\"webrtcvad\").version\\nexcept ImportError:\\n    from importlib.metadata import version as _get_version\\n    __version__ = _get_version(\"webrtcvad\")\\n'
).replace(
    \"__version__ = pkg_resources.get_distribution('webrtcvad').version\\n\", ''
)
open(p, 'w').write(t)
"
fi

# speechbrain: calls torchaudio.list_audio_backends() removed in torchaudio >= 2.10
SB_BACKEND="$SITE_PKG/speechbrain/utils/torch_audio_backend.py"
if [[ -f "$SB_BACKEND" ]] && grep -q 'torchaudio.list_audio_backends()' "$SB_BACKEND"; then
    info "  Patching speechbrain torch_audio_backend.py (list_audio_backends)..."
    sed -i 's/available_backends = torchaudio.list_audio_backends()/available_backends = torchaudio.list_audio_backends() if hasattr(torchaudio, "list_audio_backends") else ["ffmpeg"]/' "$SB_BACKEND"
fi

# acoustics: uses scipy.special.sph_harm renamed to sph_harm_y in scipy >= 1.14
ACOUSTICS_DIR="$SITE_PKG/acoustics/directivity.py"
if [[ -f "$ACOUSTICS_DIR" ]] && grep -q 'from scipy.special import sph_harm' "$ACOUSTICS_DIR"; then
    info "  Patching acoustics/directivity.py (sph_harm → sph_harm_y)..."
    sed -i 's/from scipy.special import sph_harm.*/try:\n    from scipy.special import sph_harm\nexcept ImportError:\n    from scipy.special import sph_harm_y as _sph_harm_y\n    def sph_harm(m, n, theta, phi):\n        return _sph_harm_y(n, m, theta, phi)/' "$ACOUSTICS_DIR"
fi

# openwakeword data.py: add tqdm progress bar to augment_clips batch loop
OWW_DATA="$TRAINING_DIR/repos/openwakeword/openwakeword/data.py"
if [[ -f "$OWW_DATA" ]] && grep -q 'for i in range(0, len(clip_paths), batch_size)' "$OWW_DATA" && ! grep -q 'tqdm' "$OWW_DATA"; then
    info "  Patching openwakeword/data.py (adding tqdm to augment_clips)..."
    # Add tqdm import at the top if missing
    sed -i '1s/^/from tqdm import tqdm\n/' "$OWW_DATA"
    # Wrap the batch loop with tqdm
    sed -i 's/for i in range(0, len(clip_paths), batch_size)/for i in tqdm(range(0, len(clip_paths), batch_size), desc="Augmenting", unit="batch")/' "$OWW_DATA"
fi

# torch_audiomentations: uses torchaudio.info() removed in torchaudio >= 2.10
TAM_IO="$SITE_PKG/torch_audiomentations/utils/io.py"
if [[ -f "$TAM_IO" ]] && grep -q 'torchaudio.info' "$TAM_IO" && ! grep -q 'hasattr.*torchaudio.*info' "$TAM_IO"; then
    info "  Patching torch_audiomentations/utils/io.py (torchaudio.info → soundfile)..."
    "$PYTHON" -c "
p = '$TAM_IO'
old = open(p).read()
old_block = '''        info = torchaudio.info(str(file_path))
        # Deal with backwards-incompatible signature change.
        # See https://github.com/pytorch/audio/issues/903 for more information.
        if type(info) is tuple:
            si, ei = info
            num_samples = si.length
            sample_rate = si.rate
        else:
            num_samples = info.num_frames
            sample_rate = info.sample_rate
        return num_samples, sample_rate'''
new_block = '''        if hasattr(torchaudio, \"info\"):
            info = torchaudio.info(str(file_path))
            if type(info) is tuple:
                si, ei = info
                num_samples = si.length
                sample_rate = si.rate
            else:
                num_samples = info.num_frames
                sample_rate = info.sample_rate
        else:
            import soundfile as sf
            info = sf.info(str(file_path))
            num_samples = info.frames
            sample_rate = info.samplerate
        return num_samples, sample_rate'''
new = old.replace(old_block, new_block)
if new != old:
    open(p, 'w').write(new)
    print('    patched')
else:
    print('    already patched or block not found')
"
fi

# ---------------------------------------------------------------------------
# 10. Smoke-test key imports
# ---------------------------------------------------------------------------
info "Verifying installation..."
"$PYTHON" - << 'PYCHECK'
import sys
ok = True
checks = [
    ("torch",          lambda: __import__("torch").__version__),
    ("numpy",          lambda: __import__("numpy").__version__),
    ("onnx",           lambda: __import__("onnx").__version__),
    ("yaml",           lambda: __import__("yaml").__version__),
    ("datasets",       lambda: __import__("datasets").__version__),
    ("openwakeword",   lambda: __import__("openwakeword") and "ok"),
]
for name, fn in checks:
    try:
        ver = fn()
        print(f"  ✓  {name:<20} {ver}")
    except Exception as e:
        print(f"  ✗  {name:<20} MISSING ({e})")
        ok = False

# Optional but important
for name in ("onnx2torch", "ai_edge_torch"):
    try:
        m = __import__(name)
        ver = getattr(m, "__version__", "ok")
        print(f"  ✓  {name:<20} {ver}")
    except (ImportError, RuntimeError) as e:
        print(f"  !  {name:<20} not available ({e}) — tflite conversion may fall back")

import torch
print(f"\n  PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

if not ok:
    print("\nWARNING: some packages are missing — check errors above.")
    sys.exit(1)
PYCHECK

# Check piper binary separately (bash, not Python)
if [[ -f "$TRAINING_DIR/piper_binary/piper/piper" ]]; then
    info "  ✓  piper binary         ready"
else
    warn "  ✗  piper binary         NOT FOUND at $TRAINING_DIR/piper_binary/piper/piper"
    warn "     TTS clip generation will fail. Re-run setup.sh to retry the download."
fi

# ---------------------------------------------------------------------------
# 10. Write run_training.sh convenience wrapper
# ---------------------------------------------------------------------------
cat > "$SCRIPT_DIR/run_training.sh" << RUNSH
#!/usr/bin/env bash
# Activate the project venv and run train_wakeword.py.
# Usage:  ./run_training.sh --config <config.yaml> [--phase PHASE] [--skip-features]
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/.venv/bin/activate"
exec python "\$SCRIPT_DIR/train_wakeword.py" "\$@"
RUNSH
chmod +x "$SCRIPT_DIR/run_training.sh"

# ---------------------------------------------------------------------------
echo ""
info "Setup complete!"
echo ""
echo "Usage:"
echo "  ./run_training.sh --config ru_jarvis.yaml                # run all phases"
echo "  ./run_training.sh --config ru_jarvis.yaml --phase voices # individual phase"
echo "  ./run_training.sh --config ru_jarvis.yaml --skip-features # skip ACAV100M download"
echo ""
echo "To clean up after training:"
echo "  rm -rf .venv training/"
echo ""
echo "Estimated disk usage:"
echo "  .venv (all Python packages) : ~5 GB"
echo "  Piper voices (4 × ~70 MB)   : ~280 MB"
echo "  ACAV100M feature file       : ~7 GB"
echo "  Background audio            : ~5 GB"
echo "  Generated TTS clips         : ~8 GB"
echo "  Total                       : ~26 GB"
echo ""
echo "Estimated runtime:"
echo "  TTS clip generation (CPU)   : 6-10 hours"
echo "  Augmentation                : 2-4 hours"
echo "  Training      (GPU)         : 1-3 hours | (CPU): 8-12 hours"
echo "  Total         (GPU)         : ~10 hours | (CPU): ~22 hours"
