#!/usr/bin/env bash
# =============================================================================
# setup_mww.sh — One-time environment setup for microWakeWord training
#
# Run this once before your first training run:
#   chmod +x setup_mww.sh && ./setup_mww.sh
#
# What this script does:
#   1. Installs system packages (ffmpeg, wget, git, python3-venv)
#   2. Creates a Python virtualenv at .venv/
#   3. Installs microWakeWord from GitHub (brings TensorFlow + kws_streaming)
#   4. Installs pymicro-features, scipy, huggingface_hub, and other deps
#   5. Downloads the piper standalone binary for TTS clip generation
#   6. Writes run_mww.sh convenience wrapper
#
# Everything is installed into .venv/ — delete it to clean up.
# After setup, use ./run_mww.sh instead of calling python3 directly.
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
# 0. Check Python
# ---------------------------------------------------------------------------
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install with: sudo apt-get install python3 python3-venv"
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PY_VER detected"

if [[ $(echo "$PY_VER" | cut -d. -f1) -lt 3 ]] || \
   [[ $(echo "$PY_VER" | cut -d. -f1) -eq 3 && $(echo "$PY_VER" | cut -d. -f2) -lt 9 ]]; then
    error "Python 3.9+ required (found $PY_VER)"
fi

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
info "Checking system packages..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        ffmpeg wget git curl build-essential \
        python3-dev python3-venv 2>/dev/null || true
else
    warn "apt-get not found — ensure ffmpeg, wget, git, python3-venv are installed"
fi

# ---------------------------------------------------------------------------
# 2. Create virtual environment
# ---------------------------------------------------------------------------
info "Creating virtual environment..."
if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    info "  Created: $VENV_DIR"
else
    info "  Already exists: $VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PYTHON="$VENV_DIR/bin/python"

"$PIP" install --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# 3. Install microWakeWord from GitHub (brings TensorFlow + dependencies)
# ---------------------------------------------------------------------------
info "Installing microWakeWord..."
"$PIP" install "git+https://github.com/kahrendt/microWakeWord.git" \
    || error "Failed to install microWakeWord. Check the GitHub repo."

# ---------------------------------------------------------------------------
# 4. Install additional dependencies
# ---------------------------------------------------------------------------
info "Installing additional dependencies..."
"$PIP" install \
    "pymicro-features>=2.0" \
    "scipy>=1.10.0" \
    "tqdm>=4.65.0" \
    "huggingface_hub>=0.20.0" \
    "PyYAML>=6.0" \
    "numpy>=1.24.0,<2.0"

# ---------------------------------------------------------------------------
# 5. Download Piper standalone binary (for TTS clip generation)
# ---------------------------------------------------------------------------
PIPER_BIN_DIR="$TRAINING_DIR/piper_binary"
mkdir -p "$PIPER_BIN_DIR"

if [[ ! -f "$PIPER_BIN_DIR/piper/piper" ]]; then
    info "Downloading Piper binary (Linux x86_64)..."
    wget -q --show-progress \
        -O "$PIPER_BIN_DIR/piper_linux_x86_64.tar.gz" \
        "https://github.com/rhasspy/piper/releases/latest/download/piper_linux_x86_64.tar.gz"
    tar -xzf "$PIPER_BIN_DIR/piper_linux_x86_64.tar.gz" -C "$PIPER_BIN_DIR"
    rm -f "$PIPER_BIN_DIR/piper_linux_x86_64.tar.gz"
    chmod +x "$PIPER_BIN_DIR/piper/piper"
    info "  Piper binary ready: $PIPER_BIN_DIR/piper/piper"
else
    info "  Piper binary already present"
fi

# ---------------------------------------------------------------------------
# 6. Smoke-test imports
# ---------------------------------------------------------------------------
info "Verifying installation..."
"$PYTHON" - << 'PYCHECK'
import sys
ok = True
checks = [
    ("tensorflow",       lambda: __import__("tensorflow").__version__),
    ("microwakeword",    lambda: __import__("microwakeword") and "ok"),
    ("pymicro_features", lambda: __import__("pymicro_features") and "ok"),
    ("numpy",            lambda: __import__("numpy").__version__),
    ("yaml",             lambda: __import__("yaml").__version__),
    ("scipy",            lambda: __import__("scipy").__version__),
    ("tqdm",             lambda: __import__("tqdm").__version__),
]
for name, fn in checks:
    try:
        ver = fn()
        print(f"  ✓  {name:<20} {ver}")
    except Exception as e:
        print(f"  ✗  {name:<20} MISSING ({e})")
        ok = False

if not ok:
    print("\nWARNING: some packages are missing — check errors above.")
    sys.exit(1)
PYCHECK

# Check piper binary
if [[ -f "$TRAINING_DIR/piper_binary/piper/piper" ]]; then
    info "  ✓  piper binary         ready"
else
    warn "  ✗  piper binary         NOT FOUND"
fi

# ---------------------------------------------------------------------------
# 7. Write run_mww.sh wrapper
# ---------------------------------------------------------------------------
cat > "$SCRIPT_DIR/run_mww.sh" << RUNSH
#!/usr/bin/env bash
# Activate venv and run the standalone microWakeWord trainer.
# Usage:  ./run_mww.sh --config <config.yaml> [--phase PHASE]
set -euo pipefail
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
source "\$SCRIPT_DIR/.venv/bin/activate"
exec python "\$SCRIPT_DIR/train_mww.py" "\$@"
RUNSH
chmod +x "$SCRIPT_DIR/run_mww.sh"

# ---------------------------------------------------------------------------
echo ""
info "Setup complete!"
echo ""
echo "Usage:"
echo "  ./run_mww.sh --config ru_jarvis.yaml                # run all phases"
echo "  ./run_mww.sh --config ru_jarvis.yaml --phase voices  # individual phase"
echo ""
echo "Phases: setup → voices → generate → features → train → export"
echo ""
echo "To clean up:  rm -rf .venv training/"
echo ""
echo "Estimated runtime:"
echo "  TTS generation (CPU) : 6-10 hours (50k clips)"
echo "  Feature extraction   : 1-2 hours"
echo "  Training (10k steps) : 0.5-2 hours"
echo "  Total                : ~8-14 hours"
