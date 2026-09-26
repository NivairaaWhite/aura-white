#!/usr/bin/env sh
# Linux / macOS installer:  ./install.sh   or   ./install.sh cu126   (NVIDIA build of PyTorch)
# Add "studio" to also install the texturing extras:  ./install.sh cu126 studio
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
STUDIO=0
for a in "$@"; do [ "$a" = "studio" ] && STUDIO=1; done
[ -n "$1" ] && [ "$1" != "studio" ] && pip install torch --index-url "https://download.pytorch.org/whl/$1"
pip install -e ".[recommended]" || pip install -e .
[ "$STUDIO" = "1" ] && { pip install -r requirements-studio.txt || echo "some optional extras failed - built-in fallbacks are used"; }
python -m aura_white doctor
echo "Next:  .venv/bin/aura-white serve     |     .venv/bin/aura-white studio art.png"
