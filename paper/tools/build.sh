#!/usr/bin/env bash
# Build manuscript figures, PDF, and Word file.
# Usage: bash tools/build.sh   (run from paper/ or repo root)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PAPER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_DIR="$(cd "$PAPER_DIR/.." && pwd)"

cd "$PAPER_DIR"

python_has_module() {
  "$1" -c "import $2" >/dev/null 2>&1
}

if [[ -n "${FIG_PY:-}" ]]; then
  FIG_PY="$FIG_PY"
elif [[ -n "${PYTHON:-}" ]]; then
  FIG_PY="$PYTHON"
elif [[ -x "$REPO_DIR/talk/.venv/bin/python" ]]; then
  FIG_PY="$REPO_DIR/talk/.venv/bin/python"
elif command -v python3 >/dev/null && python_has_module "$(command -v python3)" matplotlib; then
  FIG_PY="$(command -v python3)"
elif command -v python3 >/dev/null; then
  FIG_PY="python3"
else
  FIG_PY="python"
fi

if [[ -n "${DOC_PY:-}" ]]; then
  DOC_PY="$DOC_PY"
elif [[ -n "${PYTHON:-}" ]]; then
  DOC_PY="$PYTHON"
elif command -v python3 >/dev/null && python_has_module "$(command -v python3)" docx; then
  DOC_PY="$(command -v python3)"
elif command -v python3 >/dev/null; then
  DOC_PY="python3"
else
  DOC_PY="python"
fi

if ! command -v pdflatex >/dev/null; then
  export PATH="$PATH:$HOME/Library/TinyTeX/bin/universal-darwin"
fi
if ! command -v pdflatex >/dev/null; then
  echo "ERROR: pdflatex not found. Install TinyTeX or add pdflatex to PATH." >&2
  exit 1
fi

echo "==> Generating manuscript figures"
"$FIG_PY" tools/generate_figures.py

echo "==> Compiling manuscript.pdf"
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex >/dev/null

mkdir -p build
cp manuscript.pdf build/manuscript.pdf
cp highlights.txt build/highlights.txt

if python_has_module "$DOC_PY" docx; then
  echo "==> Building manuscript.docx"
  "$DOC_PY" tools/build_docx.py
  cp build/manuscript.docx manuscript.docx
else
  echo "==> Skipping manuscript.docx (install python-docx, or set DOC_PY to an env that has it)"
fi

echo "==> Cleaning LaTeX intermediates"
rm -f manuscript.aux manuscript.log manuscript.out

echo "Done:"
echo "  $PAPER_DIR/build/manuscript.pdf"
if [[ -f "$PAPER_DIR/build/manuscript.docx" ]]; then
  echo "  $PAPER_DIR/build/manuscript.docx"
fi
echo "  $PAPER_DIR/build/highlights.txt"
