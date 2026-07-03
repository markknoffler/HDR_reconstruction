#!/usr/bin/env bash
# Local LaTeX build for the TriGate-HDR paper (no Overleaf).
# Four-pass cycle so \ref and \cite resolve. Output: build/main.pdf
set -e
cd "$(dirname "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
bibtex build/main || true
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
echo "==> build/main.pdf"
