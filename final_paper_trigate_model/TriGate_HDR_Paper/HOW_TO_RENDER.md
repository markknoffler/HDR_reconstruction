# How to Render the TriGate-HDR Paper (locally, no Overleaf)

This paper is a plain LaTeX project that compiles with a standard local TeX toolchain.
No Overleaf, no internet, no special packages beyond a normal TeX Live install.

---

## TL;DR

```bash
cd final_paper_trigate_model/TriGate_HDR_Paper
./build.sh
# open the result:
xdg-open build/main.pdf     # Linux
```

That's it. `build/main.pdf` is the paper.

---

## 1. One-time setup (install a TeX toolchain)

You need `pdflatex` and `bibtex`. On your machine they are already installed
(`/usr/bin/pdflatex`, `/usr/bin/bibtex`). If you ever move to a fresh machine:

**Ubuntu / Debian**
```bash
sudo apt-get update
sudo apt-get install -y texlive-latex-base texlive-latex-recommended \
                        texlive-latex-extra texlive-fonts-recommended
```

**Check it works**
```bash
pdflatex --version
bibtex --version
```

> Note: this project deliberately does **not** depend on the `algorithm`/`algorithmic`
> packages (they are not in the base TeX Live install). The two algorithm blocks are
> typeset with a small self-contained macro defined in `main.tex`, so nothing extra
> is required.

---

## 2. Build the PDF

### Option A — the provided script (recommended)
```bash
cd final_paper_trigate_model/TriGate_HDR_Paper
./build.sh
```
`build.sh` runs the standard four-pass cycle so cross-references and citations resolve:
```
pdflatex → bibtex → pdflatex → pdflatex
```
Output: **`build/main.pdf`** (all auxiliary files stay in `build/`).

### Option B — by hand
```bash
cd final_paper_trigate_model/TriGate_HDR_Paper
pdflatex -output-directory=build main.tex
bibtex   build/main
pdflatex -output-directory=build main.tex
pdflatex -output-directory=build main.tex
```

### Option C — latexmk (if installed)
```bash
latexmk -pdf -output-directory=build main.tex
```

---

## 3. View the PDF

```bash
xdg-open build/main.pdf      # Linux (default viewer)
evince   build/main.pdf      # GNOME
okular   build/main.pdf      # KDE
```

---

## 4. Regenerating the figures (when results change)

The result figures are generated from the metrics CSV so they always match the data:

```bash
cd final_paper_trigate_model/TriGate_HDR_Paper
python3 scripts/make_figures.py     # needs matplotlib, numpy, pandas
```

This reads `data/benchmark_metrics.csv` and writes the vector PDFs in `figures/`.
After regenerating, rebuild the paper with `./build.sh`.

To point the figures at a fresh run, copy the new CSV over `data/benchmark_metrics.csv`
and set `SELECTED_EPOCH` at the top of `scripts/make_figures.py` to the chosen checkpoint.

---

## 5. Optional: preview pages as images

Useful for a quick look without a PDF viewer:
```bash
mkdir -p build/preview
pdftoppm -png -r 120 build/main.pdf build/preview/p    # needs poppler-utils
```

---

## 6. Switching to an official conference style

The source uses a standalone CVPR-style skeleton so it compiles anywhere. To use the
**official** CVPR/ICCV kit instead:

1. Drop `cvpr.sty` and `ieeenat_fullname.bst` (from the `cvpr-org/author-kit` repo)
   into this directory.
2. Change the first line of `main.tex` to `\usepackage[review]{cvpr}` style per the kit
   and switch the bibliography style to `ieeenat_fullname`.
3. Rebuild with `./build.sh`.

For ICLR/ICML, drop in their `.sty` (e.g. `iclr2026_conference.sty`) and adjust
`\documentclass` accordingly.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `File 'algorithm.sty' not found` | You edited the preamble; this project doesn't use it. Restore the self-contained macro block, or `sudo apt-get install texlive-science`. |
| Citations show as `[?]` | Run the full cycle (`./build.sh`); a single `pdflatex` pass isn't enough. |
| `figures/*.pdf` missing | Run `python3 scripts/make_figures.py` first. |
| Old references linger | Delete `build/` and rebuild: `rm -rf build && ./build.sh`. |
| Fonts look wrong | Install `texlive-fonts-recommended` (provides Times via the `times` package). |
