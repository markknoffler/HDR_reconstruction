# TriGate-HDR — Paper Assets

This archive contains all figures, LaTeX source, and supporting documents for the TriGate-HDR CVPR paper.

## Contents

### `figures/`
- `architecture_diagram.png` — the full TriGate-HDR pipeline diagram (Figure 1 in the paper), showing Stage 1 (Path-G, generative diffusion), Stage 2 (Path-C, cold diffusion), Stage 3 (Path-S, seaming WGAN), and the GPURE unified energy.

### `latex_source/`
- `main.tex` — the complete LaTeX source for the paper. Compiles cleanly with `pdflatex` + `bibtex` (verified locally: 8 pages, no undefined references after the standard four-pass cycle).
- `references.bib` — the BibTeX bibliography (24 entries, all fact-checked against arXiv / official venue listings).
- `figures/architecture_diagram.png` — local copy of the figure so the LaTeX source compiles standalone without needing to reach into the top-level `figures/` folder.
- `TriGate_HDR_compiled_preview.pdf` — a pre-compiled PDF preview of the paper, generated directly from `main.tex` in this archive, so you can see the final formatted output without compiling it yourself.

### `source_documents/`
The original technical documentation this paper was derived from (for traceability / fact-checking against the actual codebase):
- `model_architecture.md` — authoritative architecture reference (math, tensor flows, GPURE specification).
- `EXPO_BENCHMARK.md` — ExpoCM benchmark protocol and target metrics.
- `GPURE_NOVELTY_CHECK.md` — the novelty analysis and literature map used to position the paper's contributions against prior work.
- `TRIGATE_IMPLEMENTATION_HISTORY.md` — design philosophy and training history (including the v3–v8 Stage 2 debugging log referenced in the paper's Discussion / Limitations section).
- `TRIGATE_PIPELINE_GUIDE.md` — operational training guide and Stage 2 debug log.

## Using the LaTeX source in Overleaf

1. Upload `main.tex` and `references.bib` to a new Overleaf project.
2. Create a `figures/` folder in the Overleaf project and upload `architecture_diagram.png` into it.
3. Compile with the standard pdfLaTeX → BibTeX → pdfLaTeX → pdfLaTeX cycle (Overleaf does this automatically via "Recompile").

See `TriGate_HDR_compiled_preview.pdf` for what the final output should look like.

## Note on experimental results

Tables 1 and 2 in the paper include both finalized values (the published baselines from FHDR, SingleHDR, ArtHDR-Net, HistoHDR-Net, sourced from their respective papers) and placeholders ("TBD" / "--") for TriGate-HDR's own full GPURE joint-training results, which were still in progress at the time of writing per the training logs in `TRIGATE_PIPELINE_GUIDE.md`. Update these cells with final numbers once the `gpure_joint` training runs complete.
