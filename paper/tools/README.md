# Manuscript build tools

- `generate_figures.py` — rebuilds core paper figures from `analysis/grid` and
  `analysis/topic_level` using the filenames referenced in `manuscript.tex`
  (`workflow_scoring`, `grid_performance`, `operational_comparison`,
  `modality_hierarchy`, `topic_rank_comparison`). JAR and scoring-example
  figures are separate assets under `paper/figures/`.
- `build.sh` — figures + pdflatex + Word export.
- `build_docx.py` — LaTeX to DOCX helper for FQ&P revision stage.
