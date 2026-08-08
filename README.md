# Supporting materials: model-cognition fit in AI-supported sensory analysis

This repository holds **supporting materials for one paper**:

> Ennis, Worch, and Mahieu. *Is more thinking always better? Think again:
> Model-cognition fit in AI-supported sensory analysis.* Food Quality and
> Preference manuscript **FQAP-D-26-01039**.

It is not a general software product, product ranking, or multi-project monorepo.
Use it to audit the analyses reported in that manuscript.

## Layout

| Path | Contents |
|------|----------|
| `analysis/raw/` | Open cooked-ham dataset, questionnaire, manifest |
| `analysis/grid/` | Aggregate 27-configuration results, bootstrap summaries, downstream learners, feature importance |
| `analysis/topic_level/` | Per-evaluation topic alignment scores, JAR validation, related summaries |
| `analysis/embedding_baselines/` | Separate EmbeddingGemma sensitivity work (not the manuscript $R^2=0.200$ text-embedding-004 floor) |
| `scripts/` | Local analysis helpers used for the paper |
| `notebooks/sensory_grid_comparison_with_temperature.ipynb` | Primary scoring-grid notebook (15 January 2026 run) |
| `paper/` | Manuscript sources, greyscale figures, figure generators |

## Audit archived results (no API)

Use the CSVs under `analysis/` as the durable record. Configuration-level primary
grid numbers are in `analysis/grid/`. Topic-level per-evaluation scores are in
`analysis/topic_level/topic_level_*_scores.csv`.

## Regenerate figures

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install matplotlib pandas numpy scipy openpyxl
python paper/tools/generate_figures.py
python scripts/compare_models_jar.py
python scripts/validate_llm_vs_jar.py
```

## Build the manuscript PDF

```bash
cd paper
tectonic manuscript.tex
```

## Optional live scoring

Topic-level and related scripts can call a Gemini API if you supply credentials
(`GEMINI_API_KEY` or `--keypool-env`). That path may cost money, aliases can
change, and a rerun today will not reproduce the 15 January 2026 responses
exactly. Prefer the archived tables for audit.

## What is not here

- Peer-review correspondence or internal revision packets
- Credentials or raw vendor response dumps
- Per-evaluation score matrices for all 27 primary-grid cells (aggregate tables only)
- Private cluster launchers and unrelated poster projects

## License

Code and analysis scripts: MIT (see LICENSE).
Third-party data remain under their original terms (Visalli et al. 2024 Data in
Brief open dataset).
