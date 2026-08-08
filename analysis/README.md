# Analysis artifacts

Canonical data and scored outputs for the FQ&P paper
(*Is more thinking always better? Think again*).

| Path | Contents |
|---|---|
| `raw/` | Cooked-ham open dataset (`dataset.xlsx`), questionnaire, manifest |
| `grid/` | Aggregate 27-config results (not per-evaluation primary-grid scores), bootstrap summaries, downstream learners, feature importance |
| `embedding_baselines/` | Separate EmbeddingGemma sensitivity outputs (not the manuscript text-embedding-004 $R^2=0.200$ floor) |
| `topic_level/` | Topic-family per-evaluation LLM scores, JAR validation, related summaries |
| `data_dictionary.md` | Column-level notes for the main tables |

Scripts that read or write these paths live in `../scripts/`.
