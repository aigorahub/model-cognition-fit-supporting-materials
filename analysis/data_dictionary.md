# Data dictionary

Paths are relative to `analysis/`.

## `raw/dataset.xlsx`

Original cooked-ham workbook used for the analysis.

Relevant sheets:

- `product sensory properties`: product-level consumer evaluations, Free-Comment descriptions and liking scores.
- `consumer questionnaire (home)`: consumer-level ideal ham descriptions used to pair each actual product evaluation with the same consumer's ideal.

Core columns used by the scripts:

- `Consumer`: consumer identifier.
- `Product`: product identifier.
- `Liking`: 0 to 10 liking score.
- `DescriptionVisual`: actual-product visual Free-Comment.
- `DescriptionTexture`: actual-product texture Free-Comment.
- `DescriptionFlavor`: actual-product flavor Free-Comment.
- `IdealVisual`: ideal-product visual Free-Comment.
- `IdealTexture`: ideal-product texture Free-Comment.
- `IdealFlavor`: ideal-product flavor Free-Comment.

## `grid/`

Saved outputs from the model-grid comparison. The primary grid crossed model
family, Likert scale granularity, and generation temperature (27 configurations).
Also includes bootstrap summaries, family win probabilities, and downstream
learner comparison tables cited in the manuscript.

## `embedding_baselines/`

EmbeddingGemma three-score actual-versus-ideal baseline (raw, tagged, Q+A, and
retrieval-document framings), plus local TabPFN and remote AutoGluon outputs.
See `three_score_embedding_findings.md`.

## `topic_level/`

Topic-level LLM alignment scores for 17 sensory topic families, MR-CA comparison
artifacts, and JAR validation summaries.
