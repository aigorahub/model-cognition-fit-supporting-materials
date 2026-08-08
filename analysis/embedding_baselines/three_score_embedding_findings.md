# Three-score EmbeddingGemma baseline

This note records a focused embedding baseline for the cooked-ham analysis. The goal was to test whether a modern embedding model can produce the same kind of compact input representation used in the main paper: three actual-vs-ideal similarity scores per consumer-product pair, one for visual appearance, one for texture, and one for flavor.

## Research question

Can `google/embeddinggemma-300m` turn each pair of actual and ideal comments into three modality-specific similarity scores that predict liking well enough to compete with direct LLM scoring?

## Data and split

- Source data: `analysis/raw/dataset.xlsx`.
- Rows: 2,758 consumer-product evaluations.
- Consumers: 483.
- Target: 0-10 liking score.
- Features: exactly three similarity scores: visual, texture, flavor.
- Holdout discipline: splits are by consumer, not by row, to avoid consumer-level leakage.
- Local TabPFN and remote AutoGluon split: fixed `GroupShuffleSplit` with random state 42, 20 percent held out.
- Local scikit-learn split: 50 repeated consumer-level `GroupShuffleSplit` evaluations.

The exported input files live in `analysis/embedding_baselines/tabpfn_inputs/`. The folder name is historical: these CSVs are generic ML inputs and were used with local TabPFN and AutoGluon.

## Embedding variants tested

- `raw_sts`: direct STS cosine similarity between actual and ideal comments within each modality.
- `tagged_sts`: same as raw STS, but each comment is prefixed with its modality label.
- `qa_sts`: each comment is framed as an answer to a modality-specific similarity question before embedding.
- `retrieval_doc`: modality-tagged comments embedded with the Retrieval-document prompt.
- `whole_text_one_score`: one whole-comment actual-vs-ideal similarity score, used only in the local scikit-learn comparison.

The Q+A framing is closest to the strategy recommended in the embedding-practice guide: give the embedding model more of the comparison context instead of asking it to infer the task from bare text.

## Local TabPFN results

TabPFN ran locally from `<local-path> on MPS, using the explicit consumer holdout split column.

| Feature set | Model | R2 | MAE | RMSE | Time (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| Q+A STS | TabPFN v2 | 0.170 | 1.731 | 2.164 | 16.1 |
| Tagged STS | TabPFN v2 | 0.108 | 1.804 | 2.244 | 18.1 |
| Raw STS | TabPFN v2 | 0.088 | 1.820 | 2.269 | 18.0 |
| Retrieval-document | TabPFN v2 | 0.057 | 1.847 | 2.306 | 18.0 |

## Remote AutoGluon results

AutoGluon ran through the `tabpfn-launcher` Mini2 path, using the same explicit consumer holdout split column.

| Feature set | Best model | R2 | MAE | RMSE | Pearson r |
| --- | --- | ---: | ---: | ---: | ---: |
| Q+A STS | WeightedEnsemble_L2 | 0.063 | 1.830 | 2.300 | 0.313 |
| Tagged STS | ExtraTreesMSE | 0.009 | 1.872 | 2.364 | 0.237 |
| Raw STS | ExtraTreesMSE | 0.008 | 1.898 | 2.366 | 0.227 |
| Retrieval-document | ExtraTreesMSE | -0.048 | 1.961 | 2.432 | 0.158 |

AutoGluon reports error metrics as negative values internally because higher-is-better metrics are used in its evaluation dictionary. The table above reports absolute MAE and RMSE.

## Local scikit-learn results

The best repeated-split result was Q+A STS with ExtraTrees:

| Feature set | Model | Mean R2 | R2 SD | MAE | RMSE |
| --- | --- | ---: | ---: | ---: | ---: |
| Q+A STS | ExtraTrees | 0.135 | 0.030 | 1.889 | 2.331 |
| Q+A STS | SVR RBF | 0.128 | 0.043 | 1.862 | 2.339 |
| Q+A STS | Gradient Boosting | 0.125 | 0.030 | 1.897 | 2.344 |
| Tagged STS | ExtraTrees | 0.095 | 0.031 | 1.949 | 2.383 |
| Raw STS | ExtraTrees | 0.056 | 0.026 | 1.990 | 2.435 |
| Retrieval-document | ExtraTrees | 0.048 | 0.023 | 2.000 | 2.445 |
| Whole text one score | ExtraTrees | 0.044 | 0.018 | 2.012 | 2.451 |
| Mean baseline | Dummy | -0.004 | 0.006 | 2.065 | 2.512 |

## Interpretation

The embedding baseline improved when the comments were framed around the actual task. Q+A STS was better than raw STS, tagged STS, retrieval-style embeddings, and whole-comment cosine similarity.

Even with that improvement, the embedding-only three-score route is far below the direct LLM scoring result used in the paper and talk. The best local repeated-split scikit-learn result reached mean R2 of about 0.135. Local TabPFN improved the fixed-holdout result to R2 = 0.170 and MAE = 1.73. The direct LLM scoring pipeline reached approximately R2 = 0.573 and MAE = 1.22 on the main held-out evaluation.

This supports the paper's central interpretation. Direct sensory scoring by the language model is not just measuring generic semantic closeness between actual and ideal comments. It appears to be doing a more task-aligned judgment: reading the comments, focusing on the relevant sensory modality, and translating that comparison into a structured similarity score.

## Reproduction pointers

Generate the embedding baselines:

```bash
python scripts/run_three_score_embedding_baseline.py --splits 50
```

Export fixed-split CSVs for remote ML tools:

```bash
python scripts/export_three_score_embedding_features.py
```

Run AutoGluon through `tabpfn-launcher`:

```bash
python scripts/run_remote_autogluon_embedding_sweep.py --launcher-dir "$TABPFN_LAUNCHER_DIR"
```

Run local TabPFN:

```bash
python scripts/run_local_tabpfn_embedding_sweep.py --tabpfn-python "$TABPFN_PYTHON" --runner "$TABPFN_RUNNER"
```

The AutoGluon JSON result files are stored in `analysis/embedding_baselines/autogluon_outputs/`. The local TabPFN result files are stored in `analysis/embedding_baselines/tabpfn_local_outputs/`.
